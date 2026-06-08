import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Polygon
import logging

from .lcc import LabelConnectedComp

logger = logging.getLogger(__name__)


class NPZAHNFuser:
    """
    NPZ-based AHN fuser compatible with Pipeline.
    Uses grid-based connected component filtering instead of DBSCAN for speed.

    Parameters
    ----------
    grow_facade : bool (default False)
        Only active when ``target='building'``.  After the initial AHN-based
        building label, runs a 3-D voxel LCC to grow the label into overhanging
        facade elements (balconies, awnings, cornices) that lie above
        ``facade_floor`` and are geometrically connected to already-labeled
        building points.  Avoids merging street-level furniture (benches, bikes)
        because those are below the floor cutoff.
    facade_floor : float (default 1.5 m)
        Height above ground below which points are excluded from the LCC
        container.  Set to ≥ 1.5 m to keep benches and bicycles out.
    facade_grid_size : float (default 0.25 m)
        Voxel size for the LCC connectivity check.
    facade_min_comp : int (default 20)
        Minimum component size (points) to be included in the grow.
    """

    TARGETS = ('ground', 'building')

    def __init__(self, label, npz_reader, target='ground', epsilon=0.2,
                 grid_size=0.4, min_comp_size=20,
                 grow_facade=False, facade_floor=1.5,
                 facade_grid_size=0.25, facade_min_comp=20):
        if target not in self.TARGETS:
            raise ValueError(f"Target must be one of {self.TARGETS}")
        self.label = label
        self.npz_reader = npz_reader
        self.target = target
        self.epsilon = epsilon
        self.grid_size = grid_size
        self.min_comp_size = min_comp_size
        self.grow_facade = grow_facade and (target == 'building')
        self.facade_floor = facade_floor
        self.facade_grid_size = facade_grid_size
        self.facade_min_comp = facade_min_comp
        self._load_surface()

    def _load_surface(self):
        data = np.load(self.npz_reader)
        self.x = data['x']
        self.y = data['y']
        self.z = data[self.target]

        self.interpolator = RegularGridInterpolator(
            (self.y, self.x),
            self.z,
            bounds_error=False,
            fill_value=np.nan
        )

        if self.grow_facade:
            self.ground_interpolator = RegularGridInterpolator(
                (self.y, self.x),
                data['ground'],
                bounds_error=False,
                fill_value=np.nan
            )

    def _grid_connected_components(self, points_xy):
        """
        Simple grid-based labeling: divide XY plane into grid cells and
        label connected clusters. Very fast for large clouds.
        Returns a boolean mask keeping clusters larger than min_comp_size.
        """
        if len(points_xy) == 0:
            return np.zeros(0, dtype=bool)

        # Compute grid indices
        x_idx = np.floor(points_xy[:, 0] / self.grid_size).astype(int)
        y_idx = np.floor(points_xy[:, 1] / self.grid_size).astype(int)
        keys = list(zip(x_idx, y_idx))

        # Map grid cells to point indices
        from collections import defaultdict
        cell_points = defaultdict(list)
        for i, key in enumerate(keys):
            cell_points[key].append(i)

        # Identify clusters
        cluster_mask = np.zeros(len(points_xy), dtype=bool)
        for indices in cell_points.values():
            if len(indices) >= self.min_comp_size:
                cluster_mask[indices] = True
        return cluster_mask

    def _grow_facade(self, points, labels):
        """
        Expand building label to overhanging facade elements via 3-D voxel LCC.

        Container: points with height above ground >= facade_floor that are
        either unlabeled (0) or already labeled as building.  Components that
        contain at least one building-labeled point are grown; only currently
        unlabeled points within those components are relabeled.
        """
        coords = np.vstack((points[:, 1], points[:, 0])).T  # Y, X
        ground_z = self.ground_interpolator(coords)
        heights = points[:, 2] - ground_z

        container_mask = (
            (heights >= self.facade_floor)
            & ~np.isnan(ground_z)
            & ((labels == 0) | (labels == self.label))
        )
        container_idx = np.where(container_mask)[0]
        if len(container_idx) < 2:
            return labels

        lcc = LabelConnectedComp(
            grid_size=self.facade_grid_size,
            min_component_size=self.facade_min_comp,
        )
        comp_labels = lcc.get_components(points[container_idx])

        # Seed: components touching already-labeled building points
        seed_comps = set(comp_labels[labels[container_idx] == self.label]) - {-1}
        if not seed_comps:
            return labels

        # Grow into unlabeled points in seeded components only
        grow_mask = (
            np.isin(comp_labels, list(seed_comps))
            & (labels[container_idx] == 0)
        )
        n_added = int(grow_mask.sum())
        labels[container_idx[grow_mask]] = self.label
        logger.info(f"NPZAHNFuser facade grow: {n_added} points added.")
        return labels

    def get_labels(self, points, labels, mask, tilecode):
        if np.count_nonzero(mask) == 0:
            return labels

        pts_masked = points[mask]

        # Interpolate AHN surface
        coords = np.vstack((pts_masked[:,1], pts_masked[:,0])).T  # Y,X
        surface_z = self.interpolator(coords)

        # Height difference
        height_diff = pts_masked[:,2] - surface_z

        # Initial selection
        if self.target == 'ground':
            selected = np.abs(height_diff) <= self.epsilon
        else:
            selected = height_diff <= self.epsilon

        if np.count_nonzero(selected) == 0:
            return labels

        # Grid-based cluster filtering
        xy_selected = pts_masked[selected][:, 0:2]
        cluster_mask = self._grid_connected_components(xy_selected)

        # Map back to full mask
        final_mask = np.zeros(len(selected), dtype=bool)
        selected_indices = np.where(selected)[0]
        final_mask[selected_indices[cluster_mask]] = True

        # Update labels
        labels_masked = np.zeros_like(mask)
        labels_masked[mask] = final_mask
        labels[labels_masked] = self.label

        logger.info(f"NPZAHNFuser ({self.target}): {np.count_nonzero(final_mask)} points labeled.")

        if self.grow_facade:
            labels = self._grow_facade(points, labels)

        return labels