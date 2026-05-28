import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Polygon
import logging

logger = logging.getLogger(__name__)


class NPZAHNFuser:
    """
    NPZ-based AHN fuser compatible with Pipeline.
    Uses grid-based connected component filtering instead of DBSCAN for speed.
    """

    TARGETS = ('ground', 'building')

    def __init__(self, label, npz_reader, target='ground', epsilon=0.2,
                 grid_size=0.4, min_comp_size=20):
        if target not in self.TARGETS:
            raise ValueError(f"Target must be one of {self.TARGETS}")
        self.label = label
        self.npz_reader = npz_reader
        self.target = target
        self.epsilon = epsilon
        self.grid_size = grid_size
        self.min_comp_size = min_comp_size
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
        return labels