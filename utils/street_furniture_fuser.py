"""BGT street furniture fuser (benches, bins) — ported from upcp."""

import numpy as np
import logging
from scipy.spatial import ConvexHull

from .lcc import LabelConnectedComp

logger = logging.getLogger(__name__)


def _minimum_bounding_rectangle(points):
    """
    Return the minimum-area bounding rectangle of a set of 2-D points.

    Returns
    -------
    (rect_corners, hull_points, width, length, center)
    """
    pi2 = np.pi / 2.0
    hull_pts = points[ConvexHull(points).vertices]

    edges = hull_pts[1:] - hull_pts[:-1]
    angles = np.unique(np.abs(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), pi2)))
    rotations = np.vstack([np.cos(angles),
                           np.cos(angles - pi2),
                           np.cos(angles + pi2),
                           np.cos(angles)]).T.reshape(-1, 2, 2)

    rot_pts = np.dot(rotations, hull_pts.T)
    min_x = rot_pts[:, 0, :].min(axis=1)
    max_x = rot_pts[:, 0, :].max(axis=1)
    min_y = rot_pts[:, 1, :].min(axis=1)
    max_y = rot_pts[:, 1, :].max(axis=1)

    areas = (max_x - min_x) * (max_y - min_y)
    best = np.argmin(areas)
    r = rotations[best]
    x1, x2 = max_x[best], min_x[best]
    y1, y2 = max_y[best], min_y[best]

    center = np.dot([(x1 + x2) / 2, (y1 + y2) / 2], r)
    rect = np.array([np.dot([x1, y2], r),
                     np.dot([x2, y2], r),
                     np.dot([x2, y1], r),
                     np.dot([x1, y1], r)])
    dims = sorted([abs(x1 - x2), abs(y1 - y2)])
    return rect, hull_pts, dims[0], dims[1], center


class BGTStreetFurnitureFuser:
    """
    Labels street furniture objects (city benches, rubbish bins) using BGT.

    For each BGT point location the fuser searches for a cluster of the
    right physical dimensions nearby and assigns the label to all points
    in that cluster.

    Parameters
    ----------
    label : int
        Class label to assign.
    bgt_type : str
        ``'bank'`` (bench) or ``'afvalbak'`` (bin).
    bgt_reader : BGTPointReader
    ahn_reader : PolygonNPZReader
    grid_size : float (default: 0.05 m)
        Voxel size for LCC clustering.
    min_component_size : int (default: 1500)
        Minimum voxel cluster size to consider.
    padding : float (default: 0)
        Extra metres around the tile for BGT lookup.
    max_dist : float (default: 1.0 m)
        Maximum distance from BGT point to cluster centre.
    params : dict
        Override shape filter defaults (see Notes).

    Notes
    -----
    Default ``params``:

    ``min_height`` (0.3 m), ``max_height`` (1.5 m)
        Allowed range of cluster top height above ground.
    ``min_width`` (0.2 m), ``max_width`` (1.0 m)
        Allowed range for the short dimension of the bounding rectangle.
    ``min_length`` (0.3 m), ``max_length`` (2.5 m)
        Allowed range for the long dimension.
    """

    _DEFAULTS = {
        'min_height': 0.3,
        'max_height': 1.5,
        'min_width': 0.2,
        'max_width': 1.0,
        'min_length': 0.3,
        'max_length': 2.5,
    }

    def __init__(self, label, bgt_type, bgt_reader, ahn_reader,
                 grid_size=0.05, min_component_size=1500,
                 padding=0, max_dist=1.0, params=None):
        self.label = label
        self.bgt_type = bgt_type
        self.bgt_reader = bgt_reader
        self.ahn_reader = ahn_reader
        self.grid_size = grid_size
        self.min_component_size = min_component_size
        self.padding = padding
        self.max_dist = max_dist
        self.params = {**self._DEFAULTS, **(params or {})}

    def _label_furniture_components(self, points, ground_z, components,
                                    bgt_points,
                                    min_height, max_height,
                                    min_width, max_width,
                                    min_length, max_length):
        # Pass 1: every component that independently passes the height/shape
        # filter, regardless of BGT proximity yet -- the candidate pool.
        candidates = []  # list of (cc_mask, centre)
        for cc in set(np.unique(components)).difference((-1,)):
            cc_mask = components == cc
            valid_z = ground_z[cc_mask]
            valid_z = valid_z[np.isfinite(valid_z)]
            if valid_z.size == 0:
                continue

            cc_ground = float(np.mean(valid_z))
            cluster_top = float(np.max(points[cc_mask, 2]))
            if not (cc_ground + min_height <= cluster_top
                    <= cc_ground + max_height):
                continue

            try:
                _, _, width, length, centre = _minimum_bounding_rectangle(
                    points[cc_mask, :2])
            except Exception:
                continue

            if not (min_width < width < max_width
                    and min_length < length < max_length):
                continue

            candidates.append((cc_mask, centre))

        # Pass 2: one-to-one assignment between BGT points and candidate
        # components. A row of 2+ real objects close together (e.g. several
        # waste streams sharing one enclosure, each with its own BGT point)
        # previously let multiple BGT points all claim whichever component
        # happened to be checked first, or left a real second component
        # unmatched -- there was no bookkeeping preventing either. Instead,
        # collect every (BGT point, component) pair within max_dist, sort
        # by distance, and greedily claim pairs nearest-first, skipping a
        # pair once either side is already claimed. This gives each nearby
        # BGT point its own distinct component when enough real, separate
        # ones exist, and correctly leaves a BGT point unmatched (rather
        # than double-claiming) when only one physical object is actually
        # there for two nearby BGT records.
        pairs = []
        for bp in bgt_points:
            bp_arr = np.array(bp)
            for ci, (_, centre) in enumerate(candidates):
                dist = np.linalg.norm(bp_arr - centre)
                if dist <= self.max_dist:
                    pairs.append((dist, tuple(bp), ci))
        pairs.sort(key=lambda p: p[0])

        furniture_mask = np.zeros(len(points), dtype=bool)
        claimed_components, claimed_bgt = set(), set()
        object_count = 0
        for dist, bp_key, ci in pairs:
            if ci in claimed_components or bp_key in claimed_bgt:
                continue
            cc_mask, _ = candidates[ci]
            furniture_mask[cc_mask] = True
            claimed_components.add(ci)
            claimed_bgt.add(bp_key)
            object_count += 1

        logger.debug(f'{object_count} {self.bgt_type} objects labelled.')
        return furniture_mask

    def get_labels(self, points, labels, mask, tilecode):
        """
        Label street furniture in the point cloud.

        Parameters
        ----------
        points : (N, 3) array
        labels : (N,) array
        mask   : (N,) bool array
        tilecode : str

        Returns
        -------
        Updated labels array.
        """
        logger.info(f'BGTStreetFurnitureFuser [{self.bgt_type}] '
                    f'(label={self.label}).')

        bgt_points = self.bgt_reader.filter_tile(
            tilecode, bgt_types=[self.bgt_type],
            padding=self.padding, return_types=False)
        if not bgt_points:
            logger.debug(f'No {self.bgt_type} in tile, skipping.')
            return labels

        ground_z = self.ahn_reader.interpolate(
            tilecode, points[mask], surface='ground_surface')

        lcc = LabelConnectedComp(
            label=self.label,
            grid_size=self.grid_size,
            min_component_size=self.min_component_size)
        components = lcc.get_components(points[mask])

        furniture_mask_sub = self._label_furniture_components(
            points[mask], ground_z, components, bgt_points, **self.params)

        label_mask = np.zeros(len(points), dtype=bool)
        label_mask[mask] = furniture_mask_sub
        labels[label_mask] = self.label
        return labels
