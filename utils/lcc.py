"""Voxel-grid connected-components clustering (ported from upcp)."""

import numpy as np
import logging
from scipy.ndimage import label as ndimage_label

logger = logging.getLogger(__name__)


class LabelConnectedComp:
    """
    Clusters points using a 3-D voxel-grid with 26-connectivity.

    Points are binned into voxels of size ``grid_size``.  Occupied voxels
    that share a face, edge, or corner are merged into one component.  Each
    point then inherits its voxel's component label.  Components smaller than
    ``min_component_size`` are labelled -1.

    Parameters
    ----------
    label : int
        Label to assign when labelling grown regions.
    grid_size : float (default: 0.1)
        Voxel cell size in metres.
    min_component_size : int (default: 100)
        Minimum number of points a component must have to be retained.
    """

    def __init__(self, label=-1, grid_size=0.1, min_component_size=100):
        self.label = label
        self.grid_size = grid_size
        self.min_component_size = min_component_size

    def _label_connected_comp(self, points):
        if len(points) == 0:
            return np.array([], dtype=np.int64)

        mins = points.min(axis=0)
        indices = np.floor((points - mins) / self.grid_size).astype(np.int32)

        grid_shape = tuple(indices.max(axis=0) + 1)
        voxel_grid = np.zeros(grid_shape, dtype=bool)
        idx = tuple(indices[:, d] for d in range(points.shape[1]))
        voxel_grid[idx] = True

        structure = np.ones([3] * points.shape[1], dtype=bool)
        labeled_grid, _ = ndimage_label(voxel_grid, structure=structure)

        components = labeled_grid[idx].astype(np.int64) - 1

        if self.min_component_size > 1:
            cc_labels, counts = np.unique(components, return_counts=True)
            small = cc_labels[counts < self.min_component_size]
            components[np.isin(components, small)] = -1

        return components

    def get_components(self, points):
        """
        Return per-point component labels without any seed logic.

        Parameters
        ----------
        points : array of shape (N, 3) or (N, 2)

        Returns
        -------
        Array of shape (N,) with integer component labels.  Components
        smaller than ``min_component_size`` are labelled -1.
        """
        return self._label_connected_comp(points)
