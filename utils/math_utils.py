"""Geometry utilities — ported from upcp (no numba dependency)."""

import numpy as np
from scipy.spatial import ConvexHull


def minimum_bounding_rectangle(points):
    """
    Find the minimum-area bounding rectangle for a set of 2D points.

    Returns
    -------
    min_bounding_rect : (4, 2) array — corner coordinates
    hull_points       : (m, 2) array — convex hull vertices
    mbr_width         : float — shorter side length
    mbr_length        : float — longer side length
    center_point      : (2,) array — centre of the MBR
    """
    pi2 = np.pi / 2.0

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    edges = hull_points[1:] - hull_points[:-1]
    angles = np.abs(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), pi2))
    angles = np.unique(angles)

    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles),
    ]).T.reshape(-1, 2, 2)

    rot_points = np.dot(rotations, hull_points.T)
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    areas = (max_x - min_x) * (max_y - min_y)
    best = np.argmin(areas)

    x1, x2 = max_x[best], min_x[best]
    y1, y2 = max_y[best], min_y[best]
    r = rotations[best]

    corners = np.array([
        np.dot([x1, y2], r),
        np.dot([x2, y2], r),
        np.dot([x2, y1], r),
        np.dot([x1, y1], r),
    ])
    center_point = np.dot([(x1 + x2) / 2, (y1 + y2) / 2], r)

    dims = sorted([(x1 - x2), (y1 - y2)])
    return corners, hull_points, dims[0], dims[1], center_point
