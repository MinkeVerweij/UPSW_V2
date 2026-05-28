# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

import numpy as np
import numba
from numba import jit
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


@jit(nopython=True, cache=True, parallel=True)
def vector_angle(u, v=np.array([0., 0., 1.])):
    """
    Returns the angle in degrees between vectors 'u' and 'v'. If only 'u' is
    provided, the angle between 'u' and the vertical axis is returned.
    """
    # see https://stackoverflow.com/a/2827466/425458
    c = np.dot(u/np.linalg.norm(u), v/np.linalg.norm(v))
    clip = np.minimum(1, np.maximum(c, -1))
    return np.rad2deg(np.arccos(clip))


@jit(nopython=True, cache=True, parallel=True)
def get_octree_level(points, grid_size):
    """Compute nearest octree level based on a desired grid_size."""
    dims = np.zeros((points.shape[1],))
    for d in range(points.shape[1]):
        dims[d] = np.max(points[:, d]) - np.min(points[:, d])
    max_dim = np.max(dims)
    if max_dim < 0.001:
        return 0
    octree_level = np.rint(-np.log(grid_size / max_dim) / (np.log(2)))
    if octree_level > 0:
        return np.int64(octree_level)
    return 1


@jit(nopython=True, cache=True, parallel=True)
def compute_bounding_box(points):
    """
    Get the min/max values of a point list.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The (x, y) coordinates of the points. Any further dimensions will be
        ignored.

    Returns
    -------
    tuple
        (x_min, y_min, x_max, y_max)
    """
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    return (x_min, y_min, x_max, y_max)


def convex_hull_poly(points):
    """Return convex hull as a shapely Polygon."""
    return Polygon(points[ConvexHull(points, qhull_options='QJ').vertices])


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    # Calculate center point and project onto rotated frame
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_point = np.dot([center_x, center_y], r)

    min_bounding_rect = np.zeros((4, 2))
    min_bounding_rect[0] = np.dot([x1, y2], r)
    min_bounding_rect[1] = np.dot([x2, y2], r)
    min_bounding_rect[2] = np.dot([x2, y1], r)
    min_bounding_rect[3] = np.dot([x1, y1], r)

    # Compute the dims of the min bounding rectangle
    dims = [(x1 - x2), (y1 - y2)]

    return min_bounding_rect, hull_points, min(dims), max(dims), center_point

def _point_inside_poly(polygon, point):
    """
    Improved version of the Crossing Number algorithm that checks if a point is
    inside a polygon.
    Implementation taken from https://github.com/sasamil/PointInPolygon_Py
    """
    length = len(polygon) - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/below/right from
        # the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0]
                              or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = (dy * (polygon[jj][0] - polygon[ii][0])
                     / (dy-dy2) + polygon[ii][0])

                if point[0] > F:
                    # if line is left from the point - the ray moving towards
                    # left, will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and
            # dx*dx2<=0)
            elif (dy2 == 0
                  and (point[0] == polygon[jj][0]
                       or (dy == 0 and (point[0] - polygon[ii][0])
                           * (point[0] - polygon[jj][0]) <= 0))):
                return 2

        ii = jj
        jj += 1

    return intersections & 1

def is_inside(x, y, polygon):
    """
    Checks for each point in a list whether that point is inside a polygon.

    Parameters
    ----------
    x : list
        X-coordinates.
    y : list
        Y-coordinates.
    polygon : list of tuples
        Polygon as linear ring.

    Returns
    -------
    An array of shape (len(x),) with dtype bool, where each entry indicates
    whether the corresponding point is inside the polygon.
    """
    n = len(x)
    #mask = np.empty((n,), dtype=numba.boolean)
    mask = np.empty(n, dtype=np.bool_)
    for i in numba.prange(n):
        mask[i] = _point_inside_poly(polygon, (x[i], y[i]))
    return mask

def rectangle_clip(points, rect):
    """
    Clip all points within a rectangle.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    rect : tuple of floats
        (x_min, y_min, x_max, y_max)

    Returns
    -------
    A boolean mask with True entries for all points within the rectangle.
    """
    clip_mask = ((points[:, 0] >= rect[0]) & (points[:, 0] <= rect[2])
                 & (points[:, 1] >= rect[1]) & (points[:, 1] <= rect[3]))
    return clip_mask

def box_clip(points, rect, bottom=-np.inf, top=np.inf):
    """
    Clip all points within a 3D box.

    Parameters
    ----------
    points : array of shape (n_points, 3)
        The points.
    rect : tuple of floats
        (x_min, y_min, x_max, y_max)
    bottom : float (default: -inf)
    top : float (default: inf)

    Returns
    -------
    A boolean mask with True entries for all points within the box.
    """
    clip_mask = rectangle_clip(points, rect)
    clip_mask = clip_mask & (points[:, 2] >= bottom) & (points[:, 2] <= top)
    return clip_mask


def cylinder_clip(points, center, radius, bottom=-np.inf, top=np.inf):
    """
    Clip all points within a vertical cylinder.

    Parameters
    ----------
    points : array of shape (n_points, 3)
        The points.
    center : array-like of shape (2,)
        XY centre of the cylinder.
    radius : float
        Radius of the cylinder.
    bottom : float (default: -inf)
    top : float (default: inf)

    Returns
    -------
    A boolean mask with True entries for all points within the cylinder.
    """
    clip_mask = ((points[:, 0] - center[0])**2
                 + (points[:, 1] - center[1])**2 <= radius**2)
    clip_mask = clip_mask & (points[:, 2] >= bottom) & (points[:, 2] <= top)
    return clip_mask


def poly_clip(points, poly):
    """
    Clip all points within a polygon.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    poly : shapely.geometry Polygon object
        Polygon to clip. Can have interior gaps.

    Returns
    -------
    A boolean mask with True entries for all points within the polygon.
    """
    clip_mask = np.zeros((len(points),), dtype=bool)

    # Convert to numpy to work with numba jit in nopython mode.
    exterior = np.array(poly.exterior.coords)
    interiors = [np.array(interior.coords) for interior in poly.interiors]

    if len(exterior) < 3:
        # Polygon has no interior.
        return clip_mask

    # Clip exterior to include points.
    bbox_mask = rectangle_clip(
                    points, compute_bounding_box(exterior))
    exterior_mask = is_inside(points[bbox_mask, 0], points[bbox_mask, 1],
                              exterior)
    bbox_inds = np.where(bbox_mask)[0]
    clip_mask[bbox_inds[exterior_mask]] = True

    # Clip interior(s) to exclude points.
    for interior in interiors:
        if len(interior) < 3:
            # Polygon has no interior.
            continue
        bbox_mask = rectangle_clip(
                        points, compute_bounding_box(interior))
        interior_mask = is_inside(points[bbox_mask, 0], points[bbox_mask, 1],
                                  interior)
        bbox_inds = np.where(bbox_mask)[0]
        clip_mask[bbox_inds[interior_mask]] = False

    return clip_mask
