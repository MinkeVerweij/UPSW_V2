"""
Rule-based obstacle classifier for the wheelchair routing project.

For each obstacle polygon detected by notebook 3, this module:
1. Extracts 3D geometric features from the labeled point cloud
2. Looks up which BGT surface type (voetpad, fietspad, road) the obstacle sits on
3. Classifies the obstacle type and permanence using dimension-based rules

Dimension ranges are derived from the parameter defaults in
pole_fuser.py and street_furniture_fuser.py.
"""

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point

from .labels import Labels

# BGT label codes that can appear in the bgt_labeled LAZ files
_BGT_LABEL_SET = {
    Labels.TREE, Labels.CAR, Labels.STREET_LIGHT, Labels.TRAFFIC_LIGHT,
    Labels.TRAFFIC_SIGN, Labels.BOLLARD, Labels.CITY_BENCH, Labels.RUBBISH_BIN,
    Labels.LARGE_CONTAINER, Labels.BICYCLE_RACK,
}

_BGT_PERMANENCE = {
    Labels.TREE:            'permanent',
    Labels.CAR:             'temporary',
    Labels.STREET_LIGHT:    'permanent',
    Labels.TRAFFIC_LIGHT:   'permanent',
    Labels.TRAFFIC_SIGN:    'permanent',
    Labels.BOLLARD:         'permanent',
    Labels.CITY_BENCH:      'permanent',
    Labels.RUBBISH_BIN:     'permanent',
    Labels.LARGE_CONTAINER: 'permanent',
    Labels.BICYCLE_RACK:    'permanent',
}

# Must match obstacles_utils.py
_GROUND_GRID_SIZE = 0.5
_HEIGHT_THRESHOLD  = 0.25

# Surface categories used in classification rules
_ROAD_SURFACES = {'road', 'parkeervlak'}
_FOOT_SURFACES = {'voetpad'}
_BIKE_SURFACES = {'fietspad'}


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _mbr_dims(points_2d):
    """
    Return (width, length) — short and long side of the minimum-area
    bounding rectangle for a set of 2-D points.
    Copied from street_furniture_fuser._minimum_bounding_rectangle.
    """
    if len(points_2d) < 3:
        dx = float(points_2d[:, 0].max() - points_2d[:, 0].min())
        dy = float(points_2d[:, 1].max() - points_2d[:, 1].min())
        return sorted([dx, dy])

    pi2 = np.pi / 2.0
    try:
        hull_pts = points_2d[ConvexHull(points_2d).vertices]
    except Exception:
        dx = float(points_2d[:, 0].max() - points_2d[:, 0].min())
        dy = float(points_2d[:, 1].max() - points_2d[:, 1].min())
        return sorted([dx, dy])

    edges = hull_pts[1:] - hull_pts[:-1]
    angles = np.unique(np.abs(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), pi2)))
    rots = np.vstack([np.cos(angles), np.cos(angles - pi2),
                      np.cos(angles + pi2), np.cos(angles)]).T.reshape(-1, 2, 2)
    rot_pts = np.dot(rots, hull_pts.T)
    w = rot_pts[:, 0, :].max(axis=1) - rot_pts[:, 0, :].min(axis=1)
    h = rot_pts[:, 1, :].max(axis=1) - rot_pts[:, 1, :].min(axis=1)
    best = np.argmin(w * h)
    return sorted([float(w[best]), float(h[best])])


# ---------------------------------------------------------------------------
# Ground grid (mirrors obstacles_utils.build_ground_grid logic)
# ---------------------------------------------------------------------------

def build_ground_grid(xyz, labels_arr):
    """
    Build a minimum-elevation ground grid from GROUND-labeled points.

    Returns
    -------
    grid : dict {(gx, gy): min_z}
    xmin, ymin : float — grid origin
    """
    ground = xyz[labels_arr == Labels.GROUND]
    if len(ground) == 0:
        return {}, 0.0, 0.0

    xmin = float(ground[:, 0].min())
    ymin = float(ground[:, 1].min())
    grid = {}
    gx = ((ground[:, 0] - xmin) / _GROUND_GRID_SIZE).astype(int)
    gy = ((ground[:, 1] - ymin) / _GROUND_GRID_SIZE).astype(int)
    for ix, iy, z in zip(gx, gy, ground[:, 2].astype(float)):
        key = (int(ix), int(iy))
        if key not in grid or z < grid[key]:
            grid[key] = z
    return grid, xmin, ymin


def _ground_elevation(x, y, grid, xmin, ymin, fallback):
    gx = int((x - xmin) / _GROUND_GRID_SIZE)
    gy = int((y - ymin) / _GROUND_GRID_SIZE)
    return grid.get((gx, gy), fallback)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_cluster_features(xyz, labels_arr, polygon, ground_grid, xmin, ymin):
    """
    Extract 3D geometric features and BGT labels for points inside an
    obstacle polygon.

    Parameters
    ----------
    xyz : (N, 3) float array
    labels_arr : (N,) int array — labels from the bgt_labeled LAZ
    polygon : shapely.Polygon — obstacle footprint from notebook 3 output
    ground_grid : dict — from build_ground_grid()
    xmin, ymin : float — grid origin from build_ground_grid()

    Returns
    -------
    dict with keys height, width, length, point_density, bgt_labels
    or None when fewer than 3 points fall inside the polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds
    bbox = ((xyz[:, 0] >= minx) & (xyz[:, 0] <= maxx) &
            (xyz[:, 1] >= miny) & (xyz[:, 1] <= maxy))
    if not bbox.any():
        return None

    pts_box = xyz[bbox]
    lbl_box = labels_arr[bbox]

    inside = np.array([polygon.contains(Point(p[0], p[1])) for p in pts_box])
    n_inside = inside.sum()
    if n_inside < 3:
        return None

    pts_in  = pts_box[inside]
    lbl_in  = lbl_box[inside]

    bgt_labels = {int(l) for l in np.unique(lbl_in) if int(l) in _BGT_LABEL_SET}

    fallback_z = float(pts_in[:, 2].min())
    gz = np.array([
        _ground_elevation(p[0], p[1], ground_grid, xmin, ymin, fallback_z)
        for p in pts_in
    ])
    h_above = pts_in[:, 2] - gz
    obs_mask = h_above > _HEIGHT_THRESHOLD

    if obs_mask.sum() < 3:
        height = float(h_above.max()) if len(h_above) else 0.0
        width, length = 0.0, 0.0
    else:
        height = float(h_above[obs_mask].max())
        width, length = _mbr_dims(pts_in[obs_mask, :2])

    area = polygon.area
    point_density = n_inside / area if area > 0 else 0.0

    return {
        'height':        round(height, 3),
        'width':         round(width,  3),
        'length':        round(length, 3),
        'point_density': round(point_density, 1),
        'bgt_labels':    bgt_labels,
    }


# ---------------------------------------------------------------------------
# Surface type lookup
# ---------------------------------------------------------------------------

def lookup_surface_types(obstacle_polygon, surface_polygons):
    """
    Return the set of all surface types that the obstacle polygon intersects.

    Using the full polygon rather than just the centroid correctly handles
    obstacles that straddle two surfaces (e.g. a car parked half on the road
    and half on the voetpad).

    Parameters
    ----------
    obstacle_polygon : shapely.Polygon
    surface_polygons : dict[str, list[shapely.Polygon]]
        Keys are surface names ('voetpad', 'fietspad', 'road', 'parkeervlak').

    Returns
    -------
    set[str] — all overlapping surface names; {'other_ground'} when none match
    """
    found = set()
    for name, polys in surface_polygons.items():
        for poly in polys:
            try:
                if poly.intersects(obstacle_polygon):
                    found.add(name)
                    break  # one match per surface type is enough
            except Exception:
                continue
    return found if found else {'other_ground'}


# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------

def classify_obstacle(height, width, length, surface_types, bgt_labels):
    """
    Classify an obstacle cluster into a type and permanence category.

    BGT-matched obstacles are classified directly from their BGT label.
    Unmatched obstacles use dimension rules derived from the fuser parameter
    defaults in pole_fuser.py and street_furniture_fuser.py.

    Parameters
    ----------
    height, width, length : float — metres, width <= length
    surface_types : set[str] — from lookup_surface_types(); an obstacle that
        straddles two surfaces will have both names in the set
    bgt_labels : set[int] — BGT label codes found inside the polygon

    Returns
    -------
    (type_label: int, type_name: str, permanence: str)
    permanence is 'permanent', 'temporary', or 'unknown'
    """
    def on(*surfaces):
        return bool(surface_types & set(surfaces))

    # BGT match takes priority — use the most specific (highest) label
    if bgt_labels:
        dominant = max(bgt_labels)
        return (dominant,
                Labels.STR_DICT.get(dominant, str(dominant)),
                _BGT_PERMANENCE.get(dominant, 'permanent'))

    # Ignore near-ground noise
    if height < 0.4:
        return (Labels.UNKNOWN, 'Unknown', 'unknown')

    aspect = length / width if width > 0.01 else 0.0

    # Car — road or parking, but also catches cars parked half on the voetpad
    if (1.2 <= height <= 2.2
            and 1.4 <= width <= 2.5
            and 2.5 <= length <= 6.0
            and on('road', 'parkeervlak', 'voetpad', 'other_ground')):
        return (Labels.CAR, 'Car', 'temporary')

    # Large container — voetpad only, large footprint
    if (0.7 <= height <= 2.5
            and 0.4 <= width <= 2.5
            and 1.0 <= length <= 6.0
            and width * length >= 1.5
            and on('voetpad', 'other_ground')):
        return (Labels.LARGE_CONTAINER, 'Large container', 'permanent')

    # City bench — elongated, low-medium, voetpad
    # Max height 1.0 m keeps it below handlebar height of a parked bicycle
    if (0.3 <= height <= 1.0
            and 0.3 <= width <= 0.9
            and 0.8 <= length <= 2.0
            and aspect > 2.0
            and on('voetpad', 'other_ground')):
        return (Labels.CITY_BENCH, 'City bench', 'permanent')

    # Bicycle rack — voetpad only; ≥2.2 m long (space for 2+ bikes) to separate from single bicycle
    if (0.5 <= height <= 1.4
            and 0.3 <= width <= 1.2
            and length >= 2.2
            and aspect > 2.5
            and on('voetpad', 'other_ground')):
        return (Labels.BICYCLE_RACK, 'Bicycle rack', 'permanent')

    # Rubbish bin — compact, roughly square footprint, voetpad
    if (0.4 <= height <= 1.2
            and 0.2 <= width <= 0.8
            and 0.2 <= length <= 0.8
            and aspect <= 2.0
            and on('voetpad', 'other_ground')):
        return (Labels.RUBBISH_BIN, 'Rubbish bin', 'permanent')

    # Bollard — narrow post, any surface
    if (0.3 <= height <= 1.5
            and 0.05 <= width <= 0.4
            and 0.05 <= length <= 0.6
            and aspect <= 3.0):
        return (Labels.BOLLARD, 'Bollard', 'permanent')

    # Bicycle / scooter — voetpad only; min height 1.0 m (handlebar) separates from bench
    if (1.0 <= height <= 1.8
            and 0.3 <= width <= 1.0
            and 0.4 <= length <= 2.0
            and on('voetpad', 'other_ground')):
        return (44, 'Bicycle', 'temporary')

    return (Labels.UNKNOWN, 'Unknown', 'unknown')
