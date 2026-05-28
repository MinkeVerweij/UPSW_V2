"""
2D obstacle extraction via ground-height filtering + DBSCAN on a 2D voxel grid.

Replaces the 3D connected-component approach in obstacles_utils.py.  The key
difference is that we project obstacle candidates onto the XY plane before
clustering, so a truck (with gaps between cab, trailer, or scan angles) is
grouped as a single footprint instead of several disconnected blobs.

Efficiency notes for city-scale use:
  - Ground grid and height computation are fully vectorised (no Python loops).
  - DBSCAN runs on 2D voxel centroids, not raw points, so the input size is
    1-2 orders of magnitude smaller than the point cloud.
  - Per-tile memory stays well under 1 GB for typical 500 m tiles.
"""

import numpy as np
import laspy
import geopandas as gpd
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN

# ── defaults ──────────────────────────────────────────────────────────────────
GROUND_LABEL     = 9
UNKNOWN_LABEL    = 0     # keep only points with this label; everything else is a
                         # known BGT object and should not be re-detected
HEIGHT_THRESHOLD = 0.25          # m above ground to be an obstacle candidate
GROUND_GRID_SIZE = 0.5           # m, resolution of the ground-height raster
VOXEL_2D         = 0.25          # m, 2-D footprint voxel side length
DBSCAN_EPS       = 0.6           # m, DBSCAN neighbourhood radius
DBSCAN_MIN_VOXELS = 5            # minimum occupied voxels for a cluster
MIN_AREA         = 0.2           # m², post-clustering area filter (lower bound)
MAX_AREA         = 40.0          # m², post-clustering area filter (upper bound)


# ── ground grid ───────────────────────────────────────────────────────────────

def build_ground_grid(xyz: np.ndarray, labels: np.ndarray,
                      grid_size: float = GROUND_GRID_SIZE):
    """
    Build a 2-D raster of minimum ground elevation.

    Returns
    -------
    grid   : (W, H) float32 array, NaN where no ground data
    xmin   : float
    ymin   : float
    """
    ground = xyz[labels == GROUND_LABEL]
    if len(ground) == 0:
        return None, None, None

    xmin = float(ground[:, 0].min())
    ymin = float(ground[:, 1].min())

    gx = np.floor((ground[:, 0] - xmin) / grid_size).astype(np.int32)
    gy = np.floor((ground[:, 1] - ymin) / grid_size).astype(np.int32)
    gz = ground[:, 2].astype(np.float32)

    W, H = int(gx.max()) + 1, int(gy.max()) + 1
    grid = np.full((W, H), np.inf, dtype=np.float32)
    np.minimum.at(grid, (gx, gy), gz)
    grid[grid == np.inf] = np.nan

    return grid, xmin, ymin


def compute_heights(xyz: np.ndarray, grid: np.ndarray,
                    xmin: float, ymin: float,
                    grid_size: float = GROUND_GRID_SIZE) -> np.ndarray:
    """
    Vectorised height-above-ground for every point in xyz.

    Points outside the ground raster extent get height 0 (not obstacle
    candidates) rather than a fallback that might incorrectly include them.
    """
    W, H = grid.shape

    px = np.floor((xyz[:, 0] - xmin) / grid_size).astype(np.int32)
    py = np.floor((xyz[:, 1] - ymin) / grid_size).astype(np.int32)

    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)

    gz = np.full(len(xyz), np.nan, dtype=np.float32)
    gz[in_bounds] = grid[px[in_bounds], py[in_bounds]]

    # Points with no ground data → height 0 (excluded from obstacle candidates)
    heights = np.where(np.isnan(gz), 0.0, xyz[:, 2] - gz)
    return heights.astype(np.float32)


# ── main extraction ───────────────────────────────────────────────────────────

def extract_obstacles(
    laz_file: str,
    voxel_2d: float         = VOXEL_2D,
    eps: float              = DBSCAN_EPS,
    min_voxels: int         = DBSCAN_MIN_VOXELS,
    height_threshold: float = HEIGHT_THRESHOLD,
    unknown_label: int      = UNKNOWN_LABEL,
    verbose: bool           = True,
) -> list[np.ndarray]:
    """
    Extract 2-D obstacle clusters from a labelled LAZ/LAS file.

    Designed for use with ``bgt_labeled_`` point clouds: only points whose
    label equals ``unknown_label`` (default 0) are considered obstacle
    candidates, so every object already identified by the BGT pipeline
    (road, ground, building, tree, street light, …) is automatically excluded
    without having to enumerate them.

    Parameters
    ----------
    laz_file         : path to the labelled point cloud
    voxel_2d         : XY voxel size for 2-D footprint reduction
    eps              : DBSCAN neighbourhood radius (metres)
    min_voxels       : minimum number of occupied voxels to keep a cluster
    height_threshold : minimum height above ground to be an obstacle candidate
    unknown_label    : only points with this label are clustered

    Returns
    -------
    List of (N,3) float arrays, one per cluster (XYZ of the original points).
    """
    def log(msg):
        if verbose:
            print(msg)

    log(f"  Loading {laz_file} …")
    pc = laspy.read(laz_file)

    xyz = np.column_stack([
        np.asarray(pc.x, dtype=np.float64),
        np.asarray(pc.y, dtype=np.float64),
        np.asarray(pc.z, dtype=np.float64),
    ])
    labels = (
        np.asarray(pc.label, dtype=np.int32)
        if "label" in pc.point_format.extra_dimension_names
        else np.zeros(len(xyz), dtype=np.int32)
    )
    log(f"  Points: {len(xyz):,}")

    # ── ground grid & heights ─────────────────────────────────────────────────
    grid, xmin, ymin = build_ground_grid(xyz, labels)
    if grid is None:
        log("  No ground points found, skipping.")
        return []

    heights = compute_heights(xyz, grid, xmin, ymin)

    # ── obstacle candidates ───────────────────────────────────────────────────
    obs_mask = (labels == unknown_label) & (heights > height_threshold)
    obs_xyz = xyz[obs_mask]
    log(f"  Obstacle candidates: {len(obs_xyz):,}")

    if len(obs_xyz) < min_voxels:
        log("  Too few candidates.")
        return []

    # ── 2-D voxelisation ─────────────────────────────────────────────────────
    vx = np.floor(obs_xyz[:, 0] / voxel_2d).astype(np.int32)
    vy = np.floor(obs_xyz[:, 1] / voxel_2d).astype(np.int32)

    # Unique occupied cells and a reverse map (point → cell index)
    cells_rc, inverse = np.unique(
        np.column_stack([vx, vy]), axis=0, return_inverse=True
    )
    cell_centers = cells_rc * voxel_2d + voxel_2d / 2.0
    log(f"  2-D voxels: {len(cell_centers):,}")

    # ── DBSCAN on voxel centres ───────────────────────────────────────────────
    db = DBSCAN(eps=eps, min_samples=min_voxels,
                algorithm="ball_tree", n_jobs=-1)
    cell_cluster_ids = db.fit_predict(cell_centers)

    n_clusters = int(cell_cluster_ids.max()) + 1
    log(f"  Clusters found: {n_clusters}  (noise voxels: {(cell_cluster_ids == -1).sum():,})")

    # ── collect original points per cluster ───────────────────────────────────
    # Map each obstacle point to its cluster via its voxel
    point_cluster_ids = cell_cluster_ids[inverse]  # shape (n_obs_pts,)

    clusters = []
    for cid in range(n_clusters):
        pts = obs_xyz[point_cluster_ids == cid]
        clusters.append(pts)

    return clusters


# ── polygons ──────────────────────────────────────────────────────────────────

def clusters_to_polygons(
    clusters: list[np.ndarray],
    crs: str,
    min_area: float = MIN_AREA,
    max_area: float = MAX_AREA,
) -> gpd.GeoDataFrame:
    """
    Convert point clusters to convex-hull polygons, filtered by area.

    Convex hull is used instead of alpha-shape for speed; it's accurate enough
    for obstacle footprints and orders of magnitude faster at city scale.
    """
    polys = []
    for pts in clusters:
        if len(pts) < 3:
            continue
        hull = MultiPoint(pts[:, :2]).convex_hull
        if not hull.is_valid or hull.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        area = hull.area
        if min_area <= area <= max_area:
            polys.append(hull)

    return gpd.GeoDataFrame(geometry=polys, crs=crs)
