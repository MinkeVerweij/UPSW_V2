"""
Cluster extraction and serialization for PointNet++ training.

Extracts DBSCAN obstacle clusters from BGT-labeled LAZ files and saves each
cluster as a compact NPZ file alongside a per-tile inventory CSV.  Downstream
notebooks (labeling review, dataset preparation, training) all consume this
format.
"""

import numpy as np
import laspy
import pandas as pd
from pathlib import Path
from datetime import datetime
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.obstacle_extractor_2d import (
    build_ground_grid,
    compute_heights,
    VOXEL_2D,
    DBSCAN_EPS,
    DBSCAN_MIN_VOXELS,
    HEIGHT_THRESHOLD,
    MIN_AREA,
    MAX_AREA,
    UNKNOWN_LABEL,
)
from utils.labels import Labels


MIN_PTS = 10  # minimum raw points to keep a cluster


def extract_and_save_clusters(
    laz_path,
    out_dir,
    tilecode,
    auto_label_thresh=0.60,
    review_thresh=0.40,
    voxel_2d=VOXEL_2D,
    eps=DBSCAN_EPS,
    min_voxels=DBSCAN_MIN_VOXELS,
    height_threshold=HEIGHT_THRESHOLD,
    min_area=MIN_AREA,
    max_area=MAX_AREA,
    verbose=True,
):
    """
    Run DBSCAN on unlabeled obstacle points in a BGT-labeled LAZ file and save
    each cluster as an NPZ alongside a per-tile inventory CSV.

    Parameters
    ----------
    laz_path : path-like
        BGT-labeled LAZ file (output of notebook 2.5).
    out_dir : path-like
        Directory for per-tile cluster folder.  Creates
        ``out_dir/<tilecode>/cluster_NNN.npz`` and
        ``out_dir/inventory_<tilecode>.csv``.
    tilecode : str
    auto_label_thresh : float
        Dominant-label fraction at or above which a cluster gets auto-labeled.
    review_thresh : float
        Fraction below which the cluster is flagged as needing manual review
        even if a dominant label exists.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Per-tile inventory.
    """

    def log(msg):
        if verbose:
            print(msg)

    laz_path = Path(laz_path)
    out_dir = Path(out_dir)
    tile_dir = out_dir / tilecode
    tile_dir.mkdir(parents=True, exist_ok=True)

    # ── load point cloud ──────────────────────────────────────────────────────
    log(f"Loading {laz_path} …")
    pc = laspy.read(laz_path)

    xyz = np.column_stack([
        np.asarray(pc.x, dtype=np.float64),
        np.asarray(pc.y, dtype=np.float64),
        np.asarray(pc.z, dtype=np.float64),
    ])

    has_label = "label" in pc.point_format.extra_dimension_names
    labels = np.asarray(pc.label, dtype=np.int32) if has_label else np.zeros(len(xyz), dtype=np.int32)

    has_rgb = all(c in pc.point_format.dimension_names for c in ("red", "green", "blue"))
    if has_rgb:
        rgb = np.column_stack([
            np.asarray(pc.red,   dtype=np.float32),
            np.asarray(pc.green, dtype=np.float32),
            np.asarray(pc.blue,  dtype=np.float32),
        ]) / 65280.0
    else:
        rgb = np.zeros((len(xyz), 3), dtype=np.float32)

    log(f"  {len(xyz):,} points loaded.")

    # ── ground grid + heights ─────────────────────────────────────────────────
    grid, gx_min, gy_min = build_ground_grid(xyz, labels)
    if grid is None:
        log("  No ground points — skipping tile.")
        return pd.DataFrame()

    heights = compute_heights(xyz, grid, gx_min, gy_min)

    # ── obstacle candidates: unknown label + above height threshold ───────────
    obs_mask = (labels == UNKNOWN_LABEL) & (heights > height_threshold)
    obs_idx = np.where(obs_mask)[0]
    obs_xyz = xyz[obs_idx]
    obs_rgb = rgb[obs_idx]
    obs_h   = heights[obs_idx]
    log(f"  Obstacle candidates: {len(obs_xyz):,}")

    if len(obs_xyz) < min_voxels:
        log("  Too few candidates — skipping.")
        return pd.DataFrame()

    # ── 2-D DBSCAN on voxel centroids ─────────────────────────────────────────
    vx = np.floor(obs_xyz[:, 0] / voxel_2d).astype(np.int32)
    vy = np.floor(obs_xyz[:, 1] / voxel_2d).astype(np.int32)

    cells_rc, inverse = np.unique(
        np.column_stack([vx, vy]), axis=0, return_inverse=True
    )
    cell_centers = cells_rc * voxel_2d + voxel_2d / 2.0

    db = DBSCAN(eps=eps, min_samples=min_voxels, algorithm="ball_tree", n_jobs=-1)
    cell_cluster_ids = db.fit_predict(cell_centers)
    point_cluster_ids = cell_cluster_ids[inverse]

    n_raw_clusters = int(cell_cluster_ids.max()) + 1
    log(f"  Raw clusters: {n_raw_clusters}")

    # ── save per-cluster NPZs ─────────────────────────────────────────────────
    rows = []
    saved = 0

    for cid in range(n_raw_clusters):
        mask = point_cluster_ids == cid
        pts_xyz = obs_xyz[mask]
        pts_rgb = obs_rgb[mask]
        pts_h   = obs_h[mask]

        if len(pts_xyz) < MIN_PTS:
            continue

        # Area filter using convex hull
        try:
            area = MultiPoint(pts_xyz[:, :2]).convex_hull.area
        except Exception:
            area = 0.0
        if not (min_area <= area <= max_area):
            continue

        # Center XY on centroid; keep Z absolute (height above ground is Z-relative)
        cx, cy = float(pts_xyz[:, 0].mean()), float(pts_xyz[:, 1].mean())
        xyz_centered = pts_xyz.astype(np.float32)
        xyz_centered[:, 0] -= cx
        xyz_centered[:, 1] -= cy

        # Determine auto-label from existing BGT labels on these points
        # We need the original per-point labels for the full cloud at obs_idx[mask]
        orig_indices = obs_idx[mask]
        cluster_labels = labels[orig_indices]

        nonzero = cluster_labels[cluster_labels > 0]
        auto_label = 0
        label_frac  = 0.0
        label_source = "unknown"

        if len(nonzero) > 0:
            vals, counts = np.unique(nonzero, return_counts=True)
            dom_label = int(vals[counts.argmax()])
            dom_frac  = float(counts.max() / len(cluster_labels))

            if dom_frac >= auto_label_thresh:
                auto_label   = dom_label
                label_frac   = dom_frac
                label_source = _label_source(dom_label)
            elif dom_frac >= review_thresh:
                auto_label   = dom_label
                label_frac   = dom_frac
                label_source = "bgt_review"

        needs_review = (
            auto_label == 0
            or label_frac < auto_label_thresh
            or label_source == "bgt_review"
        )

        cluster_idx = saved
        npz_name = f"cluster_{cluster_idx:04d}.npz"
        npz_path  = tile_dir / npz_name

        np.savez_compressed(
            npz_path,
            xyz_centered = xyz_centered,
            rgb_norm     = pts_rgb.astype(np.float32),
            height_ag    = pts_h.astype(np.float32),
            centroid_xy  = np.array([cx, cy], dtype=np.float64),
            label        = np.int32(auto_label),
            label_frac   = np.float32(label_frac),
            label_source = np.bytes_(label_source),
            n_raw_pts    = np.int32(len(pts_xyz)),
            area_m2      = np.float32(area),
            tilecode     = np.bytes_(tilecode),
            cluster_idx  = np.int32(cluster_idx),
        )

        rows.append({
            "tilecode":          tilecode,
            "cluster_idx":       cluster_idx,
            "npz_path":          str(npz_path),
            "label":             auto_label,
            "label_frac":        round(label_frac, 3),
            "label_source":      label_source,
            "final_label":       None,
            "label_source_final": None,
            "needs_review":      needs_review,
            "n_raw_pts":         len(pts_xyz),
            "area_m2":           round(area, 3),
            "centroid_x":        round(cx, 2),
            "centroid_y":        round(cy, 2),
            "timestamp":         None,
        })
        saved += 1

    log(f"  Saved {saved} clusters.")

    inventory_df = pd.DataFrame(rows)
    inv_path = out_dir / f"inventory_{tilecode}.csv"
    inventory_df.to_csv(inv_path, index=False)
    log(f"  Inventory: {inv_path}")

    return inventory_df


def _label_source(label_code):
    """Map a BGT label code to a human-readable source string."""
    tree_labels    = {Labels.TREE}
    bgt_labels     = {
        Labels.CAR, Labels.STREET_LIGHT, Labels.TRAFFIC_LIGHT, Labels.TRAFFIC_SIGN,
        Labels.CITY_BENCH, Labels.RUBBISH_BIN, Labels.LARGE_CONTAINER,
        Labels.BICYCLE_RACK, Labels.BOLLARD, Labels.PARKING_METER,
        Labels.ADVERTISING_SIGN, Labels.TERRACE, Labels.CABLE, Labels.TRAM_CABLE,
    }
    if label_code in tree_labels:
        return "bomen_auto"
    if label_code in bgt_labels:
        return "bgt_auto"
    return "bgt_auto"


def build_cluster_inventory(clusters_root):
    """
    Aggregate all per-tile ``inventory_<tilecode>.csv`` files into one DataFrame.

    Parameters
    ----------
    clusters_root : path-like
        Directory containing per-tile inventory CSVs.

    Returns
    -------
    pd.DataFrame
    """
    clusters_root = Path(clusters_root)
    parts = list(clusters_root.glob("inventory_*.csv"))
    if not parts:
        return pd.DataFrame()
    dfs = [pd.read_csv(p) for p in parts]
    combined = pd.concat(dfs, ignore_index=True)
    out_path = clusters_root / "inventory.csv"
    combined.to_csv(out_path, index=False)
    return combined


def load_cluster_npz(npz_path):
    """Load a cluster NPZ and return a plain dict with decoded string fields."""
    data = np.load(npz_path, allow_pickle=False)
    result = {k: data[k] for k in data.files}
    # Decode bytes fields to str
    for key in ("label_source", "tilecode"):
        if key in result:
            val = result[key]
            if val.dtype.kind in ("S", "U"):
                result[key] = val.item().decode() if isinstance(val.item(), bytes) else str(val.item())
    return result
