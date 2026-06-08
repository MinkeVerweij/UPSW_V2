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
    max_area=float('inf'),
    max_start_height=2.0,
    exclude_classes=None,
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
    max_area : float (default inf)
        No upper area cap by default — large clusters (building facade
        fragments, long walls) are kept and can be manually reviewed or
        filtered downstream.  Pass an explicit value to reinstate the cap.
    max_start_height : float (default 2.0 m)
        Clusters whose *lowest* point is above this height above AHN ground
        are discarded.  Removes rooftop fragments and high overhangs that
        cannot block the street.  Because ``height_ag`` is relative to the
        local AHN ground surface, the threshold is terrain-independent.
    exclude_classes : list of int, optional
        Additional label codes to exclude from obstacle candidates, on top of
        the built-in ground/road/building/cable/armatuur set.  Use this to
        remove classes handled separately by ``extract_bgt_objects()`` (e.g.
        trees, street lights) so their large point masses don't form
        oversized DBSCAN clusters.
        Default: ``[Labels.TREE, Labels.STREET_LIGHT]``
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

    # ── obstacle candidates ───────────────────────────────────────────────────
    # Always exclude structural labels. Also exclude any classes handled
    # individually by extract_bgt_objects() — their large point masses would
    # otherwise form oversized DBSCAN blobs (e.g. entire tree canopies).
    if exclude_classes is None:
        exclude_classes = [Labels.TREE, Labels.STREET_LIGHT]
    _EXCLUDE = np.array([
        Labels.GROUND, Labels.ROAD, Labels.BUILDING,
        Labels.CABLE, Labels.TRAM_CABLE, Labels.ARMATUUR,
        *exclude_classes,
    ], dtype=np.int32)
    obs_mask = (~np.isin(labels, _EXCLUDE)) & (heights > height_threshold)
    obs_idx = np.where(obs_mask)[0]
    obs_xyz = xyz[obs_idx]
    obs_rgb = rgb[obs_idx]
    obs_h   = heights[obs_idx]
    log(f"  Obstacle candidates: {len(obs_xyz):,}  "
        f"(BGT-labeled: {int((labels[obs_idx] > 0).sum()):,}, "
        f"unknown: {int((labels[obs_idx] == 0).sum()):,})")

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
    drop_min_pts = drop_area_small = drop_area_large = drop_height = 0

    for cid in range(n_raw_clusters):
        mask = point_cluster_ids == cid
        pts_xyz = obs_xyz[mask]
        pts_rgb = obs_rgb[mask]
        pts_h   = obs_h[mask]

        if len(pts_xyz) < MIN_PTS:
            drop_min_pts += 1
            continue

        # Area filter using convex hull
        try:
            area = MultiPoint(pts_xyz[:, :2]).convex_hull.area
        except Exception:
            area = 0.0
        if area < min_area:
            drop_area_small += 1
            continue
        if area > max_area:
            drop_area_large += 1
            continue

        # Drop clusters fully above street level (height above AHN ground)
        if pts_h.min() > max_start_height:
            drop_height += 1
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

    log(f"  Saved {saved}/{n_raw_clusters} clusters  "
        f"(dropped: {drop_min_pts} too few pts, "
        f"{drop_area_small} area too small (<{min_area} m²), "
        f"{drop_area_large} area too large (>{max_area} m²), "
        f"{drop_height} above {max_start_height} m)")

    inventory_df = pd.DataFrame(rows)
    inv_path = out_dir / f"inventory_{tilecode}.csv"
    inventory_df.to_csv(inv_path, index=False)
    log(f"  Inventory: {inv_path}")

    return inventory_df


# Per-class DBSCAN parameters for BGT object extraction.
# Trees have large, irregular crowns so need a wider eps.
# Poles/lights/signs are small so a tight eps keeps neighbours separate.
_BGT_OBJ_PARAMS = {
    Labels.TREE:          {'eps': 2.0,  'min_voxels': 10},
    Labels.STREET_LIGHT:  {'eps': 0.8,  'min_voxels': 3},
    Labels.TRAFFIC_LIGHT: {'eps': 0.8,  'min_voxels': 3},
    Labels.TRAFFIC_SIGN:  {'eps': 0.8,  'min_voxels': 3},
    Labels.CAR:           {'eps': 1.5,  'min_voxels': 5},
    Labels.BOLLARD:       {'eps': 0.5,  'min_voxels': 3},
}
_BGT_OBJ_DEFAULT = {'eps': 1.0, 'min_voxels': 3}


def extract_bgt_objects(
    laz_path,
    out_dir,
    tilecode,
    label_classes=None,
    voxel_2d=VOXEL_2D,
    min_area=0.01,
    max_area=200.0,
    max_start_height=2.0,
    verbose=True,
):
    """
    Extract individual BGT-labeled objects as per-object NPZs for training.

    Unlike ``extract_and_save_clusters()``, this function operates on one label
    class at a time.  Only points with that exact label enter the DBSCAN, so
    neighbouring objects of a different class (e.g. a bike next to a tree) are
    never merged into the same NPZ.  The BGT label is the hard object boundary.

    Parameters
    ----------
    laz_path : path-like
        BGT-labeled LAZ file (same input as extract_and_save_clusters).
    out_dir : path-like
        Same cluster output directory; new NPZs are appended and the existing
        ``inventory_<tilecode>.csv`` is updated in place.
    tilecode : str
    label_classes : list of int, optional
        BGT label codes to extract.
        Default: ``[Labels.TREE, Labels.STREET_LIGHT]``
    voxel_2d : float
        2-D voxel side length for footprint reduction before DBSCAN.
    min_area : float (default 0.01 m²)
        Minimum convex-hull footprint area.  Set low because pole-type objects
        (street lights, traffic signs) have a tiny 2D footprint — a pole with
        radius 0.15 m has area ≈ 0.07 m², so the 0.1 m² default used for
        obstacle clusters would incorrectly discard them.
    max_area : float (default 200 m²)
        Maximum convex-hull footprint area.  Trees can be large; this default
        is intentionally more generous than the obstacle cluster default.
    max_start_height : float (default 2.0 m)
        Objects whose lowest point is above this height above AHN ground are
        discarded (terrain-relative, same convention as extract_and_save_clusters).
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Inventory rows added by this call (subset of the updated CSV).
    """
    if label_classes is None:
        label_classes = [Labels.TREE, Labels.STREET_LIGHT]

    def log(msg):
        if verbose:
            print(msg)

    laz_path = Path(laz_path)
    out_dir  = Path(out_dir)
    tile_dir = out_dir / tilecode
    tile_dir.mkdir(parents=True, exist_ok=True)

    # ── load point cloud ──────────────────────────────────────────────────────
    log(f"[BGT objects] Loading {laz_path} …")
    pc = laspy.read(laz_path)

    xyz = np.column_stack([
        np.asarray(pc.x, dtype=np.float64),
        np.asarray(pc.y, dtype=np.float64),
        np.asarray(pc.z, dtype=np.float64),
    ])
    has_label = "label" in pc.point_format.extra_dimension_names
    labels    = np.asarray(pc.label, dtype=np.int32) if has_label else np.zeros(len(xyz), dtype=np.int32)
    has_rgb   = all(c in pc.point_format.dimension_names for c in ("red", "green", "blue"))
    if has_rgb:
        rgb = np.column_stack([
            np.asarray(pc.red,   dtype=np.float32),
            np.asarray(pc.green, dtype=np.float32),
            np.asarray(pc.blue,  dtype=np.float32),
        ]) / 65280.0
    else:
        rgb = np.zeros((len(xyz), 3), dtype=np.float32)

    # ── ground grid + heights ─────────────────────────────────────────────────
    grid, gx_min, gy_min = build_ground_grid(xyz, labels)
    if grid is None:
        log("  No ground points — skipping.")
        return pd.DataFrame()
    heights = compute_heights(xyz, grid, gx_min, gy_min)

    # ── continue cluster index from existing NPZs in tile_dir ─────────────────
    existing = sorted(tile_dir.glob("cluster_*.npz"))
    next_idx = len(existing)

    # ── load existing inventory to append to ─────────────────────────────────
    inv_path = out_dir / f"inventory_{tilecode}.csv"
    if inv_path.exists():
        existing_inv = pd.read_csv(inv_path)
    else:
        existing_inv = pd.DataFrame()

    new_rows = []

    # ── per-label extraction ──────────────────────────────────────────────────
    for lbl_code in label_classes:
        class_idx = np.where(labels == lbl_code)[0]
        if len(class_idx) == 0:
            log(f"  label {lbl_code}: no points, skipping.")
            continue

        class_xyz = xyz[class_idx]
        class_rgb = rgb[class_idx]
        class_h   = heights[class_idx]

        dbp = _BGT_OBJ_PARAMS.get(lbl_code, _BGT_OBJ_DEFAULT)
        eps, min_vox = dbp['eps'], dbp['min_voxels']

        # 2-D DBSCAN on voxelised footprint of this label's points only
        vx = np.floor(class_xyz[:, 0] / voxel_2d).astype(np.int32)
        vy = np.floor(class_xyz[:, 1] / voxel_2d).astype(np.int32)
        cells_rc, inverse = np.unique(np.column_stack([vx, vy]), axis=0, return_inverse=True)
        cell_centers = cells_rc * voxel_2d + voxel_2d / 2.0

        db = DBSCAN(eps=eps, min_samples=min_vox, algorithm="ball_tree", n_jobs=-1)
        cell_cluster_ids  = db.fit_predict(cell_centers)
        point_cluster_ids = cell_cluster_ids[inverse]

        n_instances = int(cell_cluster_ids.max()) + 1
        log(f"  label {lbl_code}: {len(class_idx):,} pts → {n_instances} instances")

        saved_this_class = 0
        for cid in range(n_instances):
            cmask   = point_cluster_ids == cid
            pts_xyz = class_xyz[cmask]
            pts_rgb = class_rgb[cmask]
            pts_h   = class_h[cmask]

            if len(pts_xyz) < MIN_PTS:
                continue

            try:
                area = MultiPoint(pts_xyz[:, :2]).convex_hull.area
            except Exception:
                area = 0.0
            if not (min_area <= area <= max_area):
                continue

            if pts_h.min() > max_start_height:
                continue

            cx = float(pts_xyz[:, 0].mean())
            cy = float(pts_xyz[:, 1].mean())
            xyz_centered = pts_xyz.astype(np.float32)
            xyz_centered[:, 0] -= cx
            xyz_centered[:, 1] -= cy

            label_src   = _label_source(lbl_code)
            cluster_idx = next_idx
            npz_path    = tile_dir / f"cluster_{cluster_idx:04d}.npz"

            np.savez_compressed(
                npz_path,
                xyz_centered = xyz_centered,
                rgb_norm     = pts_rgb.astype(np.float32),
                height_ag    = pts_h.astype(np.float32),
                centroid_xy  = np.array([cx, cy], dtype=np.float64),
                label        = np.int32(lbl_code),
                label_frac   = np.float32(1.0),
                label_source = np.bytes_(label_src),
                n_raw_pts    = np.int32(len(pts_xyz)),
                area_m2      = np.float32(area),
                tilecode     = np.bytes_(tilecode),
                cluster_idx  = np.int32(cluster_idx),
            )

            new_rows.append({
                "tilecode":           tilecode,
                "cluster_idx":        cluster_idx,
                "npz_path":           str(npz_path),
                "label":              lbl_code,
                "label_frac":         1.0,
                "label_source":       label_src,
                "final_label":        None,
                "label_source_final": None,
                "needs_review":       False,
                "n_raw_pts":          len(pts_xyz),
                "area_m2":            round(area, 3),
                "centroid_x":         round(cx, 2),
                "centroid_y":         round(cy, 2),
                "timestamp":          None,
            })
            next_idx         += 1
            saved_this_class += 1

        log(f"    → {saved_this_class} saved.")

    if not new_rows:
        return pd.DataFrame()

    new_df   = pd.DataFrame(new_rows)
    combined = pd.concat([existing_inv, new_df], ignore_index=True)
    combined.to_csv(inv_path, index=False)
    log(f"  Inventory updated: {inv_path}  ({len(new_rows)} new rows)")
    return new_df


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
