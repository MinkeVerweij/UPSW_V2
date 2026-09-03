"""YOLOv8 object detection wrapper with UPSW label mapping."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Mapping from COCO class name → UPSW label code
# Classes not in COCO (bollard, rubbish bin, bicycle rack, etc.) will appear
# as unmatched panorama detections and are candidates for LiDAR-miss flagging.
COCO_TO_UPSW = {
    "bicycle":       44,   # BICYCLE
    "motorcycle":    45,   # SCOOTER
    "car":           40,   # CAR
    "truck":         40,   # CAR
    "bus":           40,   # CAR
    "bench":         80,   # BENCH
    "traffic light": 61,   # TRAFFIC_LIGHT
    "stop sign":     62,   # TRAFFIC_SIGN
    "parking meter": 85,   # PARKING_METER
    "potted plant":  31,   # Potted plant (small shrub/plant in front of a house,
                            # not a real tree — was wrongly proxied to Tree (30)
                            # before, mislabeling shrubbery)
    "umbrella":      91,   # TERRACE (proxy for terrace umbrella)
    "chair":         80,   # BENCH (proxy)
}

# UPSW label codes whose LiDAR clusters should NOT be expected to appear in
# panorama imagery (e.g. ground, building interior) — skip them during cross-ref.
SKIP_LIDAR_LABELS = {0, 1, 9, 10, 11, 70, 79, 99}

# Generous max-plausible cluster footprint area (m²) per UPSW label, used by
# label_clusters_from_panorama() to refuse implausible matches — e.g. a giant
# AHN-building-gap fragment (see utils/ahn_utils.py) sitting near a real
# parked bike shouldn't get labeled "Bicycle" just because it's the nearest
# cluster. Generous enough to cover the broadest COCO source class mapped to
# each label (e.g. 40/Car also catches truck/bus via COCO_TO_UPSW), and to
# allow real large objects (e.g. 30/Tree covers a big mature crown).
MAX_PLAUSIBLE_AREA_M2 = {
    # Bicycle: raised from 1.5 to 8.0. A single bike is ~1.5 m2, but a real
    # cluster very often DBSCANs several bunched bikes together as one blob
    # (confirmed real examples: cluster 10 at 4.29 m2, cluster 44 at 2.96 m2,
    # both genuine multi-bike racks) — extremely common street furniture in
    # Amsterdam. 1.5 rejected these as "too large" and either left them
    # Unknown or let a farther, wrong-class detection win instead.
    44: 12.0,   # Bicycle (raised from 8.0: a long bike rack can legitimately
                # DBSCAN into one 12 m2 cluster)
    45: 3.0,    # Scooter/Motorcycle
    40: 40.0,   # Car (generous: covers truck/bus)
    80: 3.0,    # Bench (also 'chair' proxy)
    61: 1.0,    # Traffic light
    62: 1.0,    # Traffic sign
    85: 1.0,    # Parking meter
    30: 150.0,  # Tree (can have a large mature crown)
    31: 2.0,    # Potted plant / small shrub ('potted plant' proxy)
    91: 60.0,   # Terrace ('umbrella' proxy)
}
_DEFAULT_MAX_PLAUSIBLE_AREA_M2 = 10.0  # fallback if COCO_TO_UPSW grows new targets

# Min-plausible cluster footprint area (m²) for the motorised-vehicle labels,
# a second independent guard alongside MAX_PLAUSIBLE_AREA_M2. Real example
# found in review: a 1.26 m² cluster of red-scan points (two parked bikes)
# matched to "Car" because a genuine, high-confidence car detection was
# triangulated only ~0.7 m from the bike cluster's centroid (a car parked
# right next to a bike rack, as commonly happens on an Amsterdam street) —
# well within match_radius_m, so distance alone can't tell them apart.
# MAX_PLAUSIBLE_AREA_M2 doesn't catch this either: BGT-confirmed real Car
# clusters in this pipeline are themselves often tiny fragments (as small as
# 0.29 m², since the obstacle extraction only keeps a sliver of an occluded
# parked car), so the plausible range for "Car" genuinely overlaps bike-sized
# areas. This floor is intentionally *stricter* than what real BGT-matched
# cars show: BGT matching is a separate, more trustworthy pathway (matched
# against an official footprint polygon, not nearest-YOLO-detection), so a
# real small car fragment will still get labeled correctly there even if the
# panorama path is conservative and leaves it Unknown for manual review.
# No entry for 44/Bicycle — no report yet of the reverse mistake, and its
# MAX is already tight (1.5 m²).
MIN_PLAUSIBLE_AREA_M2 = {
    40: 2.0,   # Car (lowered from 3.0: only the front or back bumper is
               # sometimes captured in the point cloud, occluded by other
               # parked cars — a real fragment can be this small)
    45: 0.5,   # Scooter/Motorcycle
}
_DEFAULT_MIN_PLAUSIBLE_AREA_M2 = 0.0  # no floor unless a label is listed above

# Max-plausible height above ground (m) per UPSW label — a second, independent
# check alongside MAX_PLAUSIBLE_AREA_M2. Real examples found in review: a
# lamppost (~11m tall, ~25m² footprint from scan noise), a stray tall point
# merged into a bike rack cluster (~11m), and a tree (~11.5m) all had
# plausible-looking *area* for "Car" but an obviously wrong *height* profile
# — a real car/bike/bench is short and wide, not tall and thin, so height is
# a strong, cheap discriminator area alone misses. No entry for 30/Tree —
# height is exactly what's expected to vary there.
MAX_PLAUSIBLE_HEIGHT_M = {
    44: 2.0,   # Bicycle
    45: 2.0,   # Scooter/Motorcycle
    40: 4.0,   # Car (generous: covers truck/bus roof height)
    80: 1.5,   # Bench
    61: 5.0,   # Traffic light (pole + head)
    62: 5.0,   # Traffic sign (pole + head)
    85: 2.5,   # Parking meter
    91: 3.0,   # Terrace (umbrella height)
    31: 2.5,   # Potted plant / small shrub
}
_DEFAULT_MAX_PLAUSIBLE_HEIGHT_M = 3.0  # fallback if COCO_TO_UPSW grows new targets


@dataclass
class Detection:
    """A single object detection result."""
    bbox_xyxy: tuple          # (x1, y1, x2, y2) in patch pixel coords
    class_name: str           # COCO class name
    confidence: float
    upsw_label: Optional[int] # UPSW label code, or None if unmapped
    patch_idx: int            # index of the source patch
    # Filled in after ground projection:
    x_rd: Optional[float] = None
    y_rd: Optional[float] = None
    # Snapshot of the independent single-camera/flat-plane ground position,
    # taken before multi-camera triangulation may overwrite x_rd/y_rd — kept
    # around so the two placement methods can be compared/plotted.
    x_rd_single: Optional[float] = None
    y_rd_single: Optional[float] = None
    # Cached 3D ray (camera origin + unit direction), set once at first-pass
    # projection time and reused by multi-camera triangulation so the patch
    # geometry doesn't need to be recomputed later.
    ray_origin: Optional[Tuple[float, float, float]] = None
    ray_dir: Optional[Tuple[float, float, float]] = None
    # Filled in by utils.triangulation.triangulate_detections:
    z_rd: Optional[float] = None
    n_views: int = 1
    triangulated: bool = False


def map_coco_to_upsw(class_name):
    """Return UPSW label code for a COCO class name, or None if not mapped."""
    return COCO_TO_UPSW.get(class_name.lower())


def load_detector(weights="yolov8m.pt"):
    """
    Load a YOLOv8 model.

    On first call this downloads the weights (~50 MB for yolov8m).

    Parameters
    ----------
    weights : str
        Model name or path. COCO-pretrained options:
        'yolov8n.pt' (nano, fast), 'yolov8m.pt' (medium, default),
        'yolov8x.pt' (extra-large, most accurate).

    Returns
    -------
    ultralytics.YOLO model
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is not installed. Run: pip install ultralytics"
        )
    logger.info(f"Loading YOLOv8 model: {weights}")
    return YOLO(weights)


# utils.panorama_geometry.make_patches(heading_deg, n_horizontal=4, ...) builds
# patches at yaw = heading + i*(360/n_horizontal), so with the n_horizontal=4
# used everywhere in this pipeline (notebooks 4.9 and 9), patch_idx=2 is always
# yaw = heading+180 deg -- i.e. looking straight back along the vehicle's own
# direction of travel. The Cyclomedia mapping vehicle has its own panoramic
# camera mounted on a roof rack toward the rear, so that mast/housing sits
# somewhere in that same backward-facing patch of every single panorama,
# regardless of real heading or location. Confirmed visually across FOUR
# capture sessions on three different tiles (scratch/pm_patch2_*.png,
# scratch/rig_check*.png) -- always the same white camera-on-a-post rig, and
# every single sighting was YOLO-classified as "parking meter":
#   2025_10_02_08_47_34 (tile 119900_489300): x:[210,405] y:[354,446], 90 dets
#   2025_09_29_05_45_16 (tile 120300_488900): x:[109,134] y:[325,393]
#   2025_10_02_09_32_34 (tile 120300_488900): x:[128,168] y:[372,433]
#   2025_09_29_07_20_07 (tile 120300_489300): x:[98,197]  y:[368,437]
# Because it's physically fixed to the camera rig, its triangulated ground
# position always lands just behind/beside the vehicle -- right where real
# curbside objects (parked bikes, cars, containers) also are -- so it was
# silently overwriting real cluster labels with "Parking meter".
#
# The rig's *vertical* screen position is fairly stable session to session
# (a fixed camera height/mount), but its *horizontal* position drifts a lot
# between sessions -- likely heading/compass calibration drift on the
# recording vehicle, not the rig actually moving. A first fix just padded a
# box around the single best-evidenced session and excluded every class
# inside it; widening that box to cover all four sessions' positions was
# tested and rejected -- checked against real tile data, it also silently
# dropped 132-166 genuine "car"/"bicycle" detections per tile that happen to
# sit in that same region (real objects directly behind the vehicle are
# common and not rig-related; the wider box has no way to tell them apart by
# position or size alone -- rig and real nearby objects overlap in both).
# Since every confirmed rig sighting across all four sessions was
# classified "parking meter" and nothing else, the safe version restricts
# the wide-region exclusion to that class specifically -- real cars/bikes
# keep passing through untouched, only spurious "parking meter" hits in the
# rig's zone get dropped. Revisit if the rig ever turns up misclassified as
# something else.
SELF_VEHICLE_PATCH_IDX = 2
SELF_VEHICLE_BBOX = (85, 300, 430, 470)  # (x1,y1,x2,y2) px, union of all four
                                          # confirmed sessions' envelopes,
                                          # padded -- see SELF_VEHICLE_CLASSES
SELF_VEHICLE_CLASSES = {"parking meter"}  # only these COCO classes are dropped
                                           # inside SELF_VEHICLE_BBOX -- every
                                           # other class passes through even
                                           # inside the box (see comment above)

# Per-class confidence floor, applied on top of the global `conf` threshold
# passed to detect_patches(). "parking meter" is a chronically weak COCO
# class here even outside the self-vehicle-rig zone: of 9 detections left
# across 3 test tiles after the rig fix above, only 1 (conf 0.79) was a real
# piece of boxy street furniture at all (and even that one turned out to be
# an EV charger, not a parking meter -- COCO has no class for either, so
# "parking meter" is the nearest available guess for both and a confidence
# floor can't fully separate them). The other 8 (conf 0.43-0.69) were a
# parked car's roof, a traffic light, a waste container, a rubbish bin, and
# privacy-blurred pedestrians/cyclists -- the anonymization blur strips
# enough shape detail that YOLO repeatedly guesses "parking meter" for a
# generic blurred vertical shape. 0.70 cleanly separates the one plausible
# hit from all 8 false positives on every tile checked so far.
MIN_CONF_BY_CLASS = {"parking meter": 0.70}


def detect_patches(img_patches, patch_params_list, model, conf=0.35,
                    exclude_self_vehicle=True):
    """
    Run YOLOv8 detection on a list of perspective image patches.

    Parameters
    ----------
    img_patches : list of np.ndarray
        Each array shape (H, W, 3) uint8.
    patch_params_list : list of PatchParams
        One entry per patch (same order as img_patches).
    model : ultralytics.YOLO
    conf : float
        Confidence threshold applied by the model itself. Classes listed in
        MIN_CONF_BY_CLASS are additionally required to clear their own,
        stricter floor -- see the module-level comment above.
    exclude_self_vehicle : bool
        Drop a detection whose class is in SELF_VEHICLE_CLASSES and whose
        bbox centre falls inside SELF_VEHICLE_BBOX on
        patch_idx == SELF_VEHICLE_PATCH_IDX -- see the module-level comment
        above. Only meaningful when patches were generated with
        n_horizontal=4 (true for every current caller); set to False if
        that assumption doesn't hold, e.g. a different patch count.

    Returns
    -------
    list of Detection
    """
    detections = []
    for patch_idx, (patch, pp) in enumerate(zip(img_patches, patch_params_list)):
        results = model.predict(patch, conf=conf, verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = tuple(float(v) for v in boxes.xyxy[i].tolist())
                cls_id = int(boxes.cls[i].item())
                cls_name = result.names[cls_id]

                if (exclude_self_vehicle and patch_idx == SELF_VEHICLE_PATCH_IDX
                        and cls_name in SELF_VEHICLE_CLASSES):
                    ex1, ey1, ex2, ey2 = SELF_VEHICLE_BBOX
                    cx = (xyxy[0] + xyxy[2]) / 2
                    cy = (xyxy[1] + xyxy[3]) / 2
                    if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                        continue

                conf_val = float(boxes.conf[i].item())
                if conf_val < MIN_CONF_BY_CLASS.get(cls_name, 0.0):
                    continue
                detections.append(Detection(
                    bbox_xyxy=xyxy,
                    class_name=cls_name,
                    confidence=conf_val,
                    upsw_label=map_coco_to_upsw(cls_name),
                    patch_idx=patch_idx,
                ))
    return detections


def cross_reference(detections, cluster_inventory_df, match_radius_m=2.5):
    """
    Match panorama detections to existing LiDAR cluster centroids.

    Parameters
    ----------
    detections : list of Detection
        Each must have x_rd and y_rd set (from ground projection).
    cluster_inventory_df : pd.DataFrame
        Must have columns: centroid_x, centroid_y, cluster_id, label
        (BGT/DBSCAN auto-label from utils.cluster_io) and optionally
        final_label (human-confirmed, from 4.5. Enhanced Labeling Review).
    match_radius_m : float
        Maximum 2D distance to consider a match.

    Returns
    -------
    matches : list of dict
        Each dict: {detection, lidar_cluster_row, distance_m}
    unmatched_detections : list of Detection
        Panorama detections with no nearby LiDAR cluster (potential LiDAR miss).
    unmatched_clusters : pd.DataFrame
        LiDAR clusters not matched by any detection.
    """
    import pandas as pd

    inv = cluster_inventory_df.copy()
    # Prefer final_label (human-confirmed) only where it's actually been
    # filled in — it starts as an all-null column, so blindly preferring it
    # over label (the real BGT/DBSCAN auto-label) would silently compare
    # every row against NaN.
    if "final_label" in inv.columns and inv["final_label"].notna().any():
        label_col = "final_label"
    elif "label" in inv.columns:
        label_col = "label"
    else:
        label_col = "auto_label"

    # Filter out labels we don't expect to see in imagery
    inv = inv[~inv[label_col].isin(SKIP_LIDAR_LABELS)].reset_index(drop=True)

    matched_cluster_indices = set()
    matches = []
    unmatched_detections = []

    for det in detections:
        if det.x_rd is None or det.y_rd is None:
            continue
        dx = inv["centroid_x"].values - det.x_rd
        dy = inv["centroid_y"].values - det.y_rd
        dists = np.hypot(dx, dy)
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[nearest_idx]

        if nearest_dist <= match_radius_m:
            matches.append({
                "detection": det,
                "lidar_cluster": inv.iloc[nearest_idx],
                "distance_m": nearest_dist,
                "label_match": (det.upsw_label == inv.iloc[nearest_idx][label_col]),
            })
            matched_cluster_indices.add(nearest_idx)
        else:
            unmatched_detections.append(det)

    unmatched_clusters = inv.drop(index=list(matched_cluster_indices))
    return matches, unmatched_detections, unmatched_clusters


def label_clusters_from_panorama(inventory_df, detections, match_radius_m=3.0,
                                  label_frac=0.5, enforce_plausibility=True,
                                  min_low_points=10, low_point_height_m=2.0):
    """
    Suggest labels for fully-unresolved LiDAR clusters (label == 0) using
    nearby panorama YOLO detections, for human confirmation.

    Only touches rows where label == 0 — clusters with a weak BGT label
    (label_frac in [0.40, 0.60)) already have a human-reviewable candidate
    from BGT and are left alone. Writes into label/label_frac/label_source
    (NOT final_label), so matched clusters land in
    4.5. Enhanced Labeling Review's "Review" mode (prior label shown,
    needs_review stays True) rather than being treated as already-confirmed.

    Before accepting the nearest detection within match_radius_m, checks the
    cluster against three independent plausibility filters for that
    detection's class: area_m2 vs MAX_PLAUSIBLE_AREA_M2 and
    MIN_PLAUSIBLE_AREA_M2, and the 98th-percentile height above ground
    (loaded from the cluster's NPZ, not the raw max — see below) vs
    MAX_PLAUSIBLE_HEIGHT_M. A giant cluster (e.g. an
    AHN-building-classification-gap fragment, see utils/ahn_utils.py)
    sitting nearest to a real parked bike shouldn't get labeled "Bicycle"
    just because it's the closest centroid — and the max-area check alone
    doesn't catch everything: a lamppost, a stray tall point merged into a
    bike rack cluster, and a tree have each been observed with a
    "Car"-plausible *area* but an obviously wrong *height* (up to ~11m,
    vs. a real car's 1.5-2m). The height check uses a 98th percentile
    rather than np.max(): a 2-3 parked bikes cluster has been observed
    matched to "Car" on height because a *single* stray point reached 3.46m
    while 99.8% of the cluster (1259/1262 points) sat under 1.31m — a raw
    max lets one noisy point buy the whole cluster a pass; the 98th
    percentile doesn't, while still rejecting a lamppost/tree, since their
    real height is carried by many points along the structure, not one.
    A 1.26 m² cluster of two parked bikes has separately been observed
    matched to "Car" because a genuine car was triangulated to a position
    <1m from the bike cluster (parked right next to the bike rack) — well
    within a plausible area *and* height for "Car", but far too small a
    footprint to actually be one, hence MIN_PLAUSIBLE_AREA_M2. Implausible
    matches are left unresolved rather than falling back to the
    next-nearest detection, matching the rest of this function's
    single-nearest-neighbor design. Tree (30) has no height ceiling —
    height is exactly what's expected to vary there.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        Cluster inventory (utils.cluster_io schema): must have columns
        centroid_x, centroid_y, label, label_frac, label_source, area_m2,
        npz_path.
    detections : list of Detection
        Triangulated panorama detections (x_rd/y_rd set) for the same tile,
        e.g. from utils.triangulation.triangulate_detections.
    match_radius_m : float
        Maximum 2D distance to consider a match. Matches the MATCH_R_M_BGT
        convention used elsewhere for LiDAR<->panorama matching.
    label_frac : float
        Value written for matched rows — must land inside 4.5's Review band
        (REVIEW_THRESH..AUTO_LABEL_THRESH, default 0.40-0.60) rather than
        being read as a confident auto-label.
    enforce_plausibility : bool
        When False, skips all three plausibility filters below and accepts
        the nearest detection within match_radius_m unconditionally —
        maximizes how many clusters get *some* suggested label, at the cost
        of more wrong suggestions, e.g. a giant AHN-gap fragment could get
        labeled "Bicycle" just for being nearest. Reasonable to run with
        this off since every suggestion still lands in 4.5's Review band for
        a human to confirm or override, never written to final_label
        directly — set True (default) to restore the filters once recall
        has been prioritized enough / precision matters more again.
    min_low_points : int
        Minimum number of a candidate cluster's points that must sit below
        low_point_height_m for it to be considered at all — independent of
        enforce_plausibility, and not per matched-class like the height
        filter above. For sidewalk-obstacle-focused work, a cluster that's
        almost entirely tall (e.g. 116 points, 15.3m max, only 3 below 2m —
        a lamppost/tree fragment, not a sidewalk object) shouldn't compete
        for a label at all, regardless of what it might loosely match by
        area. Set to 0 to disable this filter entirely.
    low_point_height_m : float
        Height-above-ground threshold (metres) used by min_low_points.

    Returns
    -------
    updated : pd.DataFrame
        Copy of inventory_df with matched rows updated.
    summary : dict
        {"n_candidates", "n_matched", "n_too_large", "n_too_small",
         "n_too_tall", "n_not_sidewalk_scale", "n_unresolved"}
    """
    inv = inventory_df.copy()
    all_candidates = inv.index[inv["label"] == 0]

    n_not_sidewalk_scale = 0
    if min_low_points > 0:
        candidates = []
        for idx in all_candidates:
            try:
                h = np.load(inv.at[idx, "npz_path"])["height_ag"]
                n_low = int((h < low_point_height_m).sum())
            except Exception:
                n_low = min_low_points  # can't check -- don't penalize for a load failure
            if n_low >= min_low_points:
                candidates.append(idx)
            else:
                n_not_sidewalk_scale += 1
    else:
        candidates = list(all_candidates)

    valid_dets = [d for d in detections
                  if d.x_rd is not None and d.y_rd is not None
                  and d.upsw_label is not None]

    n_candidates = len(all_candidates)
    if len(candidates) == 0 or not valid_dets:
        return inv, {"n_candidates": n_candidates, "n_matched": 0,
                      "n_too_large": 0, "n_too_small": 0, "n_too_tall": 0,
                      "n_not_sidewalk_scale": n_not_sidewalk_scale,
                      "n_unresolved": n_candidates - n_not_sidewalk_scale}

    det_xy = np.array([[d.x_rd, d.y_rd] for d in valid_dets])

    n_matched = 0
    n_too_large = 0
    n_too_small = 0
    n_too_tall = 0
    for idx in candidates:
        cx, cy = inv.at[idx, "centroid_x"], inv.at[idx, "centroid_y"]
        dists = np.hypot(det_xy[:, 0] - cx, det_xy[:, 1] - cy)
        nearest = np.argmin(dists)
        if dists[nearest] <= match_radius_m:
            det = valid_dets[nearest]

            if enforce_plausibility:
                max_area = MAX_PLAUSIBLE_AREA_M2.get(
                    det.upsw_label, _DEFAULT_MAX_PLAUSIBLE_AREA_M2)
                if inv.at[idx, "area_m2"] > max_area:
                    n_too_large += 1
                    continue

                min_area = MIN_PLAUSIBLE_AREA_M2.get(
                    det.upsw_label, _DEFAULT_MIN_PLAUSIBLE_AREA_M2)
                if inv.at[idx, "area_m2"] < min_area:
                    n_too_small += 1
                    continue

                if det.upsw_label != 30:  # Tree: height is expected to vary, no ceiling
                    max_height = MAX_PLAUSIBLE_HEIGHT_M.get(
                        det.upsw_label, _DEFAULT_MAX_PLAUSIBLE_HEIGHT_M)
                    try:
                        # 98th percentile, not .max(): a real example found in
                        # review was 2-3 parked bikes (1259/1262 points under
                        # 1.31m, a clean bike-height cluster) with a single
                        # stray point at 3.46m — plausibly under the 4.0m Car
                        # ceiling on a raw max, even though 99.8% of the cluster
                        # isn't. One noisy point shouldn't buy a whole cluster a
                        # pass; a lamppost/tree's real height is carried by many
                        # points along its structure, so this still rejects those.
                        cluster_h = float(np.percentile(
                            np.load(inv.at[idx, "npz_path"])["height_ag"], 98))
                    except Exception:
                        cluster_h = None
                    if cluster_h is not None and cluster_h > max_height:
                        n_too_tall += 1
                        continue

            inv.at[idx, "label"] = det.upsw_label
            inv.at[idx, "label_frac"] = label_frac
            inv.at[idx, "label_source"] = "panorama_yolo"
            n_matched += 1

    summary = {
        "n_candidates": n_candidates,
        "n_matched": n_matched,
        "n_too_large": n_too_large,
        "n_too_small": n_too_small,
        "n_too_tall": n_too_tall,
        "n_not_sidewalk_scale": n_not_sidewalk_scale,
        "n_unresolved": n_candidates - n_matched - n_not_sidewalk_scale,
    }
    return inv, summary


def cluster_detection_overlap(cluster_xyz_world, cam_pose, img_shape, bbox_uv_bounds):
    """
    Fraction of a cluster's own points that project inside a panorama
    detection's equirectangular bounding envelope, as seen from one camera.

    Parameters
    ----------
    cluster_xyz_world : (N, 3) array
        Cluster points in RD New (x, y, z) -- NOT centred/relative.
    cam_pose : utils.panorama_geometry.CameraPose
    img_shape : tuple
        Source equirectangular image shape (the same one cam_pose/the
        detection's patch were derived against).
    bbox_uv_bounds : (u_min, u_max, v_min, v_max)
        From utils.panorama_geometry.patch_bbox_to_equirect_bounds().

    Returns
    -------
    (fraction, n_projected) : float in [0, 1], and how many of the input
        points had a valid projection (points behind the camera are
        excluded from both the numerator and denominator, not counted
        against the fraction).
    """
    from utils.panorama_geometry import lidar_point_to_pixel

    u_min, u_max, v_min, v_max = bbox_uv_bounds
    W = img_shape[1]
    inside = 0
    total = 0
    for p in cluster_xyz_world:
        uv = lidar_point_to_pixel(p, cam_pose, img_shape)
        if uv is None:
            continue
        total += 1
        u, v = uv
        # unwrap relative to u_min -- bbox_uv_bounds is already unwrapped
        # around one seam crossing, and the box never spans more than one
        # patch's FOV (<=90 deg here), so this is unambiguous
        u_unwrapped = u_min + ((u - u_min + W / 2) % W - W / 2)
        if u_min <= u_unwrapped <= u_max and v_min <= v <= v_max:
            inside += 1
    return (inside / total if total else 0.0), total


def resolve_shared_claims(inventory_df, cam_results, pano_dir, match_radius_m=3.0,
                           winner_min_overlap=0.05, loser_max_overlap=0.01):
    """
    Resolve LiDAR clusters that both matched the *same* panorama detection
    in label_clusters_from_panorama(), using pixel overlap to tell a
    genuine multi-fragment match (e.g. several DBSCAN sub-clusters of one
    bike rack, all correctly "Bicycle") apart from an unrelated object that
    just happened to be the nearest candidate within match_radius_m (e.g. a
    shrub 2.7m from a bike-rack detection, matched purely because nothing
    closer existed).

    Real example that motivated this (tile 119900_489300): a bike-rack
    detection triangulated from 6 camera views (1.14m ray residual -- see
    utils.triangulation) was independently claimed as "nearest" by 6
    different LiDAR clusters, including 5 real bike fragments AND one shrub
    against a facade 2.71m away. 8 of 38 matched detection groups in that
    tile (21%) were claimed by 2+ clusters this way -- not a rare edge case.

    Only touches rows with label_source == "panorama_yolo" (i.e. rows just
    written by label_clusters_from_panorama() using the same detections and
    match_radius_m). For each detection claimed by 2+ such clusters,
    computes every member's own maximum overlap with that detection's
    box(es) -- across *every* contributing camera view, not just the
    highest-confidence one. That "every view" requirement matters: checking
    only the best-confidence camera produced noisy 0% readings for pairs
    that turned out to both be genuine once every view was checked (a
    single detection box often only frames one bike out of several in a
    tightly DBSCAN'd rack, so which specific box a given fragment overlaps
    depends on which camera angle is checked).

    A cluster is only reverted to Unknown if its own overlap is ~zero
    (<= loser_max_overlap) AND another cluster in the same group clears
    winner_min_overlap. A flat "drop below X%" threshold was tested and
    rejected: it wrongly dropped a genuinely correct match (one potted
    plant in a row of several, whose detection box was drawn tightly around
    a *different*, neighbouring pot -- 5.3% overlap, visually confirmed
    correct) alongside the real mismatches. Groups where no member clears
    winner_min_overlap are left untouched entirely -- there's no confident
    call to make, so none is forced.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        Cluster inventory immediately after label_clusters_from_panorama()
        has run on it (needs label/label_frac/label_source/centroid_x/
        centroid_y/npz_path columns).
    cam_results : list of (pose_dict, CameraPose, [Detection, ...])
        Same structure produced by the detect+triangulate loop in notebook
        4.9/9 -- every Detection must carry x_rd/y_rd, upsw_label,
        patch_idx, bbox_xyxy, confidence.
    pano_dir : Path
        Directory containing this tile's cached panorama ``.jpg`` files
        (only the handful involved in a shared claim are re-opened, not the
        whole tile).
    match_radius_m : float
        Must match the value passed to label_clusters_from_panorama(), so
        the same nearest-detection-per-cluster lookup is reproduced here.
    winner_min_overlap, loser_max_overlap : float
        Thresholds described above.

    Returns
    -------
    updated : pd.DataFrame
        Copy of inventory_df with resolved rows reverted to Unknown
        (label=0, label_frac=0.0, label_source="unknown").
    summary : dict
        {"n_matched", "n_shared_groups", "n_dropped", "dropped_cluster_idx"}
    """
    import numpy as np
    from PIL import Image
    from utils.panorama_geometry import make_patches, patch_bbox_to_equirect_bounds

    inv = inventory_df.copy()
    matched = inv[inv["label_source"] == "panorama_yolo"]
    empty_summary = {"n_matched": len(matched), "n_shared_groups": 0,
                      "n_dropped": 0, "dropped_cluster_idx": []}
    if len(matched) == 0:
        return inv, empty_summary

    detections = [d for _, _, dets in cam_results for d in dets
                  if d.x_rd is not None and d.upsw_label is not None]
    if not detections:
        return inv, empty_summary
    det_xy = np.array([[d.x_rd, d.y_rd] for d in detections])

    # Re-derive which exact detection each matched cluster claimed --
    # label_clusters_from_panorama() doesn't persist this, so redo its
    # nearest-neighbor lookup here (same radius, same detections).
    claims = {}
    for idx, row in matched.iterrows():
        cx, cy = row["centroid_x"], row["centroid_y"]
        dists = np.hypot(det_xy[:, 0] - cx, det_xy[:, 1] - cy)
        nearest = int(np.argmin(dists))
        if dists[nearest] > match_radius_m:
            continue
        d = detections[nearest]
        key = (round(d.x_rd, 1), round(d.y_rd, 1), d.upsw_label)
        claims.setdefault(key, []).append(idx)

    shared = {k: v for k, v in claims.items() if len(v) > 1}
    if not shared:
        empty_summary["n_shared_groups"] = 0
        return inv, empty_summary

    img_shape_cache = {}
    def _img_shape(pano_id):
        if pano_id not in img_shape_cache:
            img_shape_cache[pano_id] = np.array(
                Image.open(pano_dir / f"{pano_id}.jpg").convert("RGB")).shape
        return img_shape_cache[pano_id]

    patch_params_cache = {}
    def _patch_params(pano_id, patch_idx, heading, img_shape):
        key = (pano_id, patch_idx)
        if key not in patch_params_cache:
            pps = make_patches(heading_deg=heading, fov_h_deg=90.0, out_hw=(640, 640),
                                n_horizontal=4, pitch_deg=0.0, img_shape=img_shape)
            patch_params_cache[key] = pps[patch_idx]
        return patch_params_cache[key]

    dropped_idx = []
    for key, member_idx in shared.items():
        tx, ty, upsw = key

        # Every raw detection (one per contributing camera) that fed this
        # triangulated group -- de-duped to the highest-confidence box per
        # camera, since a camera can occasionally contribute >1 patch box.
        by_cam = {}
        for pose_d, cam_pose, dets in cam_results:
            for d in dets:
                if (d.x_rd is not None and d.upsw_label == upsw
                        and abs(d.x_rd - tx) < 0.15 and abs(d.y_rd - ty) < 0.15):
                    if (pose_d["pano_id"] not in by_cam
                            or d.confidence > by_cam[pose_d["pano_id"]][2].confidence):
                        by_cam[pose_d["pano_id"]] = (pose_d, cam_pose, d)
        if not by_cam:
            continue

        overlaps = {}
        for idx in member_idx:
            row = inv.loc[idx]
            npz = np.load(row["npz_path"])
            xyz_c, h = npz["xyz_centered"], npz["height_ag"]
            cx, cy = row["centroid_x"], row["centroid_y"]
            best = 0.0
            for pose_d, cam_pose, d in by_cam.values():
                img_shape = _img_shape(pose_d["pano_id"])
                pp = _patch_params(pose_d["pano_id"], d.patch_idx, cam_pose.heading, img_shape)
                bounds = patch_bbox_to_equirect_bounds(d.bbox_xyxy, pp)
                world_pts = np.column_stack(
                    [xyz_c[:, 0] + cx, xyz_c[:, 1] + cy, cam_pose.z - 2.0 + h])
                frac, _ = cluster_detection_overlap(world_pts, cam_pose, img_shape, bounds)
                best = max(best, frac)
            overlaps[idx] = best

        has_winner = any(v >= winner_min_overlap for v in overlaps.values())
        if not has_winner:
            continue
        for idx, frac in overlaps.items():
            if frac <= loser_max_overlap:
                inv.at[idx, "label"] = 0
                inv.at[idx, "label_frac"] = 0.0
                inv.at[idx, "label_source"] = "unknown"
                dropped_idx.append(int(inv.at[idx, "cluster_idx"]))

    return inv, {"n_matched": len(matched), "n_shared_groups": len(shared),
                 "n_dropped": len(dropped_idx), "dropped_cluster_idx": sorted(dropped_idx)}


def detections_to_geodataframe(detections, crs="EPSG:28992"):
    """
    Convert a list of Detection objects (with x_rd, y_rd set) to a GeoDataFrame.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    rows = []
    for det in detections:
        if det.x_rd is None:
            continue
        rows.append({
            "class_name": det.class_name,
            "upsw_label": det.upsw_label,
            "confidence": det.confidence,
            "z_rd": det.z_rd,
            "n_views": det.n_views,
            "triangulated": det.triangulated,
            "geometry": Point(det.x_rd, det.y_rd),
        })
    return gpd.GeoDataFrame(rows, crs=crs)
