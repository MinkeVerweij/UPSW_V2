"""YOLOv8 object detection wrapper with UPSW label mapping."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

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
    "potted plant":  30,   # TREE (proxy for small shrub/plant)
    "umbrella":      91,   # TERRACE (proxy for terrace umbrella)
    "chair":         80,   # BENCH (proxy)
}

# UPSW label codes whose LiDAR clusters should NOT be expected to appear in
# panorama imagery (e.g. ground, building interior) — skip them during cross-ref.
SKIP_LIDAR_LABELS = {0, 1, 9, 10, 11, 70, 79, 99}


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


def detect_patches(img_patches, patch_params_list, model, conf=0.35):
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
        Confidence threshold.

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
                conf_val = float(boxes.conf[i].item())
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
        Must have columns: centroid_x, centroid_y, cluster_id, label (or auto_label).
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
    label_col = "final_label" if "final_label" in inv.columns else "auto_label"

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
            "geometry": Point(det.x_rd, det.y_rd),
        })
    return gpd.GeoDataFrame(rows, crs=crs)
