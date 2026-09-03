"""
Cross-camera data association and multi-view triangulation for panorama
YOLO detections.

Detections of the same physical object seen from >=2 panorama cameras in a
tile are triangulated from their 3D rays instead of relying on a single ray
intersected with an assumed flat ground plane. Detections seen from only one
camera fall back to a ground position corrected with real AHN ground
elevation (instead of a flat per-camera-height offset).
"""

import itertools
import math

import numpy as np

from utils.ahn_reader import FastGridInterpolator
from utils.panorama_geometry import ray_ground_intersect, triangulate_rays


def correct_camera_elevation(cam_pose, ahn_reader, tilecode, cam_height_m=2.0):
    """
    Replace a camera's z (elevation) with a real AHN ground-elevation lookup
    plus camera height, in place, when the incoming z looks like the known
    upstream 0.0 stub (abs(z) < 1.0).

    Leaves cam_pose.z untouched if the tile has no AHN data, or if z already
    looks like a real elevation.
    """
    if abs(cam_pose.z) >= 1.0:
        return cam_pose
    try:
        tile = ahn_reader.filter_tile(tilecode)
    except FileNotFoundError:
        return cam_pose
    fast_z = FastGridInterpolator(tile["x"], tile["y"], tile["ground_surface"])
    ground_z = float(fast_z(np.array([[cam_pose.x_rd, cam_pose.y_rd]]))[0])
    # AHN ground grids have large no-data gaps (areas with no ground-level
    # LiDAR return, e.g. under buildings/trees) — leave the pose untouched
    # rather than propagate a NaN elevation into every ray cast from it.
    if not np.isnan(ground_z):
        cam_pose.z = ground_z + cam_height_m
    return cam_pose


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


def _compatible(det_a, det_b):
    """Two detections are the same candidate object if their UPSW labels
    agree, or (for COCO classes with no UPSW mapping) their raw class names
    agree. Comparing two Nones would spuriously group unrelated unmapped
    classes, so upsw_label must be non-None to count."""
    if det_a.upsw_label is not None and det_a.upsw_label == det_b.upsw_label:
        return True
    return det_a.class_name == det_b.class_name


def triangulate_detections(cam_results, tilecode, ahn_reader,
                            match_radius_m, min_baseline_m=3.0,
                            max_residual_m=1.5, cam_height_m=2.0):
    """
    Cross-camera data association + triangulation for one tile, in place.

    Parameters
    ----------
    cam_results : list of (pose_dict, CameraPose, [Detection, ...])
        Same shape as ALL_TILE_RESULTS[tilecode]. Each Detection must already
        carry ray_origin/ray_dir (cached at first-pass projection) and a
        rough x_rd/y_rd (single-ray flat-plane guess), used only to seed
        cross-camera association.
    tilecode : str
    ahn_reader : utils.ahn_reader.PolygonNPZReader
    match_radius_m : float
        Max 2D distance between rough positions to consider two cross-camera
        detections the same object.
    min_baseline_m : float
        Minimum camera separation required to attempt triangulation (rejects
        near-parallel, ill-conditioned ray pairs).
    max_residual_m : float
        Max RMS ray residual to accept a triangulation.
    cam_height_m : float
        Used only as the flat-plane fallback elevation offset when a tile
        has no AHN data to correct against.

    Mutates every Detection with x_rd, y_rd, z_rd, n_views, triangulated.
    Returns cam_results.
    """
    flat = [
        (cam_idx, det)
        for cam_idx, (_, _, dets) in enumerate(cam_results)
        for det in dets
    ]
    n = len(flat)
    if n == 0:
        return cam_results

    uf = _UnionFind(n)
    for i in range(n):
        cam_i, det_i = flat[i]
        for j in range(i + 1, n):
            cam_j, det_j = flat[j]
            if cam_i == cam_j or not _compatible(det_i, det_j):
                continue
            dist = math.hypot(det_i.x_rd - det_j.x_rd, det_i.y_rd - det_j.y_rd)
            if dist <= match_radius_m:
                uf.union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    fallback_idx = []   # flat-indices of detections needing the single-view AHN fallback
    n_triangulated_groups = 0

    for members in groups.values():
        distinct_cams = {flat[i][0] for i in members}
        if len(distinct_cams) < 2:
            fallback_idx.extend(members)
            continue

        # One ray per camera — a camera contributing >1 member to a group is
        # rare (only possible at adjacent-patch edges); use its most
        # confident detection as the representative ray.
        per_cam = {}
        for i in members:
            cam_idx, det = flat[i]
            if cam_idx not in per_cam or det.confidence > flat[per_cam[cam_idx]][1].confidence:
                per_cam[cam_idx] = i

        cam_poses = [cam_results[c][1] for c in per_cam]
        baseline = max(
            math.hypot(a.x_rd - b.x_rd, a.y_rd - b.y_rd)
            for a, b in itertools.combinations(cam_poses, 2)
        )
        if baseline < min_baseline_m:
            fallback_idx.extend(members)
            continue

        origins = [np.array(flat[i][1].ray_origin) for i in per_cam.values()]
        directions = [np.array(flat[i][1].ray_dir) for i in per_cam.values()]
        try:
            point, residual = triangulate_rays(origins, directions)
        except ValueError:
            fallback_idx.extend(members)
            continue
        if residual > max_residual_m:
            fallback_idx.extend(members)
            continue

        for i in members:
            det = flat[i][1]
            det.x_rd, det.y_rd, det.z_rd = float(point[0]), float(point[1]), float(point[2])
            det.n_views = len(distinct_cams)
            det.triangulated = True
        n_triangulated_groups += 1

    # ── Single-view fallback, corrected with real AHN ground elevation ─────
    # Batched per tile: one filter_tile()/FastGridInterpolator build, one
    # vectorised interpolate() call, mirroring utils/pole_fuser.py's pattern.
    if fallback_idx:
        try:
            tile = ahn_reader.filter_tile(tilecode)
        except FileNotFoundError:
            print(f"{tilecode}: no AHN data — {len(fallback_idx)} single-view "
                  f"detection(s) keep the flat-plane ground estimate")
            for i in fallback_idx:
                cam_idx, det = flat[i]
                det.z_rd = cam_results[cam_idx][1].z - cam_height_m
        else:
            fast_z = FastGridInterpolator(tile["x"], tile["y"], tile["ground_surface"])
            pts = np.array([[flat[i][1].x_rd, flat[i][1].y_rd] for i in fallback_idx])
            ground_zs = fast_z(pts)
            for i, ground_z in zip(fallback_idx, ground_zs):
                cam_idx, det = flat[i]
                cam_pose = cam_results[cam_idx][1]
                ground_z = float(ground_z)
                # AHN ground grids have large no-data gaps — if this
                # detection's cell is one of them, keep the flat-plane
                # estimate already computed upstream rather than propagate NaN.
                if np.isnan(ground_z):
                    det.z_rd = cam_pose.z - cam_height_m
                    continue
                result = ray_ground_intersect(cam_pose, np.array(det.ray_dir), ground_z)
                if result is not None:
                    det.x_rd, det.y_rd = result
                det.z_rd = ground_z

    n_triangulated = sum(1 for _, det in flat if det.triangulated)
    print(f"{tilecode}: {n_triangulated}/{n} detections triangulated "
          f"({n_triangulated_groups} objects, >=2 views) · "
          f"{n - n_triangulated} single-view (AHN ground)")

    return cam_results
