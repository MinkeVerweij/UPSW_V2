"""
Camera geometry for Amsterdam equirectangular panoramas.

Coordinate conventions
----------------------
- Global frame: North-East-Up (NEU), matching RD New (EPSG:28992) axes:
    East  = +X  (RD x)
    North = +Y  (RD y)
    Up    = +Z  (elevation)
- Azimuth: degrees clockwise from North  (0=N, 90=E, 180=S, 270=W)
- Elevation: degrees above horizon (positive = up)
- Equirectangular image: u=0 → North (azimuth=0°), u=W → full 360° wrap-around
    v=0 → zenith (elevation=+90°), v=H → nadir (elevation=-90°)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CameraPose:
    """Camera position and orientation in RD New (EPSG:28992)."""
    x_rd: float
    y_rd: float
    z: float           # elevation in metres (AHN/WGS84 ellipsoidal)
    heading: float     # degrees, 0=North, 90=East (clockwise)
    pitch: float       # degrees, positive = tilted upward
    roll: float = 0.0  # degrees (usually small; ignored in projection)

    @property
    def heading_rad(self):
        return math.radians(self.heading)

    @property
    def pitch_rad(self):
        return math.radians(self.pitch)


# ---------------------------------------------------------------------------
# Direction ↔ pixel conversions
# ---------------------------------------------------------------------------

def direction_to_pixel(az_deg, el_deg, img_shape):
    """
    Map a geographic direction (azimuth, elevation) to an equirectangular pixel.

    Parameters
    ----------
    az_deg : float or np.ndarray
        Azimuth in degrees (0=North, clockwise).
    el_deg : float or np.ndarray
        Elevation in degrees above horizon.
    img_shape : (H, W) tuple
        Equirectangular image shape.

    Returns
    -------
    (u, v) : pixel coordinates (floats, may be fractional)
    """
    H, W = img_shape[:2]
    az = np.asarray(az_deg) % 360.0
    el = np.asarray(el_deg)
    u = az / 360.0 * W
    v = (90.0 - el) / 180.0 * H
    return u, v


def pixel_to_direction(u, v, img_shape):
    """
    Map equirectangular pixel (u, v) to (azimuth_deg, elevation_deg).

    Parameters
    ----------
    u, v : float or np.ndarray
        Pixel coordinates.
    img_shape : (H, W) tuple

    Returns
    -------
    (az_deg, el_deg)
    """
    H, W = img_shape[:2]
    az_deg = (np.asarray(u) / W) * 360.0
    el_deg = 90.0 - (np.asarray(v) / H) * 180.0
    return az_deg, el_deg


def direction_to_neu(az_deg, el_deg):
    """
    Convert azimuth / elevation angles to a unit vector in NEU frame.

    Returns (east, north, up) numpy array.
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    east = math.sin(az) * math.cos(el)
    north = math.cos(az) * math.cos(el)
    up = math.sin(el)
    return np.array([east, north, up])


def neu_to_direction(neu_vec):
    """
    Convert a NEU unit vector to (azimuth_deg, elevation_deg).
    """
    e, n, u = neu_vec
    az_rad = math.atan2(e, n)         # clockwise from North
    el_rad = math.asin(np.clip(u / np.linalg.norm(neu_vec), -1, 1))
    return math.degrees(az_rad) % 360.0, math.degrees(el_rad)


# ---------------------------------------------------------------------------
# LiDAR point → panorama pixel
# ---------------------------------------------------------------------------

def lidar_point_to_pixel(xyz_rd, pose, img_shape):
    """
    Project a 3D point in RD New onto an equirectangular panorama.

    Parameters
    ----------
    xyz_rd : array-like, shape (3,)
        Point coordinates (x, y, z) in EPSG:28992.
    pose : CameraPose
    img_shape : (H, W, ...) tuple

    Returns
    -------
    (u, v) : float pixel coords, or None if the point is behind the camera
              (elevation would be irrelevant, but we return None if distance=0)
    """
    dx = xyz_rd[0] - pose.x_rd   # East offset
    dy = xyz_rd[1] - pose.y_rd   # North offset
    dz = xyz_rd[2] - pose.z      # Up offset
    dist_h = math.hypot(dx, dy)
    if dist_h < 1e-6 and abs(dz) < 1e-6:
        return None

    az_deg = math.degrees(math.atan2(dx, dy)) % 360.0   # clockwise from North
    el_deg = math.degrees(math.atan2(dz, dist_h))

    u, v = direction_to_pixel(az_deg, el_deg, img_shape)
    return float(u), float(v)


def lidar_points_to_pixels(xyz_rd_arr, pose, img_shape):
    """
    Vectorised projection of multiple 3D points.

    Parameters
    ----------
    xyz_rd_arr : np.ndarray, shape (N, 3)
    pose : CameraPose
    img_shape : (H, W) tuple

    Returns
    -------
    uv : np.ndarray, shape (N, 2)  — pixel (u, v) for each point
    """
    dx = xyz_rd_arr[:, 0] - pose.x_rd
    dy = xyz_rd_arr[:, 1] - pose.y_rd
    dz = xyz_rd_arr[:, 2] - pose.z
    dist_h = np.hypot(dx, dy)

    az_rad = np.arctan2(dx, dy)          # East=sin, North=cos → clockwise from N
    az_deg = np.degrees(az_rad) % 360.0
    el_deg = np.degrees(np.arctan2(dz, np.maximum(dist_h, 1e-9)))

    H, W = img_shape[:2]
    u = az_deg / 360.0 * W
    v = (90.0 - el_deg) / 180.0 * H
    return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------------
# Equirectangular → perspective patch
# ---------------------------------------------------------------------------

@dataclass
class PatchParams:
    """Parameters describing a perspective patch extracted from an equirectangular image."""
    yaw_deg: float     # centre azimuth (from North, clockwise)
    pitch_deg: float   # centre elevation (degrees above horizon)
    fov_h_deg: float   # horizontal field of view
    out_h: int
    out_w: int
    img_shape: tuple   # (H, W) of the source equirectangular image


def _rotation_matrix(yaw_deg, pitch_deg):
    """
    Build 3×3 rotation matrix from patch frame (x=right, y=up, z=forward)
    to global NEU frame (East, North, Up).

    For a camera pointing at azimuth=yaw_deg, elevation=pitch_deg:
        forward = (sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch))
        right   = (cos(yaw), -sin(yaw), 0)
        up      = cross(right, forward)

    The returned matrix has columns [right, up, forward].
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    sy, cy = math.sin(y), math.cos(y)
    sp, cp = math.sin(p), math.cos(p)

    fwd = np.array([sy * cp, cy * cp, sp])
    right = np.array([cy, -sy, 0.0])
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)

    return np.column_stack([right, up, fwd])   # shape (3, 3)


def equirect_to_perspective(img, patch_params):
    """
    Extract a perspective patch from an equirectangular panorama.

    Uses rectilinear (gnomonic) projection. No OpenCV required — implemented
    in numpy with bilinear interpolation.

    Parameters
    ----------
    img : np.ndarray, shape (H, W, 3)  uint8
    patch_params : PatchParams

    Returns
    -------
    patch : np.ndarray, shape (out_h, out_w, 3)  uint8
    """
    pp = patch_params
    H_src, W_src = img.shape[:2]
    out_h, out_w = pp.out_h, pp.out_w

    fov_h = math.radians(pp.fov_h_deg)
    fov_v = fov_h * out_h / out_w

    R = _rotation_matrix(pp.yaw_deg, pp.pitch_deg)

    # Build grid of ray directions in patch frame
    xs = np.linspace(-math.tan(fov_h / 2), math.tan(fov_h / 2), out_w)
    ys = np.linspace(math.tan(fov_v / 2), -math.tan(fov_v / 2), out_h)
    xg, yg = np.meshgrid(xs, ys, indexing="xy")
    zg = np.ones_like(xg)

    # Stack and normalise
    dirs = np.stack([xg, yg, zg], axis=-1)          # (out_h, out_w, 3)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs /= norms

    # Rotate to global NEU: (E, N, U)
    dirs_neu = dirs.reshape(-1, 3) @ R.T             # (out_h*out_w, 3)
    e, n, u = dirs_neu[:, 0], dirs_neu[:, 1], dirs_neu[:, 2]

    # Convert to equirectangular pixel coords
    az_rad = np.arctan2(e, n)                        # clockwise from North
    az_deg = np.degrees(az_rad) % 360.0
    el_deg = np.degrees(np.arctan2(u, np.hypot(e, n)))

    u_src = (az_deg / 360.0 * W_src).reshape(out_h, out_w)
    v_src = ((90.0 - el_deg) / 180.0 * H_src).reshape(out_h, out_w)

    # Bilinear interpolation with horizontal wrap-around
    u0 = np.floor(u_src).astype(np.int32) % W_src
    u1 = (u0 + 1) % W_src
    v0 = np.clip(np.floor(v_src).astype(np.int32), 0, H_src - 1)
    v1 = np.clip(v0 + 1, 0, H_src - 1)

    wu = (u_src - np.floor(u_src))[..., np.newaxis]
    wv = (v_src - np.floor(v_src))[..., np.newaxis]

    patch = (
        (1 - wu) * (1 - wv) * img[v0, u0] +
        wu       * (1 - wv) * img[v0, u1] +
        (1 - wu) * wv       * img[v1, u0] +
        wu       * wv       * img[v1, u1]
    ).astype(np.uint8)

    return patch


def make_patches(heading_deg, fov_h_deg=90.0, out_hw=(640, 640),
                 n_horizontal=4, pitch_deg=0.0, img_shape=None):
    """
    Create a list of PatchParams covering the panorama around the heading direction.

    Parameters
    ----------
    heading_deg : float
        Vehicle heading (centre of first patch).
    fov_h_deg : float
        Horizontal FOV per patch in degrees.
    out_hw : (int, int)
        Output patch size (height, width).
    n_horizontal : int
        Number of evenly-spaced horizontal patches (default: 4 = 360°/90°).
    pitch_deg : float
        Vertical centre of all patches (default 0° = horizon).
    img_shape : tuple or None
        Source image shape; stored in PatchParams for back-projection.

    Returns
    -------
    list of PatchParams
    """
    step = 360.0 / n_horizontal
    patches = []
    for i in range(n_horizontal):
        yaw = (heading_deg + i * step) % 360.0
        patches.append(PatchParams(
            yaw_deg=yaw,
            pitch_deg=pitch_deg,
            fov_h_deg=fov_h_deg,
            out_h=out_hw[0],
            out_w=out_hw[1],
            img_shape=img_shape or (2000, 4000),
        ))
    return patches


# ---------------------------------------------------------------------------
# Patch bbox → equirectangular pixel
# ---------------------------------------------------------------------------

def patch_bbox_to_equirect(bbox_xyxy, patch_params):
    """
    Convert a detection bounding box (in patch pixel coords) to the
    (u, v) centre pixel in the source equirectangular image.

    Parameters
    ----------
    bbox_xyxy : (x1, y1, x2, y2) in patch pixels
    patch_params : PatchParams

    Returns
    -------
    (u_src, v_src) : float pixel in the equirectangular image
    """
    pp = patch_params
    cx_patch = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
    cy_patch = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0

    fov_h = math.radians(pp.fov_h_deg)
    fov_v = fov_h * pp.out_h / pp.out_w

    # Direction in patch frame
    dx = math.tan(fov_h / 2) * (2 * cx_patch / pp.out_w - 1)
    dy = math.tan(fov_v / 2) * (1 - 2 * cy_patch / pp.out_h)
    dz = 1.0
    norm = math.sqrt(dx**2 + dy**2 + dz**2)
    d_patch = np.array([dx / norm, dy / norm, dz / norm])

    R = _rotation_matrix(pp.yaw_deg, pp.pitch_deg)
    d_neu = R @ d_patch              # (E, N, U)

    az_deg = math.degrees(math.atan2(d_neu[0], d_neu[1])) % 360.0
    el_deg = math.degrees(math.atan2(d_neu[2], math.hypot(d_neu[0], d_neu[1])))

    H_src, W_src = pp.img_shape[:2]
    u_src = az_deg / 360.0 * W_src
    v_src = (90.0 - el_deg) / 180.0 * H_src
    return u_src, v_src


# ---------------------------------------------------------------------------
# Ray → ground intersection
# ---------------------------------------------------------------------------

def pixel_to_ray_neu(u, v, img_shape):
    """
    Convert an equirectangular pixel to a unit ray in NEU frame.
    """
    az_deg, el_deg = pixel_to_direction(u, v, img_shape)
    return direction_to_neu(az_deg, el_deg)


def ray_ground_intersect(pose, ray_neu, ground_z):
    """
    Intersect a ray from the camera with a horizontal ground plane at *ground_z*.

    Parameters
    ----------
    pose : CameraPose
    ray_neu : np.ndarray, shape (3,)  unit vector (East, North, Up)
    ground_z : float
        Ground elevation in metres (same reference as pose.z).

    Returns
    -------
    (x_rd, y_rd) : ground hit point in EPSG:28992, or None if ray is upward.
    """
    dz = ray_neu[2]
    if dz >= 0:
        return None  # ray goes up or is horizontal — no ground hit

    t = (ground_z - pose.z) / dz
    x_hit = pose.x_rd + t * ray_neu[0]
    y_hit = pose.y_rd + t * ray_neu[1]
    return x_hit, y_hit


def detection_to_ground_position(bbox_xyxy, patch_params, pose, ground_z):
    """
    Map a detection bounding box centre to a ground position in RD New.

    Parameters
    ----------
    bbox_xyxy : (x1, y1, x2, y2)
    patch_params : PatchParams
    pose : CameraPose
    ground_z : float  — ground elevation at the camera tile (metres)

    Returns
    -------
    (x_rd, y_rd) or None
    """
    u_src, v_src = patch_bbox_to_equirect(bbox_xyxy, patch_params)
    ray = pixel_to_ray_neu(u_src, v_src, patch_params.img_shape)
    return ray_ground_intersect(pose, ray, ground_z)
