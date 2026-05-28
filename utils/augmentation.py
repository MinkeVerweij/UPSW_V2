"""
3-D augmentations for point cloud training data.

All transforms operate on an (N, 7) float32 array with columns
[x_c, y_c, z, r, g, b, h_ag] where:
  - x_c, y_c : XY coordinates centered on cluster centroid
  - z         : absolute Z (not centered)
  - r, g, b   : RGB in [0, 1]
  - h_ag      : height above AHN ground (metres)

Physical scale is intentionally preserved (no unit-sphere normalisation)
so PointNet++ ball query radii remain metrically meaningful.
"""

import numpy as np


def random_rotate_z(pts7, angle_range=(-np.pi, np.pi)):
    """Rotate XY around Z-axis. Z and h_ag are unchanged."""
    angle = np.random.uniform(*angle_range)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xy  = pts7[:, :2].copy()
    pts7 = pts7.copy()
    pts7[:, 0] =  cos_a * xy[:, 0] - sin_a * xy[:, 1]
    pts7[:, 1] =  sin_a * xy[:, 0] + cos_a * xy[:, 1]
    return pts7


def random_flip_x(pts7):
    """Mirror X axis (horizontal symmetry). Leaves Y, Z, h_ag unchanged."""
    pts7 = pts7.copy()
    pts7[:, 0] = -pts7[:, 0]
    return pts7


def random_jitter(pts7, sigma=0.01, clip=0.02):
    """Add Gaussian noise to XYZ only."""
    pts7 = pts7.copy()
    noise = np.clip(np.random.normal(0, sigma, (len(pts7), 3)), -clip, clip)
    pts7[:, :3] += noise.astype(pts7.dtype)
    return pts7


def random_scale(pts7, scale_range=(0.9, 1.1)):
    """Scale XYZ and h_ag uniformly. RGB is not scaled."""
    s = np.random.uniform(*scale_range)
    pts7 = pts7.copy()
    pts7[:, :3] *= s
    pts7[:, 6]  *= s
    return pts7


def random_drop_points(pts7, max_drop_frac=0.10):
    """Randomly drop up to max_drop_frac fraction of points."""
    n_drop = np.random.randint(0, max(1, int(len(pts7) * max_drop_frac)) + 1)
    if n_drop == 0 or n_drop >= len(pts7):
        return pts7.copy()
    keep = np.random.choice(len(pts7), len(pts7) - n_drop, replace=False)
    return pts7[keep]


def random_color_jitter(pts7, brightness=0.10, contrast=0.10):
    """Jitter RGB channels only. Clamps result to [0, 1]."""
    pts7 = pts7.copy()
    b = np.random.uniform(1 - brightness, 1 + brightness)
    c = np.random.uniform(1 - contrast,   1 + contrast)
    pts7[:, 3:6] = np.clip(pts7[:, 3:6] * c + (b - 1), 0.0, 1.0)
    return pts7.astype(np.float32)


def compose(transforms):
    """Return a callable that applies a list of transforms in order."""
    def fn(pts7):
        for t in transforms:
            pts7 = t(pts7)
        return pts7
    return fn


def default_train_augmentation():
    """Standard augmentation pipeline for Amsterdam street LiDAR."""
    return compose([
        random_rotate_z,
        random_flip_x,
        random_jitter,
        random_scale,
        random_drop_points,
        random_color_jitter,
    ])
