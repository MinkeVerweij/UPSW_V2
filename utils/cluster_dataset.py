"""
PyTorch Dataset for labeled obstacle clusters.

Loads cluster NPZ files referenced by an inventory DataFrame and returns
(pts_tensor, class_idx) pairs suitable for PointNet++ training.

Feature layout per point (7 channels):
  [x_c, y_c, z, r, g, b, h_ag]

Physical scale is preserved — XY is centered on cluster centroid but Z and
h_ag are absolute metres, matching the PointNet++ ball query radii.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def furthest_point_sample(pts, n):
    """Simple numpy FPS that sub-samples pts to n points."""
    N = len(pts)
    if n >= N:
        return np.arange(N)
    selected = np.zeros(n, dtype=np.int64)
    dists    = np.full(N, np.inf)
    current  = np.random.randint(N)
    for i in range(n):
        selected[i] = current
        diff = pts - pts[current]
        d    = (diff ** 2).sum(axis=1)
        dists = np.minimum(dists, d)
        current = int(np.argmax(dists))
    return selected


class ClusterDataset(Dataset):
    """
    Parameters
    ----------
    inventory_df : pd.DataFrame
        Must have columns ``npz_path`` and ``final_label``.
        Rows with ``final_label`` == NaN are excluded.
    n_points : int
        Fixed number of points per sample.  Clusters with fewer points are
        upsampled with replacement; larger clusters are downsampled via FPS.
    augment_fn : callable or None
        Applied to the (n_points, 7) array before returning.
    label_map : dict {int label_code -> int class_idx} or None
        If None, unique label codes are auto-mapped 0..K-1 sorted ascending.
    use_fps : bool
        Use farthest-point sampling for downsampling (slower but better
        coverage than random).  Default True.
    """

    def __init__(
        self,
        inventory_df,
        n_points=512,
        augment_fn=None,
        label_map=None,
        use_fps=True,
    ):
        df = inventory_df.copy()
        df = df[df['final_label'].notna()].reset_index(drop=True)
        df['final_label'] = df['final_label'].astype(int)

        if label_map is None:
            codes = sorted(df['final_label'].unique())
            label_map = {c: i for i, c in enumerate(codes)}

        # Drop rows whose label is not in the map
        df = df[df['final_label'].isin(label_map)].reset_index(drop=True)

        self.df         = df
        self.n_points   = n_points
        self.augment_fn = augment_fn
        self.label_map  = label_map
        self.use_fps    = use_fps

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        data  = np.load(row['npz_path'], allow_pickle=False)
        xyz_c = data['xyz_centered'].astype(np.float32)   # (N, 3)
        rgb   = data['rgb_norm'].astype(np.float32)        # (N, 3)
        h_ag  = data['height_ag'].astype(np.float32)       # (N,)

        # Build (N, 7) feature matrix
        pts7 = np.column_stack([xyz_c, rgb, h_ag]).astype(np.float32)
        N    = len(pts7)

        # Resample to fixed n_points
        n = self.n_points
        if N == n:
            idx = np.arange(N)
        elif N > n:
            idx = (furthest_point_sample(pts7[:, :3], n)
                   if self.use_fps
                   else np.random.choice(N, n, replace=False))
        else:
            # Upsample: repeat with replacement
            idx = np.concatenate([np.arange(N),
                                   np.random.choice(N, n - N, replace=True)])

        pts7 = pts7[idx]

        if self.augment_fn is not None:
            pts7 = self.augment_fn(pts7)

        class_idx = self.label_map[int(row['final_label'])]
        return (
            torch.from_numpy(pts7),          # (n_points, 7) float32
            torch.tensor(class_idx, dtype=torch.long),
        )

    def class_weights(self):
        """
        Returns a float tensor of inverse-frequency weights for
        torch.nn.CrossEntropyLoss(weight=...).  One weight per class,
        ordered by class_idx.
        """
        n_classes = len(self.label_map)
        counts    = np.zeros(n_classes, dtype=np.float32)
        for code, idx in self.label_map.items():
            counts[idx] = float((self.df['final_label'] == code).sum())
        counts = np.maximum(counts, 1.0)
        weights = 1.0 / counts
        weights = weights / weights.sum() * n_classes
        return torch.from_numpy(weights)

    @property
    def n_classes(self):
        return len(self.label_map)

    @classmethod
    def build_label_map(cls, inventory_df, min_samples=50, merge_into=99):
        """
        Build a label_map from inventory, including only classes with at
        least min_samples labeled examples.  Rare classes are mapped to
        merge_into (noise).

        Returns dict {label_code: class_idx}.
        """
        df = inventory_df[inventory_df['final_label'].notna()].copy()
        df['final_label'] = df['final_label'].astype(int)

        counts = df['final_label'].value_counts()
        valid  = [int(c) for c, n in counts.items() if n >= min_samples]

        # Ensure merge_into is included
        if merge_into not in valid:
            valid.append(merge_into)

        valid_sorted = sorted(valid)
        label_map    = {c: i for i, c in enumerate(valid_sorted)}

        # Remap rare codes to merge_into
        for code in df['final_label'].unique():
            if int(code) not in label_map:
                label_map[int(code)] = label_map[merge_into]

        return label_map
