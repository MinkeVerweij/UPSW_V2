"""
Self-contained PointNet++ Single-Scale Grouping (SSG) classifier.

No compiled CUDA extensions required — uses torch.cdist for ball queries,
which runs on both CPU and CUDA. Optionally swaps in torch_cluster.radius
for faster GPU ball queries if the package is available.

Physical scale is preserved: ball query radii are in metres and rely on
the fact that cluster point clouds are NOT normalised to a unit sphere.

Input:  (B, N, C) where C = n_features (default 7: x_c, y_c, z, r, g, b, h_ag)
Output: (B, n_classes) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torch_cluster import radius as tc_radius
    _HAS_TC = True
except ImportError:
    _HAS_TC = False


# ── Ball query ─────────────────────────────────────────────────────────────────

def ball_query(xyz, centroids, radius, max_samples):
    """
    For each centroid find up to max_samples points within radius.

    Parameters
    ----------
    xyz       : (B, N, 3) float
    centroids : (B, M, 3) float
    radius    : float
    max_samples : int

    Returns
    -------
    idx : (B, M, max_samples) long  — indices into xyz; padded with 0
    """
    B, N, _ = xyz.shape
    M = centroids.shape[1]

    # (B, M, N) pairwise squared distances
    dists = torch.cdist(centroids, xyz)            # (B, M, N)
    mask  = dists <= radius                         # (B, M, N) bool

    # Take up to max_samples per centroid (closest first)
    dists_masked = dists.clone()
    dists_masked[~mask] = 1e9
    sorted_idx = dists_masked.argsort(dim=2)        # (B, M, N)
    idx = sorted_idx[:, :, :max_samples]            # (B, M, max_samples)

    # Replace out-of-radius slots with the first valid index (repeat)
    valid = mask.gather(2, idx)                     # (B, M, max_samples)
    first_valid = idx[:, :, :1].expand_as(idx)
    idx = torch.where(valid, idx, first_valid)

    return idx


# ── Farthest point sampling ─────────────────────────────────────────────────────

def fps(xyz, n_samples):
    """
    (B, N, 3) → (B, n_samples) indices of farthest-point-sampled centroids.
    """
    B, N, _ = xyz.shape
    device   = xyz.device
    selected = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    dists    = torch.full((B, N), float('inf'), device=device)
    current  = torch.randint(0, N, (B,), device=device)

    for i in range(n_samples):
        selected[:, i] = current
        cur_xyz = xyz[torch.arange(B, device=device), current, :]  # (B, 3)
        diff    = xyz - cur_xyz.unsqueeze(1)                        # (B, N, 3)
        d       = (diff ** 2).sum(dim=2)                            # (B, N)
        dists   = torch.minimum(dists, d)
        current = dists.argmax(dim=1)                               # (B,)

    return selected


def index_points(pts, idx):
    """
    pts : (B, N, C)
    idx : (B, M) or (B, M, K)
    Returns : (B, M, C) or (B, M, K, C)
    """
    B    = pts.shape[0]
    device = pts.device
    if idx.dim() == 2:
        M = idx.shape[1]
        idx_exp = idx.unsqueeze(-1).expand(B, M, pts.shape[-1])
        return pts.gather(1, idx_exp)
    elif idx.dim() == 3:
        M, K = idx.shape[1], idx.shape[2]
        idx_exp = idx.unsqueeze(-1).expand(B, M, K, pts.shape[-1])
        return pts.unsqueeze(2).expand(B, pts.shape[1], K, pts.shape[-1]).gather(1, idx_exp)


# ── Set Abstraction layer ───────────────────────────────────────────────────────

class PointNetSetAbstraction(nn.Module):
    """
    One SA layer: FPS → ball query → PointNet mini-MLP per neighbourhood.

    Parameters
    ----------
    npoint   : int or None
        Number of centroids to sample.  None = global pooling (no FPS/ball).
    radius   : float
        Ball query radius in metres.  Ignored if npoint is None.
    nsample  : int
        Max points per neighbourhood.
    in_ch    : int
        Number of input feature channels (excluding XYZ).
    mlp      : list[int]
        Output channels of each MLP layer.
    """

    def __init__(self, npoint, radius, nsample, in_ch, mlp):
        super().__init__()
        self.npoint  = npoint
        self.radius  = radius
        self.nsample = nsample

        layers = []
        c_in = in_ch + 3   # concatenate relative XYZ to features
        for c_out in mlp:
            layers += [nn.Linear(c_in, c_out), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True)]
            c_in = c_out
        self.mlp = nn.Sequential(*layers)
        self.out_ch = mlp[-1]

    def forward(self, xyz, features):
        """
        xyz      : (B, N, 3)
        features : (B, N, C) or None
        Returns  : new_xyz (B, M, 3), new_feat (B, M, out_ch)
        """
        B, N, _ = xyz.shape

        if self.npoint is None:
            # Global pooling
            if features is not None:
                x = torch.cat([xyz, features], dim=2)
            else:
                x = xyz
            x = x.view(B * N, -1)
            x = self.mlp(x).view(B, N, -1)
            x, _ = x.max(dim=1, keepdim=True)  # (B, 1, out_ch)
            return xyz.mean(dim=1, keepdim=True), x

        # FPS
        fps_idx   = fps(xyz, self.npoint)              # (B, M)
        new_xyz   = index_points(xyz, fps_idx)         # (B, M, 3)

        # Ball query
        ball_idx  = ball_query(xyz, new_xyz, self.radius, self.nsample)  # (B, M, K)
        grouped_xyz = index_points(xyz, ball_idx)      # (B, M, K, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)            # relative coords

        if features is not None:
            grouped_feat = index_points(features, ball_idx)  # (B, M, K, C)
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=3)  # (B, M, K, C+3)
        else:
            grouped = grouped_xyz                            # (B, M, K, 3)

        # MLP on each point in neighbourhood
        BM, K, C = grouped.shape[0] * grouped.shape[1], grouped.shape[2], grouped.shape[3]
        x = grouped.view(B * self.npoint * K, C)
        x = self.mlp(x)
        x = x.view(B, self.npoint, K, -1)
        x, _ = x.max(dim=2)                          # max pool over neighbourhood

        return new_xyz, x


# ── PointNet++ SSG Classifier ───────────────────────────────────────────────────

class PointNet2Classifier(nn.Module):
    """
    PointNet++ SSG for point cloud classification.

    Architecture
    ------------
    SA(npoint=128, r=0.2m, K=32,  mlp=[64, 64, 128])
    SA(npoint=32,  r=0.4m, K=64,  mlp=[128, 128, 256])
    SA(npoint=None [global],       mlp=[256, 512, 1024])
    FC(1024→512) + BN + ReLU + Dropout(0.4)
    FC(512→256)  + BN + ReLU + Dropout(0.4)
    FC(256→n_classes)

    Input  : (B, N, n_features)   — n_features default 7
    Output : (B, n_classes) logits

    Ball query radii are physically meaningful metres because cluster
    coordinates are NOT normalised to a unit sphere.
    """

    def __init__(self, n_classes, n_features=7):
        super().__init__()
        # n_features includes XYZ (first 3) + extra features (last n_features - 3)
        extra = n_features - 3

        self.sa1 = PointNetSetAbstraction(
            npoint=128, radius=0.2, nsample=32,
            in_ch=extra, mlp=[64, 64, 128]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=32, radius=0.4, nsample=64,
            in_ch=128, mlp=[128, 128, 256]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_ch=256, mlp=[256, 512, 1024]
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, pts):
        """
        pts : (B, N, n_features)
        """
        xyz      = pts[:, :, :3]                 # (B, N, 3)
        features = pts[:, :, 3:] if pts.shape[2] > 3 else None  # (B, N, C)

        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        _, features   = self.sa3(xyz, features)

        x = features.squeeze(1)                  # (B, 1024)
        x = self.dp1(F.relu(self.bn1(self.fc1(x))))
        x = self.dp2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


# ── Convenience ────────────────────────────────────────────────────────────────

def build_model(n_classes, n_features=7, device='cpu'):
    model = PointNet2Classifier(n_classes=n_classes, n_features=n_features)
    return model.to(device)
