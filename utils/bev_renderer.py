"""
Bird's-eye-view (BEV) renderer for obstacle clusters.

Converts a set of 3-D points (with pre-computed heights-above-ground) into a
small 2-channel raster suitable for visual inspection and CNN classification:

  Channel 0 — max height-above-ground per pixel, normalised to [0, 1]
  Channel 1 — point density per pixel, normalised to [0, 1]

Usage
-----
    from utils.bev_renderer import render_bev, render_cluster_grid

    # single cluster
    img = render_bev(pts_xy, heights, resolution=0.05, output_size=64)

    # batch render and display
    fig = render_cluster_grid(cluster_list, heights_list, titles=None)
"""

import numpy as np


def render_bev(
    pts_xy: np.ndarray,
    heights: np.ndarray,
    resolution: float = 0.05,
    output_size: int = 64,
    height_clip: float = 2.5,
) -> np.ndarray:
    """
    Render a single cluster as a 2-channel BEV image.

    Parameters
    ----------
    pts_xy : (N, 2) float — XY coordinates of cluster points
    heights : (N,) float — height-above-ground for each point
    resolution : float — metres per pixel used for the native raster before
        resizing; 0.05 m gives a ~3 m² field of view per 64×64 pixel tile
    output_size : int — final square image size in pixels after resize
    height_clip : float — heights above this value are clipped (removes crane
        arms, overhead wires that leak into a cluster bbox)

    Returns
    -------
    img : (output_size, output_size, 2) float32, values in [0, 1]
        img[..., 0] = normalised height map
        img[..., 1] = normalised density map
    """
    if len(pts_xy) == 0:
        return np.zeros((output_size, output_size, 2), dtype=np.float32)

    h = np.clip(heights, 0.0, height_clip).astype(np.float32)

    # Centre the cluster
    cx, cy = pts_xy[:, 0].mean(), pts_xy[:, 1].mean()
    half = output_size * resolution / 2.0
    x0, y0 = cx - half, cy - half

    px = np.floor((pts_xy[:, 0] - x0) / resolution).astype(np.int32)
    py = np.floor((pts_xy[:, 1] - y0) / resolution).astype(np.int32)

    in_bounds = (px >= 0) & (px < output_size) & (py >= 0) & (py < output_size)
    px, py, h = px[in_bounds], py[in_bounds], h[in_bounds]

    height_map  = np.zeros((output_size, output_size), dtype=np.float32)
    density_map = np.zeros((output_size, output_size), dtype=np.float32)

    np.maximum.at(height_map,  (px, py), h)
    np.add.at(density_map, (px, py), 1.0)

    # Normalise
    if height_map.max() > 0:
        height_map /= height_clip
    if density_map.max() > 0:
        density_map /= density_map.max()

    return np.stack([height_map, density_map], axis=-1)


def render_cluster_grid(
    clusters_xy: list,
    clusters_heights: list,
    titles: list = None,
    resolution: float = 0.05,
    output_size: int = 64,
    ncols: int = 8,
    figsize_per_cell: float = 1.2,
):
    """
    Render a grid of BEV thumbnails for visual inspection / labeling.

    Returns a matplotlib Figure. Import matplotlib inside to keep the module
    importable in headless environments.

    Parameters
    ----------
    clusters_xy : list of (N_i, 2) arrays
    clusters_heights : list of (N_i,) arrays
    titles : optional list of strings (e.g. cluster index or current label)
    ncols : columns in the grid
    figsize_per_cell : inches per thumbnail
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n = len(clusters_xy)
    if n == 0:
        return plt.figure()

    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_cell, nrows * figsize_per_cell),
        squeeze=False,
    )

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        img = render_bev(clusters_xy[i], clusters_heights[i],
                         resolution=resolution, output_size=output_size)
        # Show height in green, density in blue
        rgb = np.zeros((output_size, output_size, 3), dtype=np.float32)
        rgb[..., 1] = img[..., 0]  # green = height
        rgb[..., 2] = img[..., 1]  # blue  = density
        ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        label = titles[i] if titles is not None else str(i)
        ax.set_title(label, fontsize=7, pad=2)

    fig.tight_layout(pad=0.3)
    return fig


def extract_cluster_bev_from_arrays(
    xyz: np.ndarray,
    heights: np.ndarray,
    polygon,
    resolution: float = 0.05,
    output_size: int = 64,
    height_clip: float = 2.5,
) -> np.ndarray:
    """
    Extract points inside a Shapely polygon and render a BEV image.

    Parameters
    ----------
    xyz : (N, 3) float — full tile point cloud
    heights : (N,) float — pre-computed heights above ground for all points
    polygon : shapely.Polygon — cluster footprint
    """
    from shapely.geometry import Point

    minx, miny, maxx, maxy = polygon.bounds
    bbox = (
        (xyz[:, 0] >= minx) & (xyz[:, 0] <= maxx) &
        (xyz[:, 1] >= miny) & (xyz[:, 1] <= maxy)
    )
    if not bbox.any():
        return np.zeros((output_size, output_size, 2), dtype=np.float32)

    pts_box = xyz[bbox]
    h_box   = heights[bbox]

    inside = np.array([polygon.contains(Point(p[0], p[1])) for p in pts_box])
    if not inside.any():
        return np.zeros((output_size, output_size, 2), dtype=np.float32)

    return render_bev(pts_box[inside, :2], h_box[inside],
                      resolution=resolution, output_size=output_size,
                      height_clip=height_clip)
