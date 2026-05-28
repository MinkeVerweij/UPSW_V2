import numpy as np
import laspy 
from scipy.spatial import cKDTree
from scipy.ndimage import label

import geopandas as gpd
from shapely.geometry import Polygon

from scipy.spatial import ConvexHull

import numpy as np
import alphashape
import laspy

GROUND_LABEL = 9

HEIGHT_THRESHOLD = 0.25
VOXEL_SIZE = 0.25
MIN_COMPONENT_SIZE = 30


def cluster_obstacles(laz_file):

    print("Loading LAZ...")
    pc = laspy.read(laz_file)

    x = pc.x
    y = pc.y
    z = pc.z
    labels = pc.label   # your custom label field

    xyz = np.vstack((x,y,z)).T

    print("Total points:", len(xyz))

    # -----------------------------
    # 1. Build ground grid
    # -----------------------------
    print("Building ground grid...")

    ground = xyz[labels == GROUND_LABEL]

    if len(ground) == 0:
        print(f"Warning: no ground points found in {laz_file}, skipping.")
        return []

    xmin = ground[:,0].min()
    ymin = ground[:,1].min()

    GRID = 0.5

    gx = ((ground[:,0] - xmin)/GRID).astype(int)
    gy = ((ground[:,1] - ymin)/GRID).astype(int)

    ground_grid = {}

    for xg,yg,zg in zip(gx,gy,ground[:,2]):
        key = (xg,yg)
        ground_grid[key] = min(zg, ground_grid.get(key,zg))

    # -----------------------------
    # 2. Height above ground
    # -----------------------------
    print("Computing height above ground...")

    px = ((xyz[:,0]-xmin)/GRID).astype(int)
    py = ((xyz[:,1]-ymin)/GRID).astype(int)

    heights = np.zeros(len(xyz))

    for i,(ix,iy,pz) in enumerate(zip(px,py,xyz[:,2])):
        gz = ground_grid.get((ix,iy),pz)
        heights[i] = pz-gz

    obstacle_pts = xyz[heights > HEIGHT_THRESHOLD]

    print("Obstacle candidates:", len(obstacle_pts))

    # -----------------------------
    # 3. Voxelization
    # -----------------------------
    print("Voxelizing...")

    v = np.floor(obstacle_pts / VOXEL_SIZE).astype(int)

    vmin = v.min(axis=0)
    v -= vmin

    grid_size = v.max(axis=0) + 1

    voxel_grid = np.zeros(grid_size, dtype=np.uint8)

    voxel_grid[v[:,0], v[:,1], v[:,2]] = 1

    # -----------------------------
    # 4. Connected components
    # -----------------------------
    print("Labeling components...")

    structure = np.ones((3,3,3), dtype=np.int8)

    labeled, num = label(voxel_grid, structure)

    print("Components found:", num)

    # -----------------------------
    # 5. Extract clusters
    # -----------------------------
    clusters = []

    for cid in range(1, num+1):

        voxels = np.argwhere(labeled == cid)

        if len(voxels) < MIN_COMPONENT_SIZE:
            continue

        pts = (voxels + vmin) * VOXEL_SIZE
        clusters.append(pts)

    return clusters

def clusters_to_concave_polygons(clusters, crs ,alpha=0.2):

    polys = []

    for cluster in clusters:

        xy = cluster[:, :2]

        if len(xy) < 10:
            continue

        poly = alphashape.alphashape(xy, alpha)

        if poly.is_valid:
            polys.append(poly)

    return gpd.GeoDataFrame(geometry=polys, crs=crs)