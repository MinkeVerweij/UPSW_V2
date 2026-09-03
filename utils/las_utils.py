import re
import laspy
import numpy as np
import pdal

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from pathlib import Path
import json

def get_tilecode_from_filename(filename):
    """Extract the tile code from a file name."""
    return re.match(r'.*(\d{6}_\d{6}).*', filename)[1]

def read_las(las_file):
    """Read a las file and return the las object."""
    return laspy.read(las_file)

def read_las_with_labels(las_path):
    """
    Read LAS file and extract point coordinates and labels.

    Parameters:
    - las_path (str): Path to the LAS file.

    Returns:
    Tuple of NumPy arrays (points, labels).
    """
    pointcloud = laspy.read(las_path)

    if 'label' not in pointcloud.point_format.extra_dimension_names:
        labels = np.zeros((len(pointcloud.x),), dtype='uint16')
    else:
        labels = pointcloud.label
    
    x = (np.array(pointcloud.x))
    y = (np.array(pointcloud.y))
    z = (np.array(pointcloud.z))
    points = np.vstack((x, y, z)).T

    return points, labels

def label_and_save_las(las, labels, outfile):
    """Label a las file using the provided class labels and save to outfile."""
    assert len(labels) == las.header.point_count
    if 'label' not in las.point_format.extra_dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="label", type="uint8",
                          description="Labels"))
    las.label = labels
    las.write(outfile)

def crop_laz_with_polygon(input_file: Path, output_file: Path, polygon: Polygon):
    """
    Crop LAZ file by Shapely polygon using PDAL.
    Preserves attributes and header metadata.
    """

    if polygon.is_empty:
        raise ValueError("Polygon is empty.")

    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    pipeline_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": str(input_file)
            },
            {
                "type": "filters.crop",
                "polygon": polygon.wkt
            },
            {
                "type": "writers.las",
                "filename": str(output_file),
                "compression": "laszip"
            }
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    count = pipeline.execute()

    if count == 0:
        print(f"Warning: No points written to {output_file}")

    return count

def voxel_downsample_las(in_path: Path, out_path: Path, voxel_size: float = 0.15,
                         chunk_size: int = 5_000_000) -> int:
    """
    Voxel-downsample a LAS/LAZ file, keeping one real point per occupied
    3-D voxel (all point attributes — RGB, gps_time, etc. — are preserved
    on whichever point is kept, nothing is synthesized).

    Streams through the input in chunks and writes matches immediately, so
    peak memory stays bounded by `chunk_size` plus the (much smaller) number
    of voxels kept so far — the full point cloud is never held in memory at
    once. Needed for dense mobile-mapping tiles (10^8+ points) that don't
    fit in memory at full resolution.

    Parameters
    ----------
    in_path : Path
        Input LAS/LAZ file.
    out_path : Path
        Output LAS/LAZ file (downsampled).
    voxel_size : float
        Voxel edge length in metres.
    chunk_size : int
        Points read per chunk.

    Returns
    -------
    int
        Number of points written.
    """
    with laspy.open(in_path) as reader:
        header = reader.header
        mins, maxs = header.mins, header.maxs

        x_off = int(np.floor(mins[0] / voxel_size))
        y_off = int(np.floor(mins[1] / voxel_size))
        z_off = int(np.floor(mins[2] / voxel_size))
        y_span = int(np.ceil((maxs[1] - mins[1]) / voxel_size)) + 2
        z_span = int(np.ceil((maxs[2] - mins[2]) / voxel_size)) + 2

        seen = set()
        n_written = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with laspy.open(out_path, mode='w', header=header) as writer:
            for chunk in reader.chunk_iterator(chunk_size):
                x_idx = np.floor(np.asarray(chunk.x) / voxel_size).astype(np.int64) - x_off
                y_idx = np.floor(np.asarray(chunk.y) / voxel_size).astype(np.int64) - y_off
                z_idx = np.floor(np.asarray(chunk.z) / voxel_size).astype(np.int64) - z_off
                keys = (x_idx * y_span + y_idx) * z_span + z_idx

                # First occurrence of each key within this chunk (vectorized),
                # then drop any already written by an earlier chunk.
                uniq_keys, first_idx = np.unique(keys, return_index=True)
                new_mask = np.array([k not in seen for k in uniq_keys])
                seen.update(uniq_keys[new_mask].tolist())
                keep_idx = first_idx[new_mask]

                if len(keep_idx) > 0:
                    keep_idx.sort()
                    writer.write_points(chunk[keep_idx])
                    n_written += len(keep_idx)

    return n_written


def build_convex_hull_polygon(las_path: Path) -> Polygon:
    """
    Reads a LAS/LAZ file and returns a Shapely Polygon
    representing the convex hull of its XY coordinates.
    """
    if not las_path.exists():
        raise FileNotFoundError(f"File not found: {las_path}")

    # Read point cloud
    with laspy.open(las_path) as reader:
        las = reader.read()

    # Extract XY only (avoid unnecessary Z in memory)
    coords = np.column_stack((las.x, las.y))

    if coords.shape[0] < 3:
        raise ValueError(f"Not enough points to compute convex hull: {las_path}")

    # Qhull overflows its internal indexing on very large point counts (seen
    # around ~10^8 points in dense mobile-mapping tiles). The hull only
    # depends on the outer boundary points, so a large random subsample gives
    # the same footprint without hitting that limit.
    MAX_HULL_POINTS = 2_000_000
    if coords.shape[0] > MAX_HULL_POINTS:
        rng = np.random.default_rng(0)
        idx = rng.choice(coords.shape[0], size=MAX_HULL_POINTS, replace=False)
        coords = coords[idx]

    # Compute convex hull
    hull = ConvexHull(coords)

    # Build polygon from hull vertices
    polygon = Polygon(coords[hull.vertices])

    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # Fix minor geometry issues

    return polygon
