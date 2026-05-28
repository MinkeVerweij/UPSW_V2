import statistics

#from upcp.utils import clip_utils, 
from shapely.geometry import Point, MultiPoint, LineString
from shapely.ops import split, snap
import numpy as np
from . import labels as Labels
from . import clipping_tools 


def create_mask(points, labels, polygons):
    """
    Create a mask for points based on labels and polygons.

    Parameters:
    - points (numpy.ndarray): Array of points.
    - labels (numpy.ndarray): Array of labels for each point.
    - polygons (list): List of polygons used for masking.

    Returns:
    numpy.ndarray: Boolean mask for points.
    """
    label_mask = np.zeros(len(points), dtype=bool)

    # Already labelled ground points can be labelled as road.
    mask = (labels == Labels.Labels.GROUND) | (labels == Labels.Labels.ROAD)
    mask_ids = np.where(mask)[0]

    road_mask = np.zeros((len(mask_ids),), dtype=bool)
    for polygon in polygons:
        clip_mask = clipping_tools.poly_clip(points[mask, :], polygon)
        road_mask = road_mask | clip_mask

    label_mask[mask_ids[road_mask]] = True

    return label_mask


def calculate_curb_height(points, labels, segment_polygon, min_nr_points):
    """
    Calculate curb height from points and labels within a segment polygon.

    Parameters:
    - points (numpy.ndarray): Array of points.
    - labels (numpy.ndarray): Array of labels for each point.
    - segment_polygon: Polygon defining the segment.
    - min_nr_points (int): Minimum number of points needed for calculation.

    Returns:
    tuple: Curb height and a boolean indicating if there are enough points for calculation.
    """
    curb_height = np.nan
    available_points = True

    label_mask = create_mask(points, labels, [segment_polygon, ])
    points_in_segment = points[label_mask]
    labels_in_segment = labels[label_mask]

    z_values_road = points_in_segment[labels_in_segment == Labels.Labels.ROAD][:, -1]

    if len(z_values_road) > min_nr_points:
        z_values_road.sort()
        road_height = statistics.median(z_values_road)
    else:
        available_points = False

    z_values_sidewalk = points_in_segment[labels_in_segment == Labels.Labels.GROUND][:, -1]

    if len(z_values_sidewalk) > min_nr_points:
        z_values_sidewalk.sort()
        sidewalk_height = statistics.median(z_values_sidewalk)
    else:
        available_points = False

    if available_points:
        curb_height = sidewalk_height - road_height

    return curb_height, available_points


def get_height_color(curb_height, available_points, min_h):
    """
    Get color based on curb height and availability of points.

    Parameters:
    - curb_height (float): Curb height.
    - available_points (bool): Whether there are enough points for calculation.
    - min_h (float): Minimum height for color differentiation.

    Returns:
    str: Color code.
    """
    if not available_points:
        color = 'black'
        return color

    if curb_height < min_h:
        color = 'green'
    else:
        color = 'orange'
    return color


def get_points_on_line(line: LineString, distance_delta: float):
    """
    Sample points along a LineString at intervals of `distance_delta`.
    Returns a list of Shapely Point objects.
    """
    if line.length == 0:
        return []

    distances = np.arange(0, line.length, distance_delta)
    points = [line.interpolate(d) for d in distances]

    # Ensure start and end points are included
    points = [Point(line.coords[0])] + points + [Point(line.coords[-1])]
    return points


def split_line_by_points(line: LineString, points: list, min_segment_length: float = 1e-6):
    """
    Splits a LineString by a list of points.
    
    Parameters:
    - line: LineString to split.
    - points: list of shapely POINTs on the line.
    - min_segment_length: minimum length of segment to keep.
    
    Returns:
    - List of LineString segments.
    """
    # Remove duplicate points
    unique_points = list({(p.x, p.y): p for p in points}.values())
    
    if not unique_points:
        return [line]  # No points to split
    
    # Convert to MultiPoint
    splitter = MultiPoint(unique_points)
    
    # Split the line
    split_result = split(line, splitter)
    
    # Filter only LineString objects and remove tiny segments
    segments = [geom for geom in split_result.geoms 
                if isinstance(geom, LineString) and geom.length > min_segment_length]
    
    return segments