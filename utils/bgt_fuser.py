import numpy as np
from shapely.geometry import Point
import logging
from . import labels as Labels
from . import clipping_tools

logger = logging.getLogger(__name__)

class BGTRoadFuser:
    """
    Road fuser using a BGT GeoDataFrame with columns:
    - 'bgt-functie' : road type string
    - 'geometry' : shapely polygon
    """

    def __init__(self, label, bgt_gdf, bgt_types=None, offset=0, padding=0):
        self.label = label
        self.bgt_gdf = bgt_gdf
        if bgt_types is None:
            self.bgt_types = bgt_gdf['bgt-functie'].unique().tolist()
        else:
            self.bgt_types = bgt_types
        self.offset = offset
        self.padding = padding

    def get_labels(self, points, labels, mask, tilecode):
        if np.count_nonzero(mask) == 0:
            return labels

        logger.info(f'BGTRoadFuser: relabeling ground points to {self.label}.')

        # Filter GeoDataFrame by requested road types
        road_polygons = self.bgt_gdf[self.bgt_gdf['bgt-functie'].isin(self.bgt_types)]['geometry'].tolist()
        if len(road_polygons) == 0:
            logger.debug('No road polygons in tile.')
            return labels

        # Consider only ground points for road labeling
        mask_ground = labels == Labels.Labels.GROUND
        pts_masked = points[mask_ground]

        if len(pts_masked) == 0:
            return labels

        inside_mask = np.zeros(len(pts_masked), dtype=bool)

        for poly in road_polygons:
            poly_buffer = poly.buffer(self.offset)
            clip_mask = clipping_tools.poly_clip(pts_masked[:, :2], poly_buffer)
            inside_mask |= clip_mask

        # Map back to original indices
        ground_indices = np.where(mask_ground)[0]
        labels[ground_indices[inside_mask]] = Labels.Labels.ROAD

        return labels