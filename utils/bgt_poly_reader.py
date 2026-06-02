"""Reader for BGT polygon objects (parkeervakken)."""

import json
import logging
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, box

logger = logging.getLogger(__name__)


def _geojson_bounds(geojson_path):
    with open(geojson_path) as f:
        data = json.load(f)
    xs, ys = [], []
    for feat in data.get('features', []):
        geom = feat.get('geometry') or {}
        gtype = geom.get('type', '')
        if gtype == 'Polygon':
            for ring in geom['coordinates']:
                for x, y in ring:
                    xs.append(x); ys.append(y)
        elif gtype == 'MultiPolygon':
            for poly in geom['coordinates']:
                for ring in poly:
                    for x, y in ring:
                        xs.append(x); ys.append(y)
    if not xs:
        raise ValueError(f'No polygon coordinates found in {geojson_path}')
    return min(xs), min(ys), max(xs), max(ys)


class BGTPolyReader:
    """
    Reads parking-space polygon data from a JSON file and provides
    tile-filtered lists of Shapely Polygons for use in CarFuser.

    The JSON file is a list of objects with at least a ``coords`` key
    containing the exterior ring coordinates [[x, y], ...] in RD New.

    Parameters
    ----------
    parkeervakken_file : str or Path
        Path to ``bgt_parkeervakken.json`` produced by the scraper.
    bbox_folder : str or Path or None
        Folder of ``bbox_<tilecode>.geojson`` files. When provided the
        actual tile polygon extent is used for filtering; otherwise the
        tilecode is parsed as ``XXXXXX_YYYYYY`` (RD origin, 50 m grid).
    """

    def __init__(self, parkeervakken_file, bbox_folder=None):
        self.bbox_folder = Path(bbox_folder) if bbox_folder else None
        path = Path(parkeervakken_file)
        if not path.exists():
            raise FileNotFoundError(f'Parkeervakken file not found: {path}')
        with open(path) as f:
            raw = json.load(f)
        self._polygons = []
        # Support both GeoJSON FeatureCollection and plain list-of-coords formats
        if isinstance(raw, dict) and raw.get('type') == 'FeatureCollection':
            items = raw.get('features', [])
        elif isinstance(raw, list):
            items = raw
        else:
            items = []
        for item in items:
            if isinstance(item, dict) and 'geometry' in item:
                # GeoJSON feature
                geom = item.get('geometry') or {}
                gtype = geom.get('type', '')
                rings = []
                if gtype == 'Polygon':
                    rings = geom.get('coordinates', [])[:1]
                elif gtype == 'MultiPolygon':
                    rings = [p[0] for p in geom.get('coordinates', []) if p]
                for ring in rings:
                    if len(ring) >= 3:
                        try:
                            poly = Polygon(ring)
                            if poly.is_valid and not poly.is_empty:
                                self._polygons.append(poly)
                        except Exception:
                            pass
            else:
                # Legacy plain-list format with 'coords' key
                coords = item.get('coords', []) if isinstance(item, dict) else []
                if len(coords) >= 3:
                    try:
                        poly = Polygon(coords)
                        if poly.is_valid and not poly.is_empty:
                            self._polygons.append(poly)
                    except Exception:
                        pass
        logger.info(f'Loaded {len(self._polygons)} parkeervakken polygons.')

    def _get_tile_bbox(self, tilecode, padding=0):
        if self.bbox_folder is not None:
            path = self.bbox_folder / f'bbox_{tilecode}.geojson'
            x_min, y_min, x_max, y_max = _geojson_bounds(path)
        else:
            parts = tilecode.split('_')
            x_min = float(parts[0])
            y_min = float(parts[1])
            x_max = x_min + 50
            y_max = y_min + 50
        return (x_min - padding, y_min - padding,
                x_max + padding, y_max + padding)

    def filter_tile(self, tilecode, padding=5):
        """
        Return Shapely Polygons that intersect the tile bounding box.

        Parameters
        ----------
        tilecode : str
        padding : float
            Extra metres around the tile when checking intersections.

        Returns
        -------
        list of shapely.geometry.Polygon
        """
        x_min, y_min, x_max, y_max = self._get_tile_bbox(tilecode, padding)
        tile_box = box(x_min, y_min, x_max, y_max)
        return [p for p in self._polygons if tile_box.intersects(p)]
