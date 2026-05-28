"""BGT readers for polygon-shaped (non-grid) tiles."""

import json
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def _geojson_bounds(geojson_path):
    """
    Return (x_min, y_min, x_max, y_max) for all geometries in a GeoJSON file
    without requiring geopandas/fiona.
    """
    with open(geojson_path) as f:
        data = json.load(f)

    xs, ys = [], []
    for feature in data.get('features', []):
        geom = feature.get('geometry') or {}
        coords_list = geom.get('coordinates', [])
        geom_type = geom.get('type', '')
        if geom_type == 'Polygon':
            for ring in coords_list:
                for x, y in ring:
                    xs.append(x); ys.append(y)
        elif geom_type == 'MultiPolygon':
            for poly in coords_list:
                for ring in poly:
                    for x, y in ring:
                        xs.append(x); ys.append(y)
    if not xs:
        raise ValueError(f'No polygon coordinates found in {geojson_path}')
    return min(xs), min(ys), max(xs), max(ys)


def get_bbox_from_polygon_file(tilecode, bbox_folder, padding=0):
    """
    Return the axis-aligned bounding box of the tile polygon stored in
    ``<bbox_folder>/bbox_<tilecode>.geojson``.

    Returns
    -------
    ((x_min, y_max), (x_max, y_min))
        Same convention as upcp's ``get_bbox_from_tile_code``.
    """
    path = Path(bbox_folder) / f'bbox_{tilecode}.geojson'
    x_min, y_min, x_max, y_max = _geojson_bounds(path)
    return (
        (x_min - padding, y_max + padding),
        (x_max + padding, y_min - padding),
    )


class BGTPointReader:
    """
    Reads BGT point objects (poles, benches, bins, …) from a CSV file with
    columns ``bgt_type, x, y`` and filters them by the spatial extent of the
    tile polygon, not by a regular grid.

    Parameters
    ----------
    bgt_file : str or Path or None
        Single CSV file to load.
    bgt_folder : str or Path or None
        Folder of CSV files to load (all ``*.csv`` files are concatenated).
    bbox_folder : str or Path or None
        Folder that contains ``bbox_<tilecode>.geojson`` files.  When
        provided, the actual tile polygon is used for spatial filtering.
        When None the tilecode is interpreted as ``XXXXXX_YYYYYY`` (full
        RD coordinates) and a 50 m × 50 m bounding box is derived.
    """

    COLUMNS = ['bgt_type', 'x', 'y']

    def __init__(self, bgt_file=None, bgt_folder=None, bbox_folder=None):
        self.bbox_folder = Path(bbox_folder) if bbox_folder else None
        self.bgt_df = pd.DataFrame(columns=self.COLUMNS)

        if bgt_file is not None and bgt_folder is not None:
            raise ValueError('Provide either bgt_file or bgt_folder, not both.')

        if bgt_file is not None:
            path = Path(bgt_file)
            if not path.exists():
                raise FileNotFoundError(f'BGT file not found: {bgt_file}')
            self.bgt_df = pd.read_csv(path, header=0, names=self.COLUMNS)

        elif bgt_folder is not None:
            folder = Path(bgt_folder)
            frames = [pd.read_csv(f, header=0, names=self.COLUMNS)
                      for f in folder.glob('*.csv')]
            if frames:
                self.bgt_df = pd.concat(frames, ignore_index=True)
            else:
                logger.warning(f'No CSV files found in {bgt_folder}.')

    def _get_bbox(self, tilecode, padding):
        """Return ((x_min, y_max), (x_max, y_min)) for the given tile."""
        if self.bbox_folder is not None:
            return get_bbox_from_polygon_file(tilecode, self.bbox_folder,
                                              padding=padding)
        # Fallback for regular 50 m grid tiles with 6-digit RD coordinates.
        parts = tilecode.split('_')
        x_min = float(parts[0]) - padding
        y_min = float(parts[1]) - padding
        return (
            (x_min, y_min + 50 + 2 * padding),
            (x_min + 50 + 2 * padding, y_min),
        )

    def filter_tile(self, tilecode, bgt_types=None, padding=0,
                    return_types=False):
        """
        Return BGT point objects that fall within the tile bounding box.

        Parameters
        ----------
        tilecode : str
        bgt_types : list of str or None
            If provided only objects whose ``bgt_type`` is in the list are
            returned.
        padding : float
            Extra metres to expand the search area around the tile.
        return_types : bool
            When True return a list of ``(bgt_type, x, y)`` tuples; when
            False (default) return a list of ``(x, y)`` tuples.

        Returns
        -------
        list
        """
        ((bx_min, by_max), (bx_max, by_min)) = self._get_bbox(tilecode,
                                                               padding)

        query = ('(x >= @bx_min) & (x <= @bx_max)'
                 ' & (y >= @by_min) & (y <= @by_max)')
        if bgt_types:
            query = '(bgt_type in @bgt_types) & ' + query

        df = self.bgt_df.query(query)
        records = list(df.to_records(index=False))

        if return_types:
            return records
        return [(float(x), float(y)) for (_, x, y) in records]
