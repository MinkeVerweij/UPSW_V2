"""Car fuser — labels parked cars using parkeervak polygon data.

Only clusters that overlap or are near a parkeervak are labelled.
"""

import logging
import warnings

import numpy as np
from shapely.geometry import Point, Polygon

from .ahn_reader import FastGridInterpolator
from .lcc import LabelConnectedComp
from .math_utils import minimum_bounding_rectangle

logger = logging.getLogger(__name__)


def _parkeervak_axes(park_poly):
    """Return (short_unit_vec, long_unit_vec) from the parkeervak MBR."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mbr = park_poly.minimum_rotated_rectangle
    coords = np.array(mbr.exterior.coords)
    e1 = coords[1] - coords[0]
    e2 = coords[2] - coords[1]
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    # Degenerate polygon — fall back to bbox axes
    if l1 < 1e-6 or l2 < 1e-6:
        bx, by, bx2, by2 = park_poly.bounds
        if (bx2 - bx) < (by2 - by):
            return np.array([1.0, 0.0]), np.array([0.0, 1.0])
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])
    if l1 < l2:
        return e1 / l1, e2 / l2
    return e2 / l2, e1 / l1


class CarFuser:
    """
    Labels parked cars using parkeervak polygons and road-surface proximity.

    Detection runs in four passes, from most to least constrained:

    1. **Parallel / strict** — full car MBR dimensions, cluster overlaps a
       parkeervak by at least ``overlap_perc`` %.
    2. **Parallel / near** — full car MBR dimensions, cluster centroid within
       ``park_proximity`` m of any parkeervak.  Catches imprecisely parked or
       angled cars.
    3. **Perpendicular / strict** — cluster overlaps a parkeervak; dimensions
       are checked in the parkeervak's own coordinate axes so only the visible
       front/back face is required.
    4. **Perpendicular / near** — same axis check, but distance-based rather
       than overlap-based.

    Parameters
    ----------
    label : int
    bgt_poly_reader : BGTPolyReader
    ahn_reader : PolygonNPZReader or None
    grid_size : float
        Voxel size for LCC clustering (default 0.1 m).
    min_component_size : int
    overlap_perc : float
        Minimum overlap (% of cluster MBR area) with a parkeervak for strict
        passes (default 20).
    params : dict
        ``min_height``, ``max_height``, ``min_width``, ``max_width``,
        ``min_length``, ``max_length``,
        ``min_partial_length`` — minimum visible depth for perpendicular cars
        (default 0.8 m),
        ``park_proximity`` — metres from cluster centroid to nearest parkeervak
        for the near passes (default 4.0 m; set 0 to disable).
    """

    _DEFAULT_PARAMS = {
        'min_height':        1.2,
        'max_height':        2.2,
        'min_width':         1.4,
        'max_width':         2.5,
        'min_length':        3.0,   # parallel parking: full car length visible
        'max_length':        6.0,
        'min_partial_length': 0.8,  # perpendicular: only front/back face visible
        'park_proximity':    4.0,   # metres to parkeervak for near-pass
    }

    def __init__(self, label, bgt_poly_reader, ahn_reader=None,
                 grid_size=0.1, min_component_size=100,
                 overlap_perc=20, params=None):
        self.label = label
        self.bgt_poly_reader = bgt_poly_reader
        self.ahn_reader = ahn_reader
        self.grid_size = grid_size
        self.min_component_size = min_component_size
        self.overlap_perc = overlap_perc
        self.params = {**self._DEFAULT_PARAMS, **(params or {})}

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    def _overlaps_park(self, mbr_poly, parks):
        """True if mbr_poly overlaps any park by at least overlap_perc%."""
        for park in parks:
            if not mbr_poly.intersects(park):
                continue
            if (mbr_poly.intersection(park).area / mbr_poly.area * 100
                    >= self.overlap_perc):
                return True
        return False

    def _near_park(self, centroid, parks, max_dist):
        """True if centroid is within max_dist metres of any parkeervak."""
        for park in parks:
            if park.distance(centroid) <= max_dist:
                return True
        return False

    def _perp_fits(self, xy, park):
        """
        True if xy cluster has car-width in the parkeervak's short axis and
        sufficient visible depth in its long axis.
        """
        p = self.params
        short_vec, long_vec = _parkeervak_axes(park)
        extent_w = float(np.ptp(xy @ short_vec))
        extent_l = float(np.ptp(xy @ long_vec))
        return (p['min_width'] <= extent_w <= p['max_width']
                and p['min_partial_length'] <= extent_l <= p['max_length'])

    def _perp_overlap_check(self, mbr_poly, xy, parks):
        """Perpendicular check against parks the cluster overlaps."""
        for park in parks:
            if not mbr_poly.intersects(park):
                continue
            if (mbr_poly.intersection(park).area / mbr_poly.area * 100
                    < self.overlap_perc):
                continue
            if self._perp_fits(xy, park):
                return True
        return False

    def _perp_near_check(self, xy, parks, max_dist):
        """Perpendicular check against parks within max_dist of the cluster."""
        centroid = Point(xy.mean(axis=0))
        for park in parks:
            if park.distance(centroid) > max_dist:
                continue
            if self._perp_fits(xy, park):
                return True
        return False

    # ------------------------------------------------------------------
    # Main clustering logic
    # ------------------------------------------------------------------

    def _label_car_clusters(self, points, ground_z, comp_labels, park_polygons):
        car_mask  = np.zeros(len(points), dtype=bool)
        car_count = 0
        p         = self.params
        park_prox = p['park_proximity']

        for cc in set(comp_labels) - {-1}:
            cc_mask = comp_labels == cc
            cc_pts  = points[cc_mask]

            # Height check
            gz       = ground_z[cc_mask]
            valid_gz = gz[np.isfinite(gz)]
            base_z   = float(np.mean(valid_gz)) if valid_gz.size else 0.0
            top_z    = float(cc_pts[:, 2].max())
            if not (base_z + p['min_height'] <= top_z <= base_z + p['max_height']):
                continue

            xy = cc_pts[:, :2]
            if len(np.unique(xy, axis=0)) < 4:
                continue

            try:
                corners, _, mbr_w, mbr_l, _ = minimum_bounding_rectangle(xy)
            except Exception:
                continue

            mbr_poly   = Polygon(np.vstack([corners, corners[0]]))
            width_ok   = p['min_width']  < mbr_w < p['max_width']
            length_ok  = p['min_length'] < mbr_l < p['max_length']

            matched = False

            if width_ok and length_ok:
                # Passes 1 & 2: full car visible
                centroid = Point(float(xy[:, 0].mean()), float(xy[:, 1].mean()))
                if self._overlaps_park(mbr_poly, park_polygons):
                    matched = True
                elif park_prox > 0 and self._near_park(centroid, park_polygons, park_prox):
                    matched = True

            if not matched:
                # Passes 3 & 4: perpendicular / partially visible
                if self._perp_overlap_check(mbr_poly, xy, park_polygons):
                    matched = True
                elif park_prox > 0 and self._perp_near_check(xy, park_polygons, park_prox):
                    matched = True

            if matched:
                car_mask[cc_mask] = True
                car_count += 1

        logger.debug(f'{car_count} cars labelled.')
        return car_mask

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def get_labels(self, points, labels, mask, tilecode):
        """
        Label parked cars in the point cloud.

        Parameters
        ----------
        points   : (N, 3) array
        labels   : (N,) array
        mask     : (N,) bool — only unlabeled points are considered
        tilecode : str

        Returns
        -------
        Updated labels array.
        """
        logger.info(f'CarFuser (label={self.label}).')

        park_polygons = self.bgt_poly_reader.filter_tile(tilecode)
        if not park_polygons:
            logger.debug('No parkeervakken for tile, skipping.')
            return labels

        if self.ahn_reader is not None:
            ahn_tile = self.ahn_reader.filter_tile(tilecode)
            fast_z   = FastGridInterpolator(
                ahn_tile['x'], ahn_tile['y'], ahn_tile['ground_surface'])
            ground_z = fast_z(points[mask][:, :2])
        else:
            ground_z = np.zeros(mask.sum())

        lcc = LabelConnectedComp(
            grid_size=self.grid_size,
            min_component_size=self.min_component_size)
        comp_labels = lcc.get_components(points[mask])

        car_mask_local = self._label_car_clusters(
            points[mask], ground_z, comp_labels, park_polygons)

        label_mask = np.zeros(len(points), dtype=bool)
        label_mask[mask] = car_mask_local
        labels[label_mask] = self.label

        logger.debug(f'{label_mask.sum():,} car points labelled.')
        return labels
