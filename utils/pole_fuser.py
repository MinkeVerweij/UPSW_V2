"""BGT pole fuser (trees, street lights, traffic signs) — ported from upcp."""

import numpy as np
import logging
from sklearn.cluster import DBSCAN
from scipy.stats import binned_statistic_2d

from .ahn_reader import FastGridInterpolator
from .lcc import LabelConnectedComp
from . import clipping_tools

logger = logging.getLogger(__name__)


class BGTPoleFuser:
    """
    Labels pole-like BGT point objects (trees, street lights, traffic signs).

    For each BGT location the fuser searches the point cloud for a matching
    cluster, then labels a cylinder of points around that cluster.

    Parameters
    ----------
    label : int
        Class label to assign.
    bgt_type : str
        BGT object type: ``'boom'``, ``'lichtmast'``, or ``'verkeersbord'``.
    bgt_reader : BGTPointReader
        Reader used to load BGT point locations.
    ahn_reader : PolygonNPZReader
        Used to retrieve ground elevation.
    padding : float (default: 0)
        Extra metres around the tile when loading BGT objects.
    params : dict
        Override default detection parameters (see Notes).

    Notes
    -----
    Default ``params``:

    ``search_pad`` (1.5 m)
        Half-width of the search box around each BGT location.
    ``max_dist`` (1.2 m)
        Maximum horizontal distance between the BGT location and the
        detected cluster centre to count as a match. Used both to accept
        the coarse candidate column and (forwarded to _find_point_cluster)
        the fine-grained trunk/pole cluster within it — it was previously
        only applied to the coarse step, with the fine step silently using
        an unrelated hardcoded 0.1m default that almost never matched.
    ``voxel_res`` (0.2 m)
        Voxel resolution for the column-height analysis.
    ``seed_height`` (1.75 m)
        Height above ground at which to cross-section the object to find
        its radius.
    ``min_height`` (2.0 m)
        Minimum column height for a voxel column to be a candidate.
    ``max_r`` (0.5 m)
        Maximum radius of a candidate cluster.
    ``min_points`` (5)
        Minimum points in a voxel column to be considered. Real lamp posts
        and trees were measured (on the 0.08m pre-fusion downsampled 2025
        clouds) reaching 63-272 and 3-230 points/column respectively — this
        just needs to clear stray-noise columns, since `min_height` and the
        med_mid shape check already reject anything that isn't a genuine
        vertical column.
    ``cluster_eps`` (0.15 m)
        DBSCAN neighbourhood radius used to find the fine-grained trunk/pole
        cluster at seed_height, once a candidate column has passed the
        min_points/min_height checks. Must be comfortably larger than the
        point spacing produced by pre-fusion voxel downsampling (0.08m as of
        writing) — an eps at or below that spacing (the previous hardcoded
        0.05m) frequently finds zero neighbours and misses real objects.
    ``z_min`` (0.2 m)
        Minimum height above ground for the search box.
    ``z_max`` (2.7 m)
        Maximum height above ground for the search box.
    ``r_mult`` (1.5)
        Radius multiplier for the final cylinder label.
    ``label_height`` (4.0 m)
        Maximum height for the initial cylinder label above ground.
    ``grow_crown`` (False)
        If True, run voxel connected-components after the initial cylinder
        label to expand into the full tree crown.  Intended for trees; keep
        False for lamp posts / traffic signs.
    ``crown_r`` (5.0 m)
        Radius of the search container used for crown growing.
    ``crown_height`` (25.0 m)
        Maximum height above ground of the crown search container.
    ``crown_floor`` (1.75 m)
        Height above ground at which crown growing starts. Points below this
        height are excluded from the LCC container, preventing ground-level
        objects (bikes, scooters, signs) from connecting to the tree.
    ``crown_grid_size`` (0.25 m)
        Voxel size for the connected-components crown expansion.
    ``crown_min_comp`` (20)
        Minimum number of points a voxel component must have to be included
        in the crown expansion.
    """

    _DEFAULTS = {
        'search_pad': 1.5,
        'max_dist': 1.2,
        'voxel_res': 0.2,
        'seed_height': 1.75,
        'min_height': 2.0,
        'max_r': 0.5,
        'min_points': 5,
        'cluster_eps': 0.15,
        'z_min': 0.2,
        'z_max': 2.7,
        'r_mult': 1.5,
        'label_height': 4.0,
        'grow_crown': False,
        'crown_r': 5.0,
        'crown_height': 25.0,
        'crown_floor': 1.75,
        'crown_grid_size': 0.25,
        'crown_min_comp': 20,
        'crown_expand_step': 2.0,
        'crown_max_iter': 2,
    }

    def __init__(self, label, bgt_type, bgt_reader, ahn_reader=None,
                 padding=0, params=None):
        self.label = label
        self.bgt_type = bgt_type
        self.bgt_reader = bgt_reader
        self.ahn_reader = ahn_reader
        self.padding = padding
        self.params = {**self._DEFAULTS, **(params or {})}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_point_cluster(self, points, point, plane_height,
                            plane_buffer=0.1, search_radius=1.0,
                            max_dist=0.1, min_points=1, max_r=0.5,
                            cluster_eps=0.15):
        """Find a cluster near ``point`` at ``plane_height``."""
        search_ids = np.where(
            clipping_tools.cylinder_clip(
                points, point, search_radius,
                bottom=plane_height - plane_buffer,
                top=plane_height + plane_buffer)
        )[0]
        if len(search_ids) < min_points:
            return []

        clustering = DBSCAN(eps=cluster_eps, min_samples=5, p=2).fit(
            points[search_ids])
        noise_mask = clustering.labels_ != -1
        cc_labels, counts = np.unique(clustering.labels_, return_counts=True)
        if min_points > 1:
            cc_labels = cc_labels[counts >= min_points]
            noise_mask = noise_mask & np.isin(clustering.labels_, cc_labels)

        c_xyr_list = []
        for cl in set(cc_labels).difference((-1,)):
            c_mask = clustering.labels_ == cl
            cx, cy = np.mean(points[search_ids[c_mask], :2], axis=0)
            cr = (np.max(points[search_ids[c_mask], :2], axis=0)
                  - np.min(points[search_ids[c_mask], :2], axis=0)).max() / 2
            if cr <= max_r and ((point[0]-cx)**2 + (point[1]-cy)**2
                                 <= (cr + max_dist)**2):
                c_xyr_list.append([cx, cy, cr])
                break
        return c_xyr_list

    def _find_seeds(self, points, point_objects, fast_z,
                    search_pad, max_dist, voxel_res, seed_height,
                    min_height, min_points, max_r, z_min, z_max,
                    cluster_eps=0.15, **_):
        """Locate seed clusters matching each BGT location."""
        seeds = []
        matches = {}
        for obj in point_objects:
            ground_z = float(fast_z(np.array([obj]))[0]) if fast_z else 0.0
            if np.isnan(ground_z):
                ground_z = 0.0

            search_box = (obj[0]-search_pad, obj[1]-search_pad,
                          obj[0]+search_pad, obj[1]+search_pad)
            box_ids = np.where(
                clipping_tools.box_clip(points, search_box,
                                        bottom=ground_z + z_min,
                                        top=ground_z + z_max)
            )[0]
            if len(box_ids) == 0:
                matches[obj] = None
                continue

            x_edge = np.arange(search_box[0], search_box[2]+0.01, voxel_res)
            y_edge = np.arange(search_box[1], search_box[3]+0.01, voxel_res)

            min_z_bin = binned_statistic_2d(
                points[box_ids, 0], points[box_ids, 1], points[box_ids, 2],
                bins=[x_edge, y_edge], statistic='min')
            max_z_bin = binned_statistic_2d(
                points[box_ids, 0], points[box_ids, 1], points[box_ids, 2],
                bins=[x_edge, y_edge], statistic='max')
            med_z_bin = binned_statistic_2d(
                points[box_ids, 0], points[box_ids, 1], points[box_ids, 2],
                bins=[x_edge, y_edge], statistic='median')
            count_bin = binned_statistic_2d(
                points[box_ids, 0], points[box_ids, 1], points[box_ids, 2],
                bins=[x_edge, y_edge], statistic='count')

            height = max_z_bin.statistic - min_z_bin.statistic
            midpoint = (min_z_bin.statistic + max_z_bin.statistic) / 2
            med_mid = np.abs(med_z_bin.statistic - midpoint) < 0.2 * height

            x_loc, y_loc = np.where(
                (height > min_height)
                & (count_bin.statistic > min_points)
                & med_mid
            )
            if len(x_loc) == 0:
                matches[obj] = None
                continue

            candidates = np.stack((x_edge[x_loc] + voxel_res/2,
                                   y_edge[y_loc] + voxel_res/2)).T
            dists = [np.linalg.norm(np.array(obj) - c) for c in candidates]
            c_prime = candidates[np.argmin(dists)]

            if min(dists) <= max_dist:
                clusters = self._find_point_cluster(
                    points, c_prime, ground_z + seed_height,
                    max_r=max_r, max_dist=max_dist, cluster_eps=cluster_eps)
                if clusters:
                    seed = clusters[0]
                    seeds.append(seed)
                    matches[obj] = (seed[0], seed[1])
                else:
                    matches[obj] = None
            else:
                matches[obj] = None

        return seeds, matches

    # ------------------------------------------------------------------
    # Crown growing
    # ------------------------------------------------------------------

    def _grow_crown(self, points, mask, label_mask, fast_z, seeds):
        """
        Expand labeled trunk points to the full crown using voxel LCC.

        Pass 0 seeds from the trunk cylinder (radius = crown_r).
        Passes 1..crown_max_iter each expand the search radius by
        crown_expand_step and re-seed from the full grown crown so that
        outer branches connected to the mid-crown (not directly to the trunk)
        are also captured.  Stops early when a pass adds no new points.
        """
        lcc = LabelConnectedComp(
            grid_size=self.params['crown_grid_size'],
            min_component_size=self.params['crown_min_comp'])

        grown_mask   = label_mask.copy()
        mask_indices = np.where(mask)[0]
        expand_step  = self.params['crown_expand_step']
        max_iter     = self.params['crown_max_iter']

        for seed in seeds:
            cx, cy = seed[0], seed[1]
            if fast_z is not None:
                ground_z = float(fast_z(np.array([[cx, cy]]))[0])
                if np.isnan(ground_z):
                    ground_z = 0.0
            else:
                ground_z = 0.0

            search_r = self.params['crown_r']

            for _ in range(1 + max_iter):
                container = clipping_tools.cylinder_clip(
                    points[mask], np.array([cx, cy]),
                    search_r,
                    bottom=ground_z + self.params['crown_floor'],
                    top=ground_z + self.params['crown_height'])

                container_ids = mask_indices[container]
                if len(container_ids) < 2:
                    break

                comp_labels = lcc.get_components(points[container_ids])

                # Seed from all currently grown points inside the container
                seed_comps = set(comp_labels[grown_mask[container_ids]]) - {-1}
                if not seed_comps:
                    break

                prev_count = grown_mask.sum()
                crown_mask = np.isin(comp_labels, list(seed_comps))
                grown_mask[container_ids[crown_mask]] = True

                if grown_mask.sum() == prev_count:
                    break  # nothing new found, stop early

                search_r += expand_step

        n_added = grown_mask.sum() - label_mask.sum()
        logger.debug(f'  Crown growing added {n_added:,} points.')
        return grown_mask

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def get_labels(self, points, labels, mask, tilecode):
        """
        Label pole-like objects in the point cloud.

        Parameters
        ----------
        points : (N, 3) array
        labels : (N,) array
        mask   : (N,) bool array — only unlabelled points are considered
        tilecode : str

        Returns
        -------
        Updated labels array.
        """
        logger.info(f'BGTPoleFuser [{self.bgt_type}] '
                    f'(label={self.label}).')

        bgt_points = self.bgt_reader.filter_tile(
            tilecode, bgt_types=[self.bgt_type],
            padding=self.padding, return_types=False)
        if not bgt_points:
            logger.debug(f'No {self.bgt_type} in tile, skipping.')
            return labels

        if self.ahn_reader is not None:
            ahn_tile = self.ahn_reader.filter_tile(tilecode)
            fast_z = FastGridInterpolator(
                ahn_tile['x'], ahn_tile['y'], ahn_tile['ground_surface'])
        else:
            fast_z = None

        seeds, matches = self._find_seeds(
            points[mask], bgt_points, fast_z, **self.params)

        label_mask = np.zeros(len(points), dtype=bool)
        for seed in seeds:
            if fast_z is not None:
                top = (float(fast_z(np.array([seed[:2]]))[0])
                       + self.params['label_height'])
            else:
                top = self.params['label_height']
            clip = clipping_tools.cylinder_clip(
                points[mask],
                np.array(seed[:2]),
                self.params['r_mult'] * seed[2],
                top=top)
            label_mask[mask] = label_mask[mask] | clip

        if self.params['grow_crown'] and label_mask.any():
            label_mask = self._grow_crown(
                points, mask, label_mask, fast_z, seeds)

        labels[label_mask] = self.label
        logger.debug(f'{len(seeds)}/{len(bgt_points)} {self.bgt_type} labelled.')
        return labels
