"""Cable Fuser — adapted from Urban PointCloud Processing (Amsterdam AI Team, GPL-3.0)"""

import logging
import warnings
import numpy as np

# np.ndarray.ptp() was removed in NumPy 2.0; pyntcloud's voxelgrid still uses it.
if not hasattr(np.ndarray, 'ptp'):
    np.ndarray.ptp = lambda self, axis=None, **kw: (
        self.max(axis=axis) - self.min(axis=axis)
    )

from .ahn_reader import FastGridInterpolator
from .clipping_tools import poly_clip, compute_bounding_box, minimum_bounding_rectangle

import open3d as o3d
import pandas as pd
from pyntcloud import PyntCloud
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from scipy import ndimage
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.optimize import OptimizeWarning

logger = logging.getLogger(__name__)


def poly_box_clip(points, poly, bottom=-np.inf, top=np.inf):
    """Clip 3D points within a polygon and a height range."""
    mask = poly_clip(points[:, :2], poly)
    mask = mask & (points[:, 2] >= bottom) & (points[:, 2] <= top)
    return mask


def get_polygon_from_tile_code(tilecode, padding=0, width=50, height=50):
    parts = tilecode.split('_')
    x_min = float(parts[0]) - padding
    y_min = float(parts[1]) - padding
    x_max = x_min + width + 2 * padding
    y_max = y_min + height + 2 * padding
    return Polygon([(x_min, y_min), (x_min, y_max),
                    (x_max, y_max), (x_max, y_min)])


def voxelize(points, voxel_size):
    cloud = PyntCloud(pd.DataFrame(points, columns=['x', 'y', 'z']))
    voxelgrid_id = cloud.add_structure(
        "voxelgrid",
        size_x=voxel_size, size_y=voxel_size, size_z=voxel_size,
        regular_bounding_box=False,
    )
    voxel_grid = cloud.structures[voxelgrid_id]
    voxel_centers = voxel_grid.voxel_centers[np.unique(voxel_grid.voxel_n)]
    inv_voxel_idx = np.unique(voxel_grid.voxel_n, return_inverse=True)[1]
    return voxel_grid, voxel_centers, inv_voxel_idx


def to_open3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def catenary_func(x, a, b, c):
    return a + c * np.cosh((x - b) / c)


def unit_vector(v1):
    return v1 / np.linalg.norm(v1)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def main_direction(points):
    cov = np.cov(points, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov)
    dir_v = eig_vec[:, eig_val.argmax()]
    if dir_v[0] < 0:
        dir_v *= -1
    return dir_v


class CableFuser:
    """
    Labels cable and suspended streetlight points in a pointcloud.

    Parameters
    ----------
    label : int
        Fallback label (also used as cable_label when cable_label is not
        differentiated).
    cable_label : int
        Label for regular cables.
    tramcable_label : int
        Label for tram overhead wires.
    streetlight_label : int
        Label for armaturen (suspended streetlight hardware).
    ahn_reader : PolygonNPZReader
        AHN elevation reader.
    bag_reader : optional
        Reader for BAG building polygons.  When None, building removal is
        skipped (buildings are already excluded via the pipeline mask).
    bgt_tram_reader : optional
        Reader for BGT tram-track lines.  When None, tram classification is
        skipped and all cables are labelled as regular cables.
    """

    def __init__(self, label, cable_label, tramcable_label, streetlight_label,
                 ahn_reader, bag_reader=None, bgt_tram_reader=None,
                 min_cable_height=4.5, max_cable_height=12,
                 building_offset=1.25, voxel_size=0.09,
                 neigh_radius=0.5, linearity_thres=0.9, max_v_angle=20,
                 grow_radius=0.3, max_merge_angle=3, min_segment_length=3,
                 cable_size=0.1, max_tramcable_height=7.5, min_cable_bending=2,
                 armatuur_params=None, cable_sag_span=2):
        if armatuur_params is None:
            armatuur_params = {'width': (0.2, 1), 'height': (0.15, 1.0),
                               'axis_offset': 0.2}
        self.label = label
        self.cable_label = cable_label
        self.tramcable_label = tramcable_label
        self.streetlight_label = streetlight_label
        self.ahn_reader = ahn_reader
        self.bag_reader = bag_reader
        self.bgt_tram_reader = bgt_tram_reader
        self.min_cable_height = min_cable_height
        self.max_cable_height = max_cable_height
        self.building_offset = building_offset
        self.voxel_size = voxel_size
        self.neigh_radius = neigh_radius
        self.linearity_thres = linearity_thres
        self.max_v_angle = max_v_angle
        self.grow_radius = grow_radius
        self.max_merge_angle = max_merge_angle
        self.min_segment_length = min_segment_length
        self.cable_size = cable_size
        self.max_tramcable_height = max_tramcable_height
        self.min_cable_bending = min_cable_bending
        self.armatuur_params = armatuur_params
        self.cable_sag_span = cable_sag_span

    # ------------------------------------------------------------------ #
    #  Pre-processing                                                       #
    # ------------------------------------------------------------------ #

    def _vertical_segmentation(self, points, tilecode):
        tile = self.ahn_reader.filter_tile(tilecode)
        fast_z = FastGridInterpolator(tile['x'], tile['y'],
                                      tile['ground_surface'])
        ground_z = fast_z(points[:, :2])
        return (
            (points[:, 2] > ground_z + self.min_cable_height) &
            (points[:, 2] < ground_z + self.max_cable_height)
        )

    def _bag_removal(self, points, tilecode, offset):
        if self.bag_reader is None:
            return np.ones(len(points), dtype=bool)

        bag_seg_mask = np.ones(len(points), dtype=bool)
        building_polygons = self.bag_reader.filter_tile(
            tilecode, bgt_types=['pand'], padding=offset, offset=offset,
            merge=True)
        logger.debug(f'Buildings in tile {len(building_polygons)}')
        if len(building_polygons) == 0:
            return bag_seg_mask

        tile_polygon = get_polygon_from_tile_code(tilecode)
        for polygon in building_polygons:
            if polygon.intersects(tile_polygon):
                clip_mask = poly_clip(points[:, :2], polygon)
                bag_seg_mask[clip_mask] = False
        return bag_seg_mask

    # ------------------------------------------------------------------ #
    #  Point feature extraction                                            #
    # ------------------------------------------------------------------ #

    def _point_features(self, pcd, radius, max_nn=50):
        pcd.estimate_covariances(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn))
        eig_val, eig_vec = np.linalg.eig(np.asarray(pcd.covariances))

        index_array = np.flip(eig_val.argsort(), axis=1)
        eig_val = np.take_along_axis(eig_val, index_array, axis=1)
        linearity = np.nan_to_num(
            (eig_val[:, 0] - eig_val[:, 1]) / eig_val[:, 0])

        principal_axis = eig_vec[
            np.arange(len(eig_vec)), :, index_array[:, 0]]

        vertical_angle = np.degrees(
            np.arccos(np.clip(
                np.dot(principal_axis, [0., 0., 1.]), -1.0, 1.0)))
        vertical_angle = 90 - np.abs(vertical_angle - 90)

        return linearity, principal_axis, vertical_angle

    def _neighborhood_analysis(self, points, radius, linearity_thres,
                                max_angle):
        pcd = to_open3d(points)
        linearity, principal_axis, vertical_angle = self._point_features(
            pcd, radius)
        candidate_mask = (linearity > linearity_thres) & \
                         (vertical_angle > (90 - max_angle))
        return candidate_mask, principal_axis

    def _candidate_cable_points(self, points, voxel_size, radius,
                                 linearity_thres, max_angle):
        if voxel_size is not None:
            _, voxel_centers, inv_voxel_idx = voxelize(points, voxel_size)
            candidate_mask, principal_axis = self._neighborhood_analysis(
                voxel_centers, radius, linearity_thres, max_angle)
            principal_axis = principal_axis[inv_voxel_idx]
            candidate_mask = candidate_mask[inv_voxel_idx]
        else:
            candidate_mask, principal_axis = self._neighborhood_analysis(
                points, radius, linearity_thres, max_angle)
        return candidate_mask, principal_axis

    # ------------------------------------------------------------------ #
    #  Cable growing and merging                                           #
    # ------------------------------------------------------------------ #

    def _grow_cables(self, points, cable_labels, principal_axis, radius):
        unassigned_mask = cable_labels == -1
        kd_tree_unassigned = KDTree(points[unassigned_mask])
        kd_tree_cable = KDTree(points[~unassigned_mask])

        indices = kd_tree_cable.query_ball_tree(kd_tree_unassigned, r=radius)
        neighbors_n_parent = np.array(
            [(j, i) for i in range(len(indices)) for j in indices[i]],
            dtype=int)
        if len(neighbors_n_parent) < 2:
            return cable_labels, principal_axis

        neighbors_idx = np.unique(neighbors_n_parent[:, 0],
                                   return_index=True)[0]
        neighbors_idx = np.where(unassigned_mask)[0][neighbors_idx]

        nn_idx = kd_tree_cable.query(
            points[neighbors_idx], distance_upper_bound=radius)[1]
        grow_parent_idx = np.where(~unassigned_mask)[0][nn_idx]
        principal_axis[neighbors_idx] = principal_axis[grow_parent_idx]
        cable_labels[neighbors_idx] = cable_labels[grow_parent_idx]

        return cable_labels, principal_axis

    def _principal_vector(self, points):
        cov = to_open3d(points).compute_mean_and_covariance()[1]
        eig_val, eig_vec = np.linalg.eig(cov)
        return eig_vec[:, eig_val.argmax()]

    def _get_end_points(self, points, principal_axis):
        d_pts = np.dot(points[:, :2], principal_axis[:2])
        idx_a, idx_b = d_pts.argmin(), d_pts.argmax()
        return np.array([points[idx_a], points[idx_b]])

    def _cable_cluster_feature(self, points):
        principal_axis = self._principal_vector(points)
        direction = np.abs(
            np.abs(np.degrees(np.arctan2(principal_axis[1],
                                          principal_axis[0]))) - 90)
        end_points = self._get_end_points(points, principal_axis)
        length = np.linalg.norm(end_points[0] - end_points[1])
        return {
            'counts': len(points),
            'principal_axis': principal_axis,
            'dir': direction,
            'end_points': end_points,
            'length': length,
        }

    def _cable_cluster_features(self, points, labels, exclude_labels=None):
        if exclude_labels is None:
            exclude_labels = []
        cable_clusters = {}
        for label in np.unique(labels):
            if label not in exclude_labels:
                cl_points = points[labels == label]
                cable_clusters[label] = self._cable_cluster_feature(cl_points)
        return cable_clusters

    def _nearest_points(self, pts_a, pts_b):
        dist = cdist(pts_a, pts_b)
        idx_a, idx_b = np.unravel_index(np.argmin(dist), dist.shape)
        return idx_a, idx_b, dist.min()

    def _cluster_distance(self, cluster_a, cluster_b):
        return self._nearest_points(
            cluster_a['end_points'], cluster_b['end_points'])[2]

    def _cluster_merge_condition(self, points, ccl_dict, cl_labels, a, b,
                                  max_dist=4):
        a_end = ccl_dict[a]['end_points']
        b_end = ccl_dict[b]['end_points']
        idx_a, idx_b = self._nearest_points(a_end, b_end)[:2]
        O = a_end[idx_a]
        A = b_end[idx_b]

        a_points_xy_axis = np.abs(
            np.dot((points[cl_labels == a, :2] - O[:2]),
                   ccl_dict[a]['principal_axis'][:2]))
        b_vec = self._principal_vector(
            points[cl_labels == a][a_points_xy_axis < 1.5])

        a_vec = A - O
        a_1 = np.dot(a_vec, b_vec) * b_vec
        a_2 = a_vec - a_1

        dir_dist = np.linalg.norm(a_1)
        offset_dist = np.linalg.norm(a_2)

        merge_bool = (dir_dist < max_dist) and \
                     (offset_dist < max(0.2 + dir_dist * 0.1, 0.2))
        return merge_bool, dir_dist, (O, A)

    def _catenary_merge(self, points, cl_labels, a, b, pt_a, pt_b,
                         cable_width=0.1):
        unmasked_idx = np.where(cl_labels < 1)[0]
        merge_line = LineString([pt_a, pt_b])

        xy_clip_mask = poly_clip(
            points[(cl_labels == a) | (cl_labels == b), :2],
            merge_line.buffer(2, cap_style=3))
        fit_points = points[(cl_labels == a) | (cl_labels == b)][xy_clip_mask]
        principal_v = unit_vector((pt_b - pt_a)[:2])
        x_fit_points = np.dot(fit_points[:, :2], principal_v)
        x_shift = np.min(x_fit_points)
        x_fit_points -= x_shift

        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            try:
                popt, _ = curve_fit(catenary_func, x_fit_points,
                                    fit_points[:, 2])
                errors = abs(catenary_func(x_fit_points, *popt)
                             - fit_points[:, 2])
                fit_score = np.sum(errors < self.cable_size) / len(errors)

                xy_clip_mask = poly_clip(
                    points[unmasked_idx, :2],
                    merge_line.buffer(0.1))
                gap_points = points[unmasked_idx[xy_clip_mask]]
                if len(gap_points) < 1:
                    return 0, 0, np.array([])
                x_gap_points = (np.dot(gap_points[:, :2], principal_v)
                                - x_shift)
                errors = abs(catenary_func(x_gap_points, *popt)
                             - gap_points[:, 2])
                gap_inliers = errors < cable_width / 2
                gap_score = np.sum(gap_inliers) / len(gap_inliers)
                inlier_idx = unmasked_idx[xy_clip_mask][gap_inliers]

            except OptimizeWarning:
                return 0, 0, None

        return fit_score, gap_score, inlier_idx

    def _box_merge(self, points, cl_labels, pt_a, pt_b, cable_width=0.1):
        unmasked_idx = np.where(cl_labels < 1)[0]
        merge_line = LineString([pt_a, pt_b]).buffer(cable_width)
        clip_mask = poly_box_clip(
            points[unmasked_idx], merge_line,
            bottom=np.min((pt_a[2], pt_b[2])) - cable_width,
            top=np.max((pt_a[2], pt_b[2])) + cable_width)

        merge_b = unit_vector(pt_b - pt_a)
        a = points[unmasked_idx[clip_mask]] - pt_a
        a_1 = np.dot(a, merge_b)[np.newaxis, :].T * merge_b
        a_2 = a - a_1
        dist = np.linalg.norm(a_2, axis=1)
        merge_idx = unmasked_idx[clip_mask][dist < cable_width / 2]
        return merge_idx

    def _cable_merging(self, points, cl_labels, max_merge_angle=5):
        for cl in set(np.unique(cl_labels)).difference((-1,)):
            if np.sum(cl_labels == cl) < 4:
                cl_labels[cl_labels == cl] = -1

        ccl_dict = self._cable_cluster_features(points, cl_labels, [-1])
        cl_ordered = sorted(ccl_dict, key=lambda x: ccl_dict[x]['counts'],
                            reverse=True)

        i = 0
        while len(cl_ordered) > i:
            main_cl = cl_ordered[i]
            candidate_cls = cl_ordered[i + 1:]
            i += 1

            search = True
            while search:
                search = False
                direction = [ccl_dict[x]['dir'] for x in candidate_cls]
                angle_mask = (
                    np.abs(ccl_dict[main_cl]['dir'] - direction)
                    < max_merge_angle)
                selection_cls = np.asarray(candidate_cls)[angle_mask]

                if len(selection_cls) > 0:
                    ccls_dist = np.array([
                        self._cluster_distance(ccl_dict[main_cl],
                                               ccl_dict[x])
                        for x in selection_cls])
                    ccls_ordered = ccls_dist.argsort()[
                        :np.sum(ccls_dist < 5)]
                    selection_cls = selection_cls[ccls_ordered]

                    for x in selection_cls:
                        valid_merge, dir_dist, (pt_a, pt_b) = \
                            self._cluster_merge_condition(
                                points, ccl_dict, cl_labels, main_cl, x)
                        if valid_merge:
                            if dir_dist < 1:
                                inlier_mask = self._box_merge(
                                    points, cl_labels, pt_a, pt_b)
                            else:
                                fit_score, _, inlier_mask = \
                                    self._catenary_merge(
                                        points, cl_labels, main_cl, x,
                                        pt_a, pt_b)
                                if fit_score < 0.8:
                                    continue

                            cl_labels[inlier_mask] = main_cl
                            cl_labels[cl_labels == x] = main_cl
                            ccl_dict.pop(x, None)
                            candidate_cls.remove(x)
                            cl_ordered.remove(x)
                            ccl_dict[main_cl] = self._cable_cluster_feature(
                                points[cl_labels == main_cl])
                            search = True
                            break

        return cl_labels, ccl_dict

    # ------------------------------------------------------------------ #
    #  Main cable detection                                                #
    # ------------------------------------------------------------------ #

    def _detect_cables(self, points):
        logger.debug('Selecting candidate cable points...')
        candidate_mask, principal_axis = self._candidate_cable_points(
            points, voxel_size=self.voxel_size, radius=self.neigh_radius,
            linearity_thres=self.linearity_thres, max_angle=self.max_v_angle)
        if np.sum(candidate_mask) == 0:
            return np.full(len(points), -1)

        logger.debug('Clustering candidates...')
        cable_labels = np.full(len(points), -1)
        clustering = DBSCAN(eps=self.neigh_radius, min_samples=1,
                            p=2).fit(points[candidate_mask])
        cable_labels[candidate_mask] = clustering.labels_

        logger.debug('Growing cables...')
        cable_labels, principal_axis = self._grow_cables(
            points, cable_labels, principal_axis, radius=self.grow_radius)

        logger.debug('Merging cables...')
        cable_labels, ccl_dict = self._cable_merging(
            points, cable_labels, max_merge_angle=self.max_merge_angle)

        short_clusters = [
            key for key in ccl_dict
            if ccl_dict[key]['length'] < self.min_segment_length]
        cable_labels[np.isin(cable_labels, short_clusters)] = -1

        return cable_labels

    # ------------------------------------------------------------------ #
    #  Tram cable classification                                           #
    # ------------------------------------------------------------------ #

    def _cable_cut(self, points):
        cable_dir = main_direction(points[:, :2])
        cable_dir_axis = np.dot(points[:, :2], cable_dir)
        cable_dir_axis -= cable_dir_axis.min()
        return cable_dir_axis

    def _linestring_cable_fit(self, points, cable_axis, binwidth_axis=0.5):
        cable_max = cable_axis.max()
        bins = np.linspace(
            0 - (binwidth_axis / 2),
            cable_max + (binwidth_axis / 2),
            int(round(cable_max / binwidth_axis) + 2))
        mean_x, _, _ = binned_statistic(cable_axis, points[:, 0],
                                         statistic='mean', bins=bins)
        mean_y, _, _ = binned_statistic(cable_axis, points[:, 1],
                                         statistic='mean', bins=bins)
        line_pts = np.vstack((mean_x, mean_y)).T
        line_pts = line_pts[~np.isnan(line_pts).any(axis=1)]
        return LineString(line_pts)

    def _classify_tram_cables(self, points, cable_labels, tilecode):
        if self.bgt_tram_reader is None:
            return []

        logger.debug('Classifying tram cables...')
        tram_cable_labels = []

        tramtracks = self.bgt_tram_reader.filter_tile(tilecode, padding=10)
        logger.debug(f'{len(tramtracks)} tramtracks in tile.')
        if len(tramtracks) == 0:
            return tram_cable_labels

        cable_mask = cable_labels > -1
        ground_z = self.ahn_reader.interpolate(
            tilecode, points, cable_mask, 'ground_surface')

        tramtracks_polygon = unary_union(tramtracks).buffer(2.8 + 2).buffer(-2.8)

        for cl in set(np.unique(cable_labels[cable_mask])).difference((-1,)):
            cl_mask = cable_labels[cable_mask] == cl
            cl_pts = points[cable_mask][cl_mask]
            cable_axis = self._cable_cut(cl_pts)
            cable_axisline = self._linestring_cable_fit(cl_pts, cable_axis)
            if tramtracks_polygon.intersects(cable_axisline.buffer(0.5)):
                cc_height = cl_pts[:, 2] - ground_z[cl_mask]
                if cc_height.min() < self.max_tramcable_height:
                    tram_cable_labels.append(cl)

        logger.debug(f'{len(tram_cable_labels)} tram cables found.')
        return tram_cable_labels

    # ------------------------------------------------------------------ #
    #  Streetlight (armatuur) detection                                    #
    # ------------------------------------------------------------------ #

    def _clip_cable_area(self, points, cable_yline, cable_zline,
                          h_buffer=0.5, w_buffer=0.5):
        cable_zpoly = cable_zline.buffer(h_buffer, cap_style=3)
        height_mask = poly_clip(points[:, [0, 2]], cable_zpoly)
        cable_axispoly = cable_yline.buffer(w_buffer, cap_style=3)
        axis_mask = poly_clip(points[:, [0, 1]], cable_axispoly)
        return height_mask & axis_mask

    def _pc_cable_rotation(self, points, mask):
        direction = main_direction(points[mask][:, :2])
        cable_dir_axis = np.dot(points[:, :2], direction)
        theta = np.arctan2(direction[1], direction[0])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        points_rotated = points.copy()
        points_rotated[:, :2] -= cable_dir_axis[mask].min() * direction
        points_rotated = points_rotated.dot(R)
        return points_rotated

    def _fit_linestring(self, points, bin_width=0.75):
        line_max = points[:, 0].max()
        bins = np.linspace(
            0 - (bin_width / 2),
            line_max + (bin_width / 2),
            int(round(line_max / bin_width) + 2))
        means, bin_edges, _ = binned_statistic(
            points[:, 0], points[:, 1], statistic='mean', bins=bins)
        x_coords = (bin_edges[:-1] + bin_edges[1:]) / 2
        line_pts = np.vstack((x_coords, means)).T
        line_pts = line_pts[~np.isnan(line_pts).any(axis=1)]
        return LineString(line_pts)

    def _compute_sagging_angle(self, x, z, span, d, fill=np.inf):
        d = int(span / d)
        bendings = np.full(len(x), fill)
        for i in range(len(x)):
            if i - d >= 0 and i + d < len(x):
                O = np.array([0, z[i + 1]])
                v_a = np.array([-span, z[i - d]]) - O
                v_b = np.array([span, z[i + d]]) - O
                bendings[i] = 180 - angle_between(v_a, v_b)
        return bendings

    def _search_armaturen(self, points, cable_mask):
        slice_width = 3
        armatuurs_mask = np.zeros(len(points), dtype=bool)

        points_rotated = self._pc_cable_rotation(points, cable_mask)
        cable_yline = self._fit_linestring(
            points_rotated[cable_mask][:, [0, 1]])
        cable_zline = self._fit_linestring(
            points_rotated[cable_mask][:, [0, 2]])

        clip_mask = self._clip_cable_area(
            points_rotated, cable_yline, cable_zline, 1, 1)
        search_mask = clip_mask & ~cable_mask

        logger.debug(f'\t{np.sum(search_mask)} points in neighbourhood of cable')
        if np.sum(search_mask) < 10:
            return armatuurs_mask

        voxel_grid = voxelize(points_rotated[search_mask], 0.05)[0]
        voxel_space = voxel_grid.get_feature_vector()

        min_x = voxel_grid.voxel_centers[0][0]
        max_x = voxel_grid.voxel_centers[-1][0]
        x_ = np.arange(min_x, max_x + 2 * voxel_grid.sizes[0],
                        voxel_grid.sizes[0])
        z_ = np.interp(x_, cable_zline.xy[0], cable_zline.xy[1])
        y_ = np.interp(x_, cable_yline.xy[0], cable_yline.xy[1])
        a_ = self._compute_sagging_angle(
            x_, z_, self.cable_sag_span, voxel_grid.sizes[0])

        attachment_voxel_space = np.zeros(voxel_space.shape).flatten()
        for i in range(0, voxel_space.shape[0], slice_width):
            row_slice = voxel_space[i:i + slice_width].sum(axis=0) > 0
            t = int((z_[i + 1] - voxel_grid.voxel_centers[0][2])
                    / voxel_grid.sizes[0]) + 1
            if np.sum(row_slice[:, :t]) > 5:
                row_slice_closed = np.pad(row_slice, 2)
                row_slice_closed = ndimage.binary_dilation(
                    row_slice_closed, iterations=2)
                row_slice_closed = ndimage.binary_erosion(
                    row_slice_closed, iterations=2)
                row_slice_closed = row_slice_closed[2:-2, 2:-2]

                lcc, n_lcc = ndimage.label(row_slice_closed)
                for l in range(1, n_lcc + 1):
                    cl = np.vstack(np.where(lcc == l)).T
                    if len(cl) > 5:
                        (x_min, y_min, x_max, y_max) = compute_bounding_box(cl)
                        y_center = int(np.round(y_min + (y_max - y_min) / 2))
                        x_center = int(np.round(x_min + (x_max - x_min) / 2))
                        box_width = (x_max - x_min) * voxel_grid.sizes[0]
                        box_height = (y_max - y_min) * voxel_grid.sizes[0]
                        cl_center = voxel_grid.voxel_centers[
                            np.ravel_multi_index(
                                (min(voxel_space.shape[0] - 1, i + 1),
                                 x_center, y_center),
                                voxel_space.shape)]
                        target_center = np.array(
                            [y_[i + 1], z_[i + 1] - (box_height / 2)])
                        z_off = z_[i + 1] - cl_center[2]
                        axis_off = np.abs(target_center[0] - cl_center[1])

                        if (box_width >= self.armatuur_params['width'][0]
                                and box_width < self.armatuur_params['width'][1]
                                and box_height >= self.armatuur_params['height'][0]
                                and box_height < self.armatuur_params['height'][1]
                                and axis_off < self.armatuur_params['axis_offset']
                                and z_off > max(0.1, box_height / 2)
                                and a_[i + 1] > self.min_cable_bending):

                            cl_indices = np.repeat(
                                (lcc == l)[np.newaxis, :, :], 3, axis=0)
                            cl_indices = np.pad(
                                cl_indices, ((min(i, 1), 1), (0, 0), (0, 0)))
                            cl_indices = ndimage.binary_dilation(
                                cl_indices, iterations=1)
                            index_start = np.ravel_multi_index(
                                (max(i - 1, 0), 0, 0), voxel_space.shape)
                            cl_indices = index_start + np.where(
                                cl_indices.flatten())[0]
                            attachment_voxel_space[
                                cl_indices[
                                    cl_indices < len(attachment_voxel_space)
                                ]] = 1

        arm_lcc, arm_n_lcc = ndimage.label(
            attachment_voxel_space.reshape(voxel_space.shape))
        logger.debug(f'\tFound {arm_n_lcc} blobs under cable.')
        for arm_l in range(1, arm_n_lcc + 1):
            arm_idx = np.isin(
                voxel_grid.voxel_n,
                np.where((arm_lcc == arm_l).flatten())[0])
            arm_mask = np.zeros(len(search_mask), dtype=bool)
            arm_mask[np.where(search_mask)[0][arm_idx]] = True

            mbr, _, min_dim, max_dim, center = minimum_bounding_rectangle(
                points[arm_mask, :2])
            if (min_dim > self.armatuur_params['width'][0]
                    and max_dim < self.armatuur_params['width'][1]):
                armatuurs_mask[arm_mask] = True

        return armatuurs_mask

    def _detect_streetlights(self, points, cable_labels):
        streetlights_mask = np.zeros(len(points), dtype=bool)
        for label in set(np.unique(cable_labels)).difference((-1,)):
            cable_mask = cable_labels == label
            if np.sum(cable_mask) > 100:
                logger.debug(
                    f'Looking for streetlights for cable {label} '
                    f'({np.sum(cable_mask)} points).')
                search_mask = self._search_armaturen(points, cable_mask)
                streetlights_mask[search_mask] = True
        return streetlights_mask

    # ------------------------------------------------------------------ #
    #  Pipeline entry-point                                                #
    # ------------------------------------------------------------------ #

    def get_labels(self, points, labels, mask, tilecode):
        """
        Returns the updated labels array with cable, tram-cable and
        armatuur points assigned.
        """
        logger.info(
            f'CableFuser (cable={self.cable_label}, '
            f'streetlight={self.streetlight_label}).')

        label_mask = np.copy(mask)

        label_mask[label_mask] = self._vertical_segmentation(
            points[label_mask], tilecode)
        label_mask[label_mask] = self._bag_removal(
            points[label_mask], tilecode, self.building_offset)

        cable_labels = self._detect_cables(points[label_mask])
        num_cables = np.sum(np.unique(cable_labels) > -1)
        logger.debug(f'Detected {num_cables} cables.')

        if num_cables > 0:
            tram_cable_labels = self._classify_tram_cables(
                points[label_mask], cable_labels, tilecode)
            tramcable_mask = np.isin(cable_labels, tram_cable_labels)
            cable_labels[tramcable_mask] = -1

            labels[np.where(label_mask)[0][cable_labels > -1]] = \
                self.cable_label
            labels[np.where(label_mask)[0][tramcable_mask]] = \
                self.tramcable_label

            num_cables = np.sum(np.unique(cable_labels) > -1)
            if num_cables > 0:
                streetlight_mask = self._detect_streetlights(
                    points[label_mask], cable_labels)
                labels[np.where(label_mask)[0][streetlight_mask]] = \
                    self.streetlight_label

        return labels
