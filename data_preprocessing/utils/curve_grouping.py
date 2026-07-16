import os
import numpy as np
import jsonlines
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, MeanShift

from .preprocessing import cal_line_mask
from .optical_transform import translate_curve


optical_flow_data_map = {'gma':
                            {'optical_flow_dir': '20000_gma-animerun-v2-ft',
                              # 'use_distance_transform': True,
                              'distance_transform_factor': 10},
                         'raft':
                             {'optical_flow_dir': '20000_raft-animerun-v2-ft_again',
                              # 'use_distance_transform': True,
                              'distance_transform_factor': 10},
                         'sain':
                             {'optical_flow_dir': 'SAIN',
                              # 'use_distance_transform': True,
                              'distance_transform_factor': 10},
                         'flowdiffuser':
                             {'optical_flow_dir': 'FlowDiffuser',
                              # 'use_distance_transform': False,
                              'distance_transform_factor': 10}
                             }


def get_curve_depth(stroke_list, depth, raster_size, line_thickness=3):
    '''
    stroke_list: (N', 4, 2)
    depth: (H, W), [0, 255]
    '''
    line_mask = cal_line_mask(stroke_list, raster_size, line_thickness)  # (H, W), [0-BG, 1-stroke]
    masked_depth = depth * line_mask  # (H, W)
    avg_depth = np.sum(masked_depth) / np.sum(line_mask)  # ()
    return avg_depth


def curve_grouping_depth(data_base, stroke_data_b_raw, img_index):
    depth_base = os.path.join(data_base, 'depth', 'params')
    depth_img_path = os.path.join(depth_base, str(img_index) + '_ref.png')
    assert os.path.exists(depth_img_path)

    depth_img = Image.open(depth_img_path).convert('RGB')
    image_size = depth_img.height
    depth = np.array(depth_img, dtype=np.int32)[:, :, 0]

    curve_depths = []
    for curve_i in range(len(stroke_data_b_raw)):
        stroke_list = stroke_data_b_raw[curve_i]  # (N', 4, 2)
        curve_depth = get_curve_depth(stroke_list, depth, image_size)
        print(curve_i, ':', curve_depth)
        curve_depths.append(curve_depth)

    plt.scatter(curve_depths, np.zeros_like(curve_depths), c='red')
    plt.show()


def kmeans_grouping(data, verbose=False):
    kmeans = KMeans()
    kmeans.fit(data)

    labels = kmeans.labels_

    if verbose:
        # centroids = kmeans.cluster_centers_
        # print('labels', labels)
        # print('centroids', centroids)

        plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
        # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, alpha=0.5)
        plt.title('KMeans Clustering')
        plt.xlabel('Data')
        plt.ylabel('Cluster')
        plt.show()

    return labels


def DBSCAN_grouping(data, verbose=False):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(data)

    labels = dbscan.labels_
    if verbose:
        # print('labels', labels)

        plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
        plt.title('DBSCAN Clustering')
        plt.xlabel('Data')
        plt.ylabel('Cluster')
        plt.show()

    return labels


def MeanShift_grouping(data, verbose=False):
    mean_shift = MeanShift()
    mean_shift.fit(data)

    labels = mean_shift.labels_

    if verbose:
        # centroids = mean_shift.cluster_centers_
        # print('labels', labels)
        # print('centroids', centroids)

        plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
        # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, alpha=0.5)
        plt.title('KMeans Clustering')
        plt.xlabel('Data')
        plt.ylabel('Cluster')
        plt.show()

    return labels


def check_is_with_single_curve(group_labels):
    '''
    group_labels: (N)
    '''
    max_group_num = np.max(group_labels) + 1
    label_num = [np.sum(group_labels == g_i) for g_i in range(max_group_num)]

    single_curve_group_ids = np.argwhere(np.array(label_num) == 1).squeeze(axis=-1)
    # print(single_curve_group_ids.shape)
    if len(single_curve_group_ids) > 0:
        single_curve_group_id = single_curve_group_ids[0]
        single_curve_group_id_idx = group_labels.tolist().index(single_curve_group_id)
        return True, single_curve_group_id, single_curve_group_id_idx
    else:
        return False, None, None


def cal_to_other_group_min_distance(group_labels, curve_offsets, single_curve_group_id, single_curve_group_id_idx):
    ''' A is in the group.
    curve_offsets: (N, 2)
    '''
    single_curve_group_offset = np.expand_dims(curve_offsets[single_curve_group_id_idx], axis=0)  # (1, 2)
    max_group_num = np.max(group_labels) + 1
    to_other_groups_distance = []
    for g_i in range(max_group_num):
        if g_i == single_curve_group_id:
            to_other_groups_distance.append(-1)
            continue

        group_i_offsets = curve_offsets[group_labels == g_i]  # (N', 2)
        to_group_i_distance = np.min(np.sqrt(np.sum(np.power(group_i_offsets - single_curve_group_offset, 2), axis=-1)))
        to_other_groups_distance.append(to_group_i_distance)

    to_other_groups_distance = np.array(to_other_groups_distance, dtype=np.float32)
    to_other_groups_distance[single_curve_group_id] = np.max(to_other_groups_distance) + 1.0
    return to_other_groups_distance


def cal_to_other_group_min_distance_outer(single_curve_offset, comp_curves_offset):
    ''' A is outside the group.
    Args:
        single_curve_offset: 2
        comp_curves_offset: len = N_comp, list of (N', 2)
    '''
    single_curve_offset = np.expand_dims(single_curve_offset, axis=0)  # (1, 2)
    to_other_groups_distance = []
    for g_i in range(len(comp_curves_offset)):
        group_i_offsets = np.array(comp_curves_offset[g_i], dtype=np.float32)  # (N', 2)
        to_group_i_distance = np.min(np.sqrt(np.sum(np.power(group_i_offsets - single_curve_offset, 2), axis=-1)))
        to_other_groups_distance.append(to_group_i_distance)

    to_other_groups_distance = np.array(to_other_groups_distance, dtype=np.float32)
    return to_other_groups_distance


def renew_group_labels(group_labels):
    '''
    group_labels: (N)
    '''
    group_labels_new = np.ones_like(group_labels)
    group_labels_new.fill(-1)
    max_group_num = np.max(group_labels) + 1

    curr_group_num = 0
    for g_i in range(max_group_num):
        if (group_labels == g_i).any():
            group_labels_new[group_labels == g_i] = curr_group_num
            curr_group_num += 1
    assert (group_labels_new != -1).all()
    return group_labels_new


def merge_single_curve(group_labels, curve_offsets, verbose=False):
    '''
    group_labels: (N)
    curve_offsets: (N, 2)
    '''
    group_labels_new = np.array(group_labels, np.int32)
    while True:
        is_with_single_curve, single_curve_group_id, single_curve_group_id_idx = \
            check_is_with_single_curve(group_labels_new)
        if not is_with_single_curve:
            break

        to_other_groups_distance = cal_to_other_group_min_distance(group_labels_new, curve_offsets, single_curve_group_id,
                                                                   single_curve_group_id_idx)  # (max_group_num)
        merge_to_group_id = np.argmin(to_other_groups_distance)
        group_labels_new[single_curve_group_id_idx] = merge_to_group_id
        group_labels_new = renew_group_labels(group_labels_new)

    group_labels_new = group_labels_new.tolist()
    if verbose:
        # print('merge_single_curve: labels', group_labels_new)
        # print('centroids', centroids)

        plt.scatter(curve_offsets[:, 0], curve_offsets[:, 1], c=group_labels_new, s=50, cmap='viridis')
        # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, alpha=0.5)
        plt.title('KMeans Clustering')
        plt.xlabel('Data')
        plt.ylabel('Cluster')
        plt.show()

    return group_labels_new


def get_optical_flow_dir_name(optical_flow_method, use_distance_transform):
    optical_flow_dir = optical_flow_data_map[optical_flow_method]['optical_flow_dir']
    # use_distance_transform = optical_flow_data_map[optical_flow_method]['use_distance_transform']
    distance_transform_factor = optical_flow_data_map[optical_flow_method]['distance_transform_factor']
    if use_distance_transform:
        optical_flow_dir += '-[DT-' + str(distance_transform_factor) + ']'
    return optical_flow_dir


def curve_grouping_flow(data_base, stroke_data_b_raw, image_size, img_index, optical_flow_method, use_distance_transform, should_merge_single_curve,
                        verbose=False, clip=None):
    optical_flow_dir = get_optical_flow_dir_name(optical_flow_method, use_distance_transform)
    optical_flow_base = os.path.join(data_base, 'optical_flow', optical_flow_dir, 'flow')
    if clip is not None:
        optical_flow_base = os.path.join(optical_flow_base, clip)
    flow_path = os.path.join(optical_flow_base, 'flow-' + str(img_index) + '.npz')
    npz = np.load(flow_path, encoding='latin1', allow_pickle=True)
    flow = npz['flow_mat']  # (H, W, 2)

    curve_offsets = []
    for curve_i in range(len(stroke_data_b_raw)):
        stroke_list = stroke_data_b_raw[curve_i]  # (N', 4, 2)
        _, curve_offset = translate_curve(stroke_list, flow, image_size)
        # curve_offset: (2), [dx, dy], in image size
        curve_offsets.append(curve_offset)
        # print(curve_i, ':', curve_offset)

    curve_offsets = np.stack(curve_offsets, axis=0)  # (N', 2)
    ## visualization
    if verbose:
        plt.scatter(curve_offsets[:, 0], curve_offsets[:, 1], c='red')
        plt.show()

    ## Grouping
    # group_labels = kmeans_grouping(curve_offsets, verbose=verbose)
    # group_labels = DBSCAN_grouping(curve_offsets, verbose=verbose)
    group_labels = MeanShift_grouping(curve_offsets, verbose=verbose)
    assert np.min(group_labels) == 0
    assert len(group_labels) == len(stroke_data_b_raw)

    ## Merge groups with single curves
    if should_merge_single_curve:
        group_labels_m = merge_single_curve(group_labels, curve_offsets, verbose=verbose)
    else:
        group_labels_m = group_labels.tolist()

    return group_labels_m, optical_flow_dir


def curve_grouping_flow_inv(data_base, stroke_data_b_tar0, stroke_data_b_raw, curves_endpoint_connected_state,
                            image_size, img_index, optical_flow_method, use_distance_transform,
                            clip=None):
    optical_flow_dir = get_optical_flow_dir_name(optical_flow_method, use_distance_transform)
    optical_flow_base = os.path.join(data_base, 'optical_flow', optical_flow_dir, 'flow')
    if clip is not None:
        optical_flow_base = os.path.join(optical_flow_base, clip)
    flow_path = os.path.join(optical_flow_base, 'flow-' + str(img_index) + '.npz')
    npz = np.load(flow_path, encoding='latin1', allow_pickle=True)
    flow = npz['flow_mat']  # (H, W, 2)

    ## Calculate curves' offset for each component in predicted target strokes (tar0)
    comp_curves_offset_tar0 = []  # len = N_comp, list of (N', 2)
    for c_i in range(len(stroke_data_b_tar0)):
        curve_b_list = stroke_data_b_tar0[c_i]  # list of (N', 4, 2)
        curves_offset = []  # list of (2), [dx, dy]
        for curve_i in range(len(curve_b_list)):
            curve = curve_b_list[curve_i]  # list (N') of (4, 2)
            _, curve_offset = translate_curve(curve, flow, image_size)
            # curve_offset: (2), [dx, dy], in image size
            curves_offset.append(curve_offset)
        comp_curves_offset_tar0.append(curves_offset)

    ## Calculate curves' group (tar1)
    group_labels = []  # len = N_curves, int
    for curve_i in range(len(stroke_data_b_raw)):
        stroke_list = stroke_data_b_raw[curve_i]  # (N', 4, 2)
        _, curve_offset = translate_curve(stroke_list, flow, image_size)
        # curve_offset: (2), [dx, dy], in image size

        curve_endpoint_connected_state = curves_endpoint_connected_state[curve_i]  # ['0_1_2', None]

        if curve_endpoint_connected_state[0] is None and curve_endpoint_connected_state[1] is None:  # 0 connected endpoint
            to_other_groups_distance = cal_to_other_group_min_distance_outer(curve_offset, comp_curves_offset_tar0)  # (N_layer)
            min_dist_group = np.argmin(to_other_groups_distance)
            group_labels.append(int(min_dist_group))
        elif curve_endpoint_connected_state[0] is not None and curve_endpoint_connected_state[1] is not None:  # 2 connected endpoint
            conn_comp0 = curve_endpoint_connected_state[0].split('_')[0]
            conn_comp1 = curve_endpoint_connected_state[1].split('_')[0]
            assert conn_comp0 == conn_comp1
            group_labels.append(int(conn_comp0))
        else:  # 1 connected endpoint
            if curve_endpoint_connected_state[0] is not None:
                conn_comp = curve_endpoint_connected_state[0].split('_')[0]
            else:
                conn_comp = curve_endpoint_connected_state[1].split('_')[0]
            group_labels.append(int(conn_comp))

    return group_labels, optical_flow_dir
