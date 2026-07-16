import numpy as np
import random
import math

from .preprocessing import deep_shape_check


def affine_trans(points, trans_matrix):
    new_points = []

    for point in points:
        point_ = np.stack([point[0], point[1], 1.0])
        new_point = np.matmul(trans_matrix, point_)[0:2]
        new_points.append(new_point)

    return new_points


def translation(points, offset):
    trans_matrix = np.array([[1, 0, offset[0]],
                             [0, 1, offset[1]],
                             [0, 0, 1]], dtype=np.float32)
    points_trans = affine_trans(points, trans_matrix)
    return points_trans


def translation_global(stroke_data, stroke_data_b, translation_thresholds):
    '''
    stroke_data: component list => curve list => stroke list (N, 2)
    stroke_data_b: component list => curve list => stroke list (N', 4, 2)
    '''
    trans_stroke_data = []
    trans_stroke_data_b = []

    trans_x = random.random() * (translation_thresholds[1] - translation_thresholds[0]) + translation_thresholds[0]
    trans_y = random.random() * (translation_thresholds[1] - translation_thresholds[0]) + translation_thresholds[0]

    for c_i in range(len(stroke_data)):
        curve_list = stroke_data[c_i]  # list of (N, 2)
        curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
        trans_curve_list = []
        trans_curve_b_list = []
        for points in curve_list:
            trans_points = translation(points, (trans_x, trans_y))
            trans_curve_list.append(np.stack(trans_points, axis=0).astype(np.float32).tolist())
        trans_stroke_data.append(trans_curve_list)
        for points in curve_b_list:  # (N', 4, 2)
            points_flatten = np.reshape(points, (-1, 2))  # (N'*4, 2)
            trans_points_flatten = translation(points_flatten, (trans_x, trans_y))
            trans_points_flatten = np.stack(trans_points_flatten, axis=0).astype(np.float32)  # (N'*4, 2)
            trans_curve_b_list.append(np.reshape(trans_points_flatten, (-1, 4, 2)).tolist())
        trans_stroke_data_b.append(trans_curve_b_list)

    deep_shape_check(stroke_data, trans_stroke_data)
    deep_shape_check(stroke_data_b, trans_stroke_data_b)
    return trans_stroke_data, trans_stroke_data_b


def rotation(points, angle_, center):
    angle = angle_ / 180.0 * math.pi

    translate_matrix = np.array([[1, 0, -center[0]],
                                 [0, 1, -center[1]],
                                 [0, 0, 1]], dtype=np.float32)

    rot_matrix = np.array([[math.cos(angle), math.sin(angle), 0],
                           [-math.sin(angle), math.cos(angle), 0],
                           [0, 0, 1]], dtype=np.float32)

    translate_reverse_matrix = np.array([[1, 0, center[0]],
                                         [0, 1, center[1]],
                                         [0, 0, 1]], dtype=np.float32)

    combined_matrix = np.matmul(translate_reverse_matrix, np.matmul(rot_matrix, translate_matrix))
    points_rotation = affine_trans(points, combined_matrix)

    return points_rotation


def rotation_global(stroke_data, stroke_data_b, angle_thresholds):
    '''
    stroke_data: component list => curve list => stroke list (N, 2)
    stroke_data_b: component list => curve list => stroke list (N', 4, 2)
    '''
    trans_stroke_data = []
    trans_stroke_data_b = []

    angle = random.random() * (angle_thresholds[1] - angle_thresholds[0]) + angle_thresholds[0]
    stroke_data_flatten = []
    for item in stroke_data:
        stroke_data_flatten += item
    all_points = np.concatenate(stroke_data_flatten, axis=0)  # (N', 2)
    center_index = random.randint(0, all_points.shape[0] - 1)
    center = all_points[center_index]

    for c_i in range(len(stroke_data)):
        curve_list = stroke_data[c_i]  # list of (N, 2)
        curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
        trans_curve_list = []
        trans_curve_b_list = []
        for points in curve_list:
            trans_points = rotation(points, angle, center=center)
            trans_curve_list.append(np.stack(trans_points, axis=0).astype(np.float32).tolist())
        trans_stroke_data.append(trans_curve_list)
        for points in curve_b_list:  # (N', 4, 2)
            points_flatten = np.reshape(points, (-1, 2))  # (N'*4, 2)
            trans_points_flatten = rotation(points_flatten, angle, center=center)
            trans_points_flatten = np.stack(trans_points_flatten, axis=0).astype(np.float32)  # (N'*4, 2)
            trans_curve_b_list.append(np.reshape(trans_points_flatten, (-1, 4, 2)).tolist())
        trans_stroke_data_b.append(trans_curve_b_list)

    deep_shape_check(stroke_data, trans_stroke_data)
    deep_shape_check(stroke_data_b, trans_stroke_data_b)
    return trans_stroke_data, trans_stroke_data_b


def scaling(points, scales, center):
    translate_matrix = np.array([[1, 0, -center[0]],
                                 [0, 1, -center[1]],
                                 [0, 0, 1]], dtype=np.float32)

    scale_matrix = np.array([[scales[0], 0, 0],
                             [0, scales[1], 0],
                             [0, 0, 1]], dtype=np.float32)

    translate_reverse_matrix = np.array([[1, 0, center[0]],
                                         [0, 1, center[1]],
                                         [0, 0, 1]], dtype=np.float32)

    combined_matrix = np.matmul(translate_reverse_matrix, np.matmul(scale_matrix, translate_matrix))
    points_scaling = affine_trans(points, combined_matrix)

    return points_scaling


def scaling_global(stroke_data, stroke_data_b, scale_thresholds):
    '''
    stroke_data: component list => curve list => stroke list (N, 2)
    stroke_data_b: component list => curve list => stroke list (N', 4, 2)
    '''
    trans_stroke_data = []
    trans_stroke_data_b = []

    scale_x = random.random() * (scale_thresholds[1] - scale_thresholds[0]) + scale_thresholds[0]
    scale_y = random.random() * (scale_thresholds[1] - scale_thresholds[0]) + scale_thresholds[0]

    stroke_data_flatten = []
    for item in stroke_data:
        stroke_data_flatten += item
    all_points = np.concatenate(stroke_data_flatten, axis=0)  # (N', 2)
    center_index = random.randint(0, all_points.shape[0] - 1)
    center = all_points[center_index]

    for c_i in range(len(stroke_data)):
        curve_list = stroke_data[c_i]  # list of (N, 2)
        curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
        trans_curve_list = []
        trans_curve_b_list = []
        for points in curve_list:
            trans_points = scaling(points, (scale_x, scale_y), center=center)
            trans_curve_list.append(np.stack(trans_points, axis=0).astype(np.float32).tolist())
        trans_stroke_data.append(trans_curve_list)
        for points in curve_b_list:  # (N', 4, 2)
            points_flatten = np.reshape(points, (-1, 2))  # (N'*4, 2)
            trans_points_flatten = scaling(points_flatten, (scale_x, scale_y), center=center)
            trans_points_flatten = np.stack(trans_points_flatten, axis=0).astype(np.float32)  # (N'*4, 2)
            trans_curve_b_list.append(np.reshape(trans_points_flatten, (-1, 4, 2)).tolist())
        trans_stroke_data_b.append(trans_curve_b_list)

    deep_shape_check(stroke_data, trans_stroke_data)
    deep_shape_check(stroke_data_b, trans_stroke_data_b)
    return trans_stroke_data, trans_stroke_data_b


def shearing(points, angles, center):
    angle_x = angles[0] / 180.0 * math.pi
    angle_y = angles[1] / 180.0 * math.pi

    translate_matrix = np.array([[1, 0, -center[0]],
                                 [0, 1, -center[1]],
                                 [0, 0, 1]], dtype=np.float32)

    shear_matrix = np.array([[1, math.tan(angle_x), 0],
                             [math.tan(angle_y), 1, 0],
                             [0, 0, 1]], dtype=np.float32)

    translate_reverse_matrix = np.array([[1, 0, center[0]],
                                         [0, 1, center[1]],
                                         [0, 0, 1]], dtype=np.float32)

    combined_matrix = np.matmul(translate_reverse_matrix, np.matmul(shear_matrix, translate_matrix))
    points_shear = affine_trans(points, combined_matrix)

    return points_shear


def shearing_global(stroke_data, stroke_data_b, angle_thresholds):
    '''
    stroke_data: component list => curve list => stroke list (N, 2)
    stroke_data_b: component list => curve list => stroke list (N', 4, 2)
    '''
    trans_stroke_data = []
    trans_stroke_data_b = []

    angle = random.random() * (angle_thresholds[1] - angle_thresholds[0]) + angle_thresholds[0]
    angles = (angle, 0.0) if random.randint(0, 1) else (0.0, angle)

    stroke_data_flatten = []
    for item in stroke_data:
        stroke_data_flatten += item
    all_points = np.concatenate(stroke_data_flatten, axis=0)  # (N', 2)
    center_index = random.randint(0, all_points.shape[0] - 1)
    center = all_points[center_index]

    for c_i in range(len(stroke_data)):
        curve_list = stroke_data[c_i]  # list of (N, 2)
        curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
        trans_curve_list = []
        trans_curve_b_list = []
        for points in curve_list:
            trans_points = shearing(points, angles, center=center)
            trans_curve_list.append(np.stack(trans_points, axis=0).astype(np.float32).tolist())
        trans_stroke_data.append(trans_curve_list)
        for points in curve_b_list:  # (N', 4, 2)
            points_flatten = np.reshape(points, (-1, 2))  # (N'*4, 2)
            trans_points_flatten = shearing(points_flatten, angles, center=center)
            trans_points_flatten = np.stack(trans_points_flatten, axis=0).astype(np.float32)  # (N'*4, 2)
            trans_curve_b_list.append(np.reshape(trans_points_flatten, (-1, 4, 2)).tolist())
        trans_stroke_data_b.append(trans_curve_b_list)

    deep_shape_check(stroke_data, trans_stroke_data)
    deep_shape_check(stroke_data_b, trans_stroke_data_b)
    return trans_stroke_data, trans_stroke_data_b


def should_do(prob):
    return random.random() <= prob


def choose_which_index(prob_list):
    prob_list_cumsum = np.cumsum(prob_list) / np.sum(prob_list)
    select_prob = random.random()
    for i in range(len(prob_list_cumsum)):
        if select_prob <= prob_list_cumsum[i]:
            return i


def sketch_global_deformation(stroke_data, stroke_data_b):
    '''
    stroke_data: component list => curve list => stroke list (N, 2)
    stroke_data_b: component list => curve list => stroke list (N', 4, 2)
    '''
    ## Hyper-parameters
    global_deform_prob = 0.95
    global_n_way_deform_probs = [0.4, 0.35, 0.25]
    global_deform_types = ['rotation', 'scaling', 'shearing']

    global_rotation_angle_thresholds = [-15.0, 15.0]
    global_scaling_thresholds = [0.9, 1.1]
    global_shearing_thresholds = [-15.0, 15.0]

    if should_do(global_deform_prob):
        n_way_deform = choose_which_index(global_n_way_deform_probs) + 1  # {1, 2, 3}
        random.shuffle(global_deform_types)
        selected_deform_types = global_deform_types[:n_way_deform]

        trans_stroke_data = [item for item in stroke_data]
        trans_stroke_data_b = [item for item in stroke_data_b]
        for selected_deform_type in selected_deform_types:
            if selected_deform_type == 'rotation':
                trans_stroke_data, trans_stroke_data_b = rotation_global(trans_stroke_data, trans_stroke_data_b,
                                                                         global_rotation_angle_thresholds)

            if selected_deform_type == 'scaling':
                trans_stroke_data, trans_stroke_data_b = scaling_global(trans_stroke_data, trans_stroke_data_b,
                                                                        global_scaling_thresholds)

            if selected_deform_type == 'shearing':
                trans_stroke_data, trans_stroke_data_b = shearing_global(trans_stroke_data, trans_stroke_data_b,
                                                                         global_shearing_thresholds)

        return trans_stroke_data, trans_stroke_data_b
    else:
        return stroke_data, stroke_data_b


def find_top_k(num_list, top_k):
    sorted_list = np.argsort(num_list)
    tops = np.zeros_like(sorted_list)
    for i in range(len(sorted_list)):
        tops[sorted_list[i]] = i
    tops += 1
    return tops > len(tops) - top_k


def dfs(adj_mat, curr_index, done_mat):
    done_mat[curr_index] = 1
    adj_indices = np.argwhere(adj_mat[curr_index] == 1)
    adj_indices_return = [curr_index]
    for adj_index in adj_indices:
        if not done_mat[adj_index[0]]:
            result = dfs(adj_mat, adj_index[0], done_mat)
            adj_indices_return += result[0]
            done_mat = result[1]
    return adj_indices_return, done_mat


def indices_clustering2(indices):
    indices_np = np.stack(indices)
    indices_np_rev = np.concatenate([indices_np[:, 1:2], indices_np[:, 0:1]], axis=-1)
    indices_np = np.concatenate([indices_np, indices_np_rev], axis=0)

    max_index = np.max(indices_np)
    adj_mat = np.zeros(shape=(max_index + 1, max_index + 1), dtype=np.float32)
    done_mat = np.zeros(shape=(max_index + 1), dtype=np.float32)
    for pos in indices_np:
        adj_mat[tuple(pos)] = 1

    cluster_list = []
    for i in range(max_index + 1):
        if not done_mat[i]:
            indices_return, done_mat = dfs(adj_mat, i, done_mat)
            indices_return.sort()
            cluster_list.append(indices_return)

    return cluster_list


def cal_curve_curve_distance(endpoints_list):
    '''
    endpoints_list: (N, 4)
    '''
    start_points_list = endpoints_list[:, 0:2]  # (N, 2)
    end_points_list = endpoints_list[:, 2:4]  # (N, 2)

    start_start_dist = np.expand_dims(start_points_list, axis=1) - np.expand_dims(start_points_list,
                                                                                  axis=0)  # (N, N, 2)
    start_start_dist = np.sqrt(np.sum(np.power(start_start_dist, 2), axis=-1))  # (N, N)

    end_end_dist = np.expand_dims(end_points_list, axis=1) - np.expand_dims(end_points_list, axis=0)  # (N, N, 2)
    end_end_dist = np.sqrt(np.sum(np.power(end_end_dist, 2), axis=-1))  # (N, N)

    start_end_dist = np.expand_dims(start_points_list, axis=1) - np.expand_dims(end_points_list, axis=0)  # (N, N, 2)
    start_end_dist = np.sqrt(np.sum(np.power(start_end_dist, 2), axis=-1))  # (N, N)
    start_end_dist = start_end_dist * (1.0 - np.eye(start_end_dist.shape[0]))  # (N, N)

    dist_mat = np.stack([start_start_dist, end_end_dist, start_end_dist], axis=-1)  # (N, N, 3)
    dist_mat = np.min(dist_mat, axis=-1)  # (N, N)
    return dist_mat


def gen_curve_group_mat(curve_group_map):
    '''
    curve_group_map: [0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7]
    '''
    num_curve = len(curve_group_map)
    curve_group_mat = np.ones(shape=(num_curve, num_curve), dtype=np.float32)
    num_group = np.max(curve_group_map) + 1
    for g_i in range(num_group):
        curve_indices = np.argwhere(np.array(curve_group_map) == g_i).squeeze(axis=-1)
        curve_indices_tile = np.expand_dims(curve_indices, axis=-1)
        curve_indices_tile = np.tile(curve_indices_tile, (1, len(curve_indices))).reshape(-1)
        curve_indices_tile2 = np.expand_dims(curve_indices, axis=0)
        curve_indices_tile2 = np.tile(curve_indices_tile2, (1, len(curve_indices))).reshape(-1)
        pair_index = (curve_indices_tile, curve_indices_tile2)  # (np.array([1, 1, 2, 2]), np.array([1, 2, 1, 2]))
        curve_group_mat[pair_index] = 0
    return curve_group_mat


def connect_close_components(stroke_data_b, connect_threshold):
    '''
    stroke_data_b: component list => curve list => stroke list (N', 4, 2)
    '''
    endpoints_list = []
    curve_component_map = []  # [0, 0, 1, 2, 2, 2, 3, 4, ...]
    for c_i in range(len(stroke_data_b)):
        curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
        for curve in curve_b_list:  # (N', 4, 2)
            endpoints_list.append(np.concatenate([curve[0][0], curve[-1][-1]]))
            curve_component_map.append(c_i)
    endpoints_list = np.stack(endpoints_list, axis=0)  # (N^, 4)
    assert len(endpoints_list) == len(curve_component_map)
    # print('>> curve_component_map', curve_component_map)

    dist_mat = cal_curve_curve_distance(endpoints_list)
    # dist_mat = dist_mat + np.eye(dist_mat.shape[0]) * (connect_threshold + 1)

    assert dist_mat.shape[0] == dist_mat.shape[1] == len(curve_component_map)
    component_curve_mat = gen_curve_group_mat(curve_component_map)  # (N^, N^)
    dist_mat *= component_curve_mat

    connect_indices = np.argwhere(dist_mat <= connect_threshold)  # list of (2)
    cluster_indices_curve = indices_clustering2(connect_indices)  # list of indices_list, [[0, 1, 3], [2], [4, 5], ...]
    # print('>> cluster_indices_curve', cluster_indices_curve)

    ## curve index to component index
    cluster_indices_component = []  # list of indices_list, [[0, 1, 3], [2], [4, 5], ...]
    for item in cluster_indices_curve:
        cluster_indices_component_single = []
        for curve_idx in item:
            cluster_indices_component_single.append(curve_component_map[curve_idx])
        cluster_indices_component_single = list(set(cluster_indices_component_single))
        cluster_indices_component.append(cluster_indices_component_single)
    # print('> cluster_indices_component', cluster_indices_component)

    cluster_indices_component_ = np.concatenate(cluster_indices_component).tolist()
    assert len(cluster_indices_component_) == len(set(cluster_indices_component_))

    return cluster_indices_component


def sketch_component_deformation(stroke_data, stroke_data_b):
    '''
    stroke_data: component list => curve list => stroke list (N, 2)
    stroke_data_b: component list => curve list => stroke list (N', 4, 2)
    '''
    ## Hyper-parameters
    connect_distance_threshold = 6

    top_k_for_special = 1

    local_deform_prob_map = {'normal': 1.0, 'special': 0.8}
    local_n_way_deform_probs = [0.2, 0.3, 0.3, 0.2]
    local_deform_types = ['translation', 'rotation', 'scaling', 'shearing']

    local_translation_thresholds_map = {'normal': [-40.0, 40.0], 'special': [-10.0, 10.0]}
    local_rotation_angle_thresholds_map = {'normal': [-20.0, 20.0], 'special': [-10.0, 10.0]}
    local_scaling_thresholds_map = {'normal': [0.8, 1.4], 'special': [0.8, 1.2]}
    local_shearing_thresholds_map = {'normal': [-15.0, 15.0], 'special': [-10.0, 10.0]}

    trans_stroke_data = [[] for _ in range(len(stroke_data))]
    trans_stroke_data_b = [[] for _ in range(len(stroke_data_b))]

    # generate component clusters first
    cluster_indices_component = connect_close_components(stroke_data_b, connect_distance_threshold)  # [[0, 1, 3], [2], [4, 5], ...]

    component_cluster_lens = []
    for cluster_i in range(len(cluster_indices_component)):
        cluster_component_ids = cluster_indices_component[cluster_i]
        component_cluster_len = 0
        for c_i in cluster_component_ids:
            curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
            component_len = np.sum([len(curve) for curve in curve_b_list])
            component_cluster_len += component_len
        component_cluster_lens.append(component_cluster_len)
    is_top_k = find_top_k(component_cluster_lens, top_k_for_special)

    # deform each cluster
    for cluster_i in range(len(cluster_indices_component)):
        cluster_component_ids = cluster_indices_component[cluster_i]  # [0, 1, 3] / [2]
        cluster_stroke_data = []  # component list => curve list => stroke list (N, 2)
        cluster_stroke_data_b = []  # component list => curve list => stroke list (N', 4, 2)
        for s_i in cluster_component_ids:
            cluster_stroke_data.append(stroke_data[s_i])
            cluster_stroke_data_b.append(stroke_data_b[s_i])

        component_type = 'special' if is_top_k[cluster_i] else 'normal'

        local_deform_prob = local_deform_prob_map[component_type]
        local_translation_thresholds = local_translation_thresholds_map[component_type]
        local_rotation_angle_thresholds = local_rotation_angle_thresholds_map[component_type]
        local_scaling_thresholds = local_scaling_thresholds_map[component_type]
        local_shearing_thresholds = local_shearing_thresholds_map[component_type]

        if should_do(local_deform_prob):
            n_way_deform = choose_which_index(local_n_way_deform_probs) + 1  # {1, 2, 3}
            random.shuffle(local_deform_types)
            selected_deform_types = local_deform_types[:n_way_deform]

            trans_cluster_stroke_data = [item for item in cluster_stroke_data]
            trans_cluster_stroke_data_b = [item for item in cluster_stroke_data_b]

            for selected_deform_type in selected_deform_types:
                if selected_deform_type == 'translation':
                    trans_cluster_stroke_data, trans_cluster_stroke_data_b = translation_global(trans_cluster_stroke_data,
                                                                                                trans_cluster_stroke_data_b,
                                                                                                local_translation_thresholds)

                if selected_deform_type == 'rotation':
                    trans_cluster_stroke_data, trans_cluster_stroke_data_b = rotation_global(trans_cluster_stroke_data,
                                                                                             trans_cluster_stroke_data_b,
                                                                                             local_rotation_angle_thresholds)

                if selected_deform_type == 'scaling':
                    trans_cluster_stroke_data, trans_cluster_stroke_data_b = scaling_global(trans_cluster_stroke_data,
                                                                                            trans_cluster_stroke_data_b,
                                                                                            local_scaling_thresholds)

                if selected_deform_type == 'shearing':
                    trans_cluster_stroke_data, trans_cluster_stroke_data_b = shearing_global(trans_cluster_stroke_data,
                                                                                             trans_cluster_stroke_data_b,
                                                                                             local_shearing_thresholds)

            for s_i, c_i in enumerate(cluster_component_ids):
                trans_stroke_data[c_i] += trans_cluster_stroke_data[s_i]
                trans_stroke_data_b[c_i] += trans_cluster_stroke_data_b[s_i]
        else:
            for s_i, c_i in enumerate(cluster_component_ids):
                trans_stroke_data[c_i] += cluster_stroke_data[s_i]
                trans_stroke_data_b[c_i] += cluster_stroke_data_b[s_i]

    return trans_stroke_data, trans_stroke_data_b, cluster_indices_component


def connect_close_curves(stroke_data_b, stroke_data_curve_connect_status, connect_threshold):
    '''
    stroke_data_b: component list => curve list => stroke list (N', 4, 2)
    stroke_data_curve_connect_status: component list => new curve connection status (K'), [0, 1, 1, 2, 2, 3, 4]
    '''
    endpoints_list = []
    curve_component_map = []  # [0, 0, 1, 2, 2, 2, 3, 4, ...]
    curve_inner_curve_map = []  # [0, 1, 0, 0, 1, 2, 0, 0, ...]
    curve_connect_status_flatten = []
    for c_i in range(len(stroke_data_b)):
        curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
        curve_connect_status = stroke_data_curve_connect_status[c_i]  # (K'), [0, 1, 1, 2, 2, 3, 4]
        if len(curve_connect_status_flatten) == 0:
            curve_connect_status_flatten += curve_connect_status
        else:
            curve_connect_status_flatten += (np.array(curve_connect_status, np.int32) + np.max(curve_connect_status_flatten) + 1).tolist()
        for curve_j, curve in enumerate(curve_b_list):  # (N', 4, 2)
            endpoints_list.append(np.concatenate([curve[0][0], curve[-1][-1]]))
            curve_component_map.append(c_i)
            curve_inner_curve_map.append(curve_j)
    endpoints_list = np.stack(endpoints_list, axis=0)  # (N^, 4)
    assert len(endpoints_list) == len(curve_component_map) == len(curve_connect_status_flatten) == len(curve_inner_curve_map)

    dist_mat = cal_curve_curve_distance(endpoints_list)
    # dist_mat = dist_mat + np.eye(dist_mat.shape[0]) * (connect_threshold + 1)

    assert dist_mat.shape[0] == dist_mat.shape[1] == len(curve_component_map)
    group_curve_mat = gen_curve_group_mat(curve_connect_status_flatten)  # (N^, N^)
    dist_mat *= group_curve_mat

    connect_indices = np.argwhere(dist_mat <= connect_threshold)  # list of (2)
    cluster_indices_curve = indices_clustering2(connect_indices)  # list of indices_list, [[0, 1, 3], [2], [4, 5], ...]
    # print('>> cluster_indices_curve', cluster_indices_curve)
    # print('>> curve_component_map', curve_component_map)
    # print('>> curve_inner_curve_map', curve_inner_curve_map)

    return cluster_indices_curve, curve_component_map, curve_inner_curve_map


def sketch_curve_deformation(stroke_data, stroke_data_b, stroke_data_curve_connect_status, cluster_indices_component,
                             parts_data):
    '''
    stroke_data: component list => curve list => stroke list (N, 2)
    stroke_data_b: component list => curve list => stroke list (N', 4, 2)
    stroke_data_curve_connect_status: component list => new curve connection status (K'), [0, 1, 1, 2, 2, 3, 4]
    cluster_indices_component: [[0, 1, 3], [2], [4, 5], ...]
    '''
    ## Hyper-parameters
    connect_distance_threshold = 10

    top_k_for_special = 1

    local_deform_prob_map = {'normal': 0.8, 'special': 0.3}
    local_n_way_deform_probs = [0.3, 0.3, 0.2, 0.2]
    local_deform_types = ['translation', 'rotation', 'scaling', 'shearing']

    local_translation_thresholds_map = {'normal': [-30.0, 30.0], 'special': [-5.0, 5.0]}
    local_rotation_angle_thresholds_map = {'normal': [-15.0, 15.0], 'special': [-5.0, 5.0]}
    local_scaling_thresholds_map = {'normal': [0.8, 1.2], 'special': [0.9, 1.1]}
    local_shearing_thresholds_map = {'normal': [-15.0, 15.0], 'special': [-5.0, 5.0]}

    ONLY_ONE_COMPONENTS = ['eye', 'beak', 'mouth', 'nose']

    trans_stroke_data = [[] for _ in range(len(stroke_data))]
    trans_stroke_data_b = [[] for _ in range(len(stroke_data_b))]

    # for each component cluster
    for cluster_i in range(len(cluster_indices_component)):
        cluster_component_ids = cluster_indices_component[cluster_i]  # [0, 1, 3] / [2] / [5, 6, 7]
        cluster_stroke_data = []  # component list => curve list => stroke list (N, 2)
        cluster_stroke_data_flatten = []  # curve list => stroke list (N^, 2)
        cluster_stroke_data_b = []  # component list => curve list => stroke list (N', 4, 2)
        cluster_stroke_data_b_flatten = []  # curve list => stroke list (N^, 4, 2)
        cluster_stroke_data_curve_connect_status = []  # component list => new curve connection status (K'), [0, 1, 1, 2, 2, 3, 4]
        cluster_stroke_data_curve_connect_status_flatten = []  # new curve connection status (K'), [0, 1, 1, 2, 2, 3, 4, 5, 6]
        for c_i in cluster_component_ids:
            cluster_stroke_data.append(stroke_data[c_i])
            cluster_stroke_data_flatten += stroke_data[c_i]
            cluster_stroke_data_b.append(stroke_data_b[c_i])
            cluster_stroke_data_b_flatten += stroke_data_b[c_i]
            cluster_stroke_data_curve_connect_status.append(stroke_data_curve_connect_status[c_i])
            if len(cluster_stroke_data_curve_connect_status_flatten) == 0:
                cluster_stroke_data_curve_connect_status_flatten += stroke_data_curve_connect_status[c_i]
            else:
                cluster_stroke_data_curve_connect_status_flatten += (np.array(stroke_data_curve_connect_status[c_i], np.int32) + np.max(
                    cluster_stroke_data_curve_connect_status_flatten) + 1).tolist()

        cluster_component_names = [parts_data[c_i] for c_i in cluster_component_ids]

        cluster_indices_curve, curve_component_map, curve_inner_curve_map = connect_close_curves(
            cluster_stroke_data_b, cluster_stroke_data_curve_connect_status,
            connect_threshold=connect_distance_threshold)
        # cluster_indices_curve: [[0, 1, 3], [2], [4, 5], ...]
        # curve_component_map: [0, 0, 1, 2, 2, 2, 3, 4, ...]
        # curve_inner_curve_map: [0, 1, 0, 0, 1, 2, 0, 0, ...]
        assert len(cluster_indices_curve) > 0

        ## mappings for stroke data (line)
        curve_component_map_ori = []
        curve_inner_curve_map_ori = []
        for s_i in range(len(cluster_stroke_data)):
            curve_component_map_ori += [s_i for _ in range(len(cluster_stroke_data[s_i]))]
            curve_inner_curve_map_ori += [ii for ii in range(len(cluster_stroke_data[s_i]))]
        # print('>> curve_component_map_ori', curve_component_map_ori)
        # print('>> curve_inner_curve_map_ori', curve_inner_curve_map_ori)

        if len(cluster_indices_curve) == 1:
            # no need to do transformation
            for s_i, c_i in enumerate(cluster_component_ids):
                trans_stroke_data[c_i] += cluster_stroke_data[s_i]
                trans_stroke_data_b[c_i] += cluster_stroke_data_b[s_i]
        else:
            trans_cluster_stroke_data = []
            trans_cluster_stroke_data_b = []
            for s_i in range(len(cluster_stroke_data)):
                num_curve = len(cluster_stroke_data[s_i])
                trans_cluster_stroke_data.append([[] for _ in range(num_curve)])
                num_curve_b = len(cluster_stroke_data_b[s_i])
                trans_cluster_stroke_data_b.append([[] for _ in range(num_curve_b)])

            curve_cluster_lens = []
            for cluster_j in range(len(cluster_indices_curve)):
                cluster_curve_ids = cluster_indices_curve[cluster_j]  # [4, 5]
                curve_cluster_len = np.sum([len(cluster_stroke_data_b_flatten[curve_j]) for curve_j in cluster_curve_ids])
                curve_cluster_lens.append(curve_cluster_len)
            is_top_k = find_top_k(curve_cluster_lens, top_k_for_special)

            # for each curve cluster
            for cluster_j in range(len(cluster_indices_curve)):
                cluster_curve_ids = cluster_indices_curve[cluster_j]  # [4, 5]
                cluster_curve_should_do = True

                ## no need to do transformation if the curve cluster contains "only-one" components
                cluster_curve_names = [cluster_component_names[curve_component_map[curve_j]] for curve_j in cluster_curve_ids]
                if len(set(cluster_curve_names).intersection(set(ONLY_ONE_COMPONENTS))) > 0:
                    cluster_curve_should_do = False

                cluster_curve_ids_ori = [cluster_stroke_data_curve_connect_status_flatten[item] for item in cluster_curve_ids]
                cluster_curve_ids_ori = list(set(cluster_curve_ids_ori))

                sub_cluster_stroke_data = []  # curve list (K) => stroke list (N, 2)
                sub_cluster_stroke_data_b = []  # curve list (K') => stroke list (N', 4, 2)
                for curve_j_ori in cluster_curve_ids_ori:
                    sub_cluster_stroke_data.append(cluster_stroke_data_flatten[curve_j_ori])
                for curve_j in cluster_curve_ids:
                    sub_cluster_stroke_data_b.append(cluster_stroke_data_b_flatten[curve_j])

                curve_type = 'special' if is_top_k[cluster_j] else 'normal'

                local_deform_prob = local_deform_prob_map[curve_type]
                local_translation_thresholds = local_translation_thresholds_map[curve_type]
                local_rotation_angle_thresholds = local_rotation_angle_thresholds_map[curve_type]
                local_scaling_thresholds = local_scaling_thresholds_map[curve_type]
                local_shearing_thresholds = local_shearing_thresholds_map[curve_type]

                if should_do(local_deform_prob) and cluster_curve_should_do:
                    n_way_deform = choose_which_index(local_n_way_deform_probs) + 1  # {1, 2, 3}
                    random.shuffle(local_deform_types)
                    selected_deform_types = local_deform_types[:n_way_deform]

                    trans_sub_cluster_stroke_data = [[item for item in sub_cluster_stroke_data]]  # treat as a single component
                    trans_sub_cluster_stroke_data_b = [[item for item in sub_cluster_stroke_data_b]]

                    for selected_deform_type in selected_deform_types:
                        if selected_deform_type == 'translation':
                            trans_sub_cluster_stroke_data, trans_sub_cluster_stroke_data_b = translation_global(
                                trans_sub_cluster_stroke_data,
                                trans_sub_cluster_stroke_data_b,
                                local_translation_thresholds)

                        if selected_deform_type == 'rotation':
                            trans_sub_cluster_stroke_data, trans_sub_cluster_stroke_data_b = rotation_global(
                                trans_sub_cluster_stroke_data,
                                trans_sub_cluster_stroke_data_b,
                                local_rotation_angle_thresholds)

                        if selected_deform_type == 'scaling':
                            trans_sub_cluster_stroke_data, trans_sub_cluster_stroke_data_b = scaling_global(
                                trans_sub_cluster_stroke_data,
                                trans_sub_cluster_stroke_data_b,
                                local_scaling_thresholds)

                        if selected_deform_type == 'shearing':
                            trans_sub_cluster_stroke_data, trans_sub_cluster_stroke_data_b = shearing_global(
                                trans_sub_cluster_stroke_data,
                                trans_sub_cluster_stroke_data_b,
                                local_shearing_thresholds)

                    trans_sub_cluster_stroke_data = trans_sub_cluster_stroke_data[0]  # curve list (K) => stroke list (N, 2)
                    trans_sub_cluster_stroke_data_b = trans_sub_cluster_stroke_data_b[0]  # curve list (K') => stroke list (N, 2)

                    for s_j, curve_j_ori in enumerate(cluster_curve_ids_ori):
                        component_j = curve_component_map_ori[curve_j_ori]
                        inner_curve_j = curve_inner_curve_map_ori[curve_j_ori]  # for placing curves in order
                        assert len(trans_cluster_stroke_data[component_j][inner_curve_j]) == 0
                        trans_cluster_stroke_data[component_j][inner_curve_j] += trans_sub_cluster_stroke_data[s_j]
                    for s_j, curve_j in enumerate(cluster_curve_ids):
                        component_j = curve_component_map[curve_j]
                        inner_curve_j = curve_inner_curve_map[curve_j]  # for placing curves in order
                        assert len(trans_cluster_stroke_data_b[component_j][inner_curve_j]) == 0
                        trans_cluster_stroke_data_b[component_j][inner_curve_j] += trans_sub_cluster_stroke_data_b[s_j]
                else:
                    for s_j, curve_j_ori in enumerate(cluster_curve_ids_ori):
                        component_j = curve_component_map_ori[curve_j_ori]
                        inner_curve_j = curve_inner_curve_map_ori[curve_j_ori]  # for placing curves in order
                        assert len(trans_cluster_stroke_data[component_j][inner_curve_j]) == 0
                        if type(sub_cluster_stroke_data[s_j]) is np.ndarray:
                            trans_cluster_stroke_data[component_j][inner_curve_j] += sub_cluster_stroke_data[s_j].tolist()
                        else:
                            trans_cluster_stroke_data[component_j][inner_curve_j] += sub_cluster_stroke_data[s_j]
                    for s_j, curve_j in enumerate(cluster_curve_ids):
                        component_j = curve_component_map[curve_j]
                        inner_curve_j = curve_inner_curve_map[curve_j]  # for placing curves in order
                        assert len(trans_cluster_stroke_data_b[component_j][inner_curve_j]) == 0
                        if type(sub_cluster_stroke_data_b[s_j]) is np.ndarray:
                            trans_cluster_stroke_data_b[component_j][inner_curve_j] += sub_cluster_stroke_data_b[s_j].tolist()
                        else:
                            trans_cluster_stroke_data_b[component_j][inner_curve_j] += sub_cluster_stroke_data_b[s_j]

            for s_i, c_i in enumerate(cluster_component_ids):  # cluster_component_ids: [0, 1, 3]
                trans_stroke_data[c_i] += trans_cluster_stroke_data[s_i]
                trans_stroke_data_b[c_i] += trans_cluster_stroke_data_b[s_i]

    deep_shape_check(stroke_data, trans_stroke_data)
    deep_shape_check(stroke_data_b, trans_stroke_data_b)
    return trans_stroke_data, trans_stroke_data_b


def check_out_of_bound(stroke_data_b, image_size, exceed_threshold=0.25):
    '''
    stroke_data_b: component list => curve list => stroke list (N', 4, 2)
    '''
    for c_i in range(len(stroke_data_b)):
        curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
        for curve_j, curve in enumerate(curve_b_list):  # (N', 4, 2)
            curve_points_start = np.array(curve)[0:1, 0, :]  # (1, 2)
            curve_points_others = np.array(curve)[:, 1:, :].reshape((-1, 2))  # (N'*3, 2)
            curve_points = np.concatenate([curve_points_start, curve_points_others], axis=0)  # (N'*3+1, 2)
            num_points = len(curve_points)
            num_out_of_bound = np.sum(np.logical_or((curve_points < 0).any(axis=-1),
                                                    (curve_points > image_size).any(axis=-1)))
            if float(num_out_of_bound) / float(num_points) > exceed_threshold:
                return True
    return False


def generate_paired_deformation(stroke_data, stroke_data_b, parts_data, stroke_data_curve_connect_status,
                                canvas_size):
    '''
    stroke_data: component list => curve list => stroke list (N, 2)
    stroke_data_b: component list => curve list => stroke list (N', 4, 2)
    stroke_data_curve_connect_status: component list => new curve connection status (K'), [0, 1, 1, 2, 2, 3, 4]
    '''
    while True:
        trans_stroke_data_list = []
        trans_stroke_data_b_list = []

        trans_stroke_data, trans_stroke_data_b = sketch_global_deformation(stroke_data, stroke_data_b)
        # trans_stroke_data: component list => curve list => stroke list (N, 2)
        # trans_stroke_data_b: component list => curve list => stroke list (N', 4, 2)
        if check_out_of_bound(trans_stroke_data_b, canvas_size):
            continue
        trans_stroke_data_list.append(trans_stroke_data)
        trans_stroke_data_b_list.append(trans_stroke_data_b)

        trans_stroke_data, trans_stroke_data_b, cluster_indices_component = sketch_component_deformation(trans_stroke_data, trans_stroke_data_b)
        # trans_stroke_data: component list => curve list => stroke list (N, 2)
        # trans_stroke_data_b: component list => curve list => stroke list (N', 4, 2)
        if check_out_of_bound(trans_stroke_data_b, canvas_size):
            continue
        trans_stroke_data_list.append(trans_stroke_data)
        trans_stroke_data_b_list.append(trans_stroke_data_b)

        trans_stroke_data, trans_stroke_data_b = sketch_curve_deformation(trans_stroke_data, trans_stroke_data_b,
                                                                           stroke_data_curve_connect_status, cluster_indices_component,
                                                                           parts_data)
        # trans_stroke_data: component list => curve list => stroke list (N, 2)
        # trans_stroke_data_b: component list => curve list => stroke list (N', 4, 2)
        if check_out_of_bound(trans_stroke_data_b, canvas_size):
            continue
        trans_stroke_data_list.append(trans_stroke_data)
        trans_stroke_data_b_list.append(trans_stroke_data_b)

        return trans_stroke_data_list, trans_stroke_data_b_list
