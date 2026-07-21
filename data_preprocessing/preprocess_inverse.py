import os
import numpy as np
import shutil
import jsonlines
from PIL import Image
import copy

from utils.draw_sketch import draw_sketch_cairo, draw_sketch_stroke_chain_cairo
from utils.curve_grouping import curve_grouping_flow_inv
from preprocess_forward import load_svg, vector_curve_splitting, component_translation_with_optical_flow_main, \
    gen_component_images, gen_stroke_images_ctrlpoint

from configs.example_configs import test_data_base, test_img_id, gen_time, example_info_map


def find_connected_points(database, connect_threshold, process_img_id):
    print('============== find_connected_points ==============')
    vector_data_dir = os.path.join(database, 'vector-params-split')

    save_base_parameter = os.path.join(database, 'vector-params-conn')
    save_base_vis = os.path.join(save_base_parameter, 'vis')
    os.makedirs(save_base_parameter, exist_ok=True)
    os.makedirs(save_base_vis, exist_ok=True)

    ref_img_path = os.path.join(database, 'raster_black', str(process_img_id) + '_ref.png')
    assert os.path.exists(ref_img_path)
    ref_img = Image.open(ref_img_path)
    image_size = ref_img.height

    params_path_tar1 = os.path.join(vector_data_dir, str(process_img_id) + '_ref.jsonl')
    with open(params_path_tar1, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b_tar1_raw = item['stroke_params']  # curve list => stroke list (N', 4, 2)

    data_base_extra = "outputs/stroke_correspondence_results"
    if gen_time > 0:
        data_base_extra += '-Gen%d' % gen_time
    params_path_tar0 = os.path.join(data_base_extra, 'params', 'tar_pred-' + str(process_img_id) + '.jsonl')
    with open(params_path_tar0, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b_tar0 = item['stroke_params']
            # stroke_data_b: a list of layers
            #   => layer: a list of stroke chains
            #     => stroke chain: a list of strokes
            #       => stroke: (4, 2)

    flatten_to_nested_map_tar0 = {}
    curve_endpoints_tar0 = []  # (N0, 2)
    for c_i, component in enumerate(stroke_data_b_tar0):
        for curve_i, curve in enumerate(component):  # (N, 4, 2)
            start_point = curve[0][0]  # (2)
            flatten_to_nested_map_tar0[len(curve_endpoints_tar0)] = '_'.join([str(c_i), str(curve_i), '0'])
            curve_endpoints_tar0.append(start_point)
            end_point = curve[-1][-1]  # (2)
            flatten_to_nested_map_tar0[len(curve_endpoints_tar0)] = '_'.join([str(c_i), str(curve_i), str(len(curve))])
            curve_endpoints_tar0.append(end_point)

    stroke_data_b_tar1_new = []  # list of (N', 4, 2), with connected point fixing, len = N_curve1
    curves_endpoint_connected_state_tar1 = []  # len = N_curve1, each with ['0_1_2', None] for starting and ending poits
    for curve_i in range(len(stroke_data_b_tar1_raw)):
        curve = stroke_data_b_tar1_raw[curve_i]  # (N', 4, 2)
        start_point = curve[0][0]  # (2)
        end_point = curve[-1][-1]  # (2)
        curve_endpoints_tar1 = np.stack([start_point, end_point], axis=0)  # (2, 2)

        tar1_tar0_dist = np.expand_dims(curve_endpoints_tar1, axis=1) - np.expand_dims(curve_endpoints_tar0, axis=0)  # (2, N0, 2)
        tar1_tar0_dist = np.sqrt(np.sum(np.power(tar1_tar0_dist, 2), axis=-1))  # (2, N0)

        connect_indices_tar1_tar0 = np.argwhere(tar1_tar0_dist <= connect_threshold)  # list of (2)

        curve_new = copy.deepcopy(curve)  # (N', 4, 2)
        endpoint_connected_state = [None, None]
        if len(connect_indices_tar1_tar0) > 0:
            connect_indices_tar1_tar0 = np.stack(connect_indices_tar1_tar0, axis=0)  # (N', 2)
            start_connect = np.argwhere(connect_indices_tar1_tar0[:, 0] == 0).squeeze(axis=1)
            end_connect = np.argwhere(connect_indices_tar1_tar0[:, 0] == 1).squeeze(axis=1)
            assert len(start_connect) <= 1 and len(end_connect) <= 1

            # starting point is connected
            if 0 < len(start_connect) <= 1:
                tar0_corr_index = connect_indices_tar1_tar0[start_connect[0]][1]
                # print('curve_i', curve_i, ', 0, dist', tar1_tar0_dist[0, tar0_corr_index])
                curve_new[0][0] = copy.deepcopy(curve_endpoints_tar0[tar0_corr_index])
                endpoint_connected_state[0] = flatten_to_nested_map_tar0[tar0_corr_index]

            # ending point is connected
            if 0 < len(end_connect) <= 1:
                tar0_corr_index = connect_indices_tar1_tar0[end_connect[0]][1]
                # print('curve_i', curve_i, ', 1, dist', tar1_tar0_dist[1, tar0_corr_index])
                curve_new[-1][-1] = copy.deepcopy(curve_endpoints_tar0[tar0_corr_index])
                endpoint_connected_state[1] = flatten_to_nested_map_tar0[tar0_corr_index]

        stroke_data_b_tar1_new.append(curve_new)
        curves_endpoint_connected_state_tar1.append(endpoint_connected_state)

    vector_data_save_path = os.path.join(save_base_parameter, str(process_img_id) + '_ref.jsonl')
    vector_data = {}
    vector_data['stroke_params'] = stroke_data_b_tar1_new
    vector_data['connect_state'] = curves_endpoint_connected_state_tar1
    with jsonlines.open(vector_data_save_path, mode='w') as json_writer:
        json_writer.write(vector_data)

    if True:  # vis
        stroke_data_b_tar1_vis = [[], []]
        for curve_i in range(len(stroke_data_b_tar1_new)):
            curve_b_points = stroke_data_b_tar1_new[curve_i]  # list (N') of (4, 2)
            endpoint_connected_state = curves_endpoint_connected_state_tar1[curve_i]  # ['0_1_2', None]
            for p_i in range(len(curve_b_points)):
                if p_i == 0 and endpoint_connected_state[0] is not None or \
                        p_i == len(curve_b_points) - 1 and endpoint_connected_state[1] is not None:
                    stroke_data_b_tar1_vis[1].append([curve_b_points[p_i]])
                else:
                    stroke_data_b_tar1_vis[0].append([curve_b_points[p_i]])

        vis_path = os.path.join(save_base_vis, str(process_img_id) + "_ref.png")
        draw_sketch_stroke_chain_cairo(stroke_data_b_tar1_vis, vis_path, is_bezier=True,
                                        side=image_size, line_diameter=line_thickness)


def curve_grouping(database, example_info_map, line_thickness, process_img_id):
    print('============== curve_grouping ==============')
    vector_data_dir = os.path.join(database, 'vector-params-conn')

    save_base_parameter = os.path.join(database, 'vector-params')
    os.makedirs(save_base_parameter, exist_ok=True)

    params_path_tar1 = os.path.join(vector_data_dir, str(process_img_id) + '_ref.jsonl')
    with open(params_path_tar1, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b_tar1_raw = item['stroke_params']  # curve list => stroke list (N', 4, 2)
            curves_endpoint_connected_state_raw = item['connect_state']  # len = N_curve, each with ['0_1_2', None] for starting and ending poits

    data_base_extra = "outputs/stroke_correspondence_results"
    if gen_time > 0:
        data_base_extra += '-Gen%d' % gen_time
    params_path_tar0 = os.path.join(data_base_extra, 'params', 'tar_pred-' + str(process_img_id) + '.jsonl')
    with open(params_path_tar0, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b_tar0 = item['stroke_params']
            # stroke_data_b: a list of layers
            #   => layer: a list of stroke chains
            #     => stroke chain: a list of strokes
            #       => stroke: (4, 2)
    max_group_num = len(stroke_data_b_tar0)

    ref_img_path = os.path.join(database, 'raster_black', str(process_img_id) + '_ref.png')
    assert os.path.exists(ref_img_path)
    ref_img = Image.open(ref_img_path)
    image_size = ref_img.height

    group_labels, optical_flow_dir = curve_grouping_flow_inv(database, stroke_data_b_tar0, stroke_data_b_tar1_raw, curves_endpoint_connected_state_raw,
                                                                image_size, process_img_id,
                                                                example_info_map[str(process_img_id)]["optical_flow_method"],
                                                                example_info_map[str(process_img_id)]["use_distance_transform"])
    # print('group_labels', group_labels)

    stroke_data_b = [[] for _ in range(max_group_num)]  # component list => curve list => stroke list (N', 4, 2)
    curves_endpoint_connected_state = [[] for _ in range(max_group_num)]  # component list => curve list => ['0_1_2', None]
    for curve_i in range(len(group_labels)):
        group_label = group_labels[curve_i]
        stroke_data_b[group_label].append(stroke_data_b_tar1_raw[curve_i])
        curves_endpoint_connected_state[group_label].append(curves_endpoint_connected_state_raw[curve_i])

    save_base_grouping = os.path.join(database, 'optical_flow', optical_flow_dir, 'vector-vis-group')
    os.makedirs(save_base_grouping, exist_ok=True)
    vector_vis_path = os.path.join(save_base_grouping, str(process_img_id) + "_ref.png")
    draw_sketch_stroke_chain_cairo(stroke_data_b, vector_vis_path, is_bezier=True,
                                    side=image_size, line_diameter=line_thickness)

    # save layer-level parameters
    save_jsonl_path = os.path.join(save_base_parameter, str(process_img_id) + '_ref.jsonl')
    vector_data_ref = {}
    vector_data_ref['stroke_params'] = stroke_data_b
    vector_data_ref['connect_state'] = curves_endpoint_connected_state
    with jsonlines.open(save_jsonl_path, mode='w') as json_writer:
        json_writer.write(vector_data_ref)

    save_base_grouping_label = os.path.join(database, 'optical_flow', optical_flow_dir, 'group_labels')
    os.makedirs(save_base_grouping_label, exist_ok=True)
    save_jsonl_path = os.path.join(save_base_grouping_label, str(process_img_id) + '_ref.jsonl')
    group_labels_data_ref = {}
    group_labels_data_ref['group_labels'] = group_labels
    with jsonlines.open(save_jsonl_path, mode='w') as json_writer:
        json_writer.write(group_labels_data_ref)


def gen_stroke_images(database, line_thickness, process_img_id):
    print('============== gen_stroke_images ==============')

    vector_data_dir = os.path.join(database, 'vector-params')

    stroke_image_base = os.path.join(database, 'raster_black_endpoint_stroke')
    stroke_image_base_prev = os.path.join(database, 'raster_black_endpoint_stroke_prev')
    stroke_image_base_next = os.path.join(database, 'raster_black_endpoint_stroke_next')
    os.makedirs(stroke_image_base, exist_ok=True)
    os.makedirs(stroke_image_base_prev, exist_ok=True)
    os.makedirs(stroke_image_base_next, exist_ok=True)

    ref_img_path = os.path.join(database, 'raster_black', str(process_img_id) + '_ref.png')
    assert os.path.exists(ref_img_path)
    ref_img = Image.open(ref_img_path)
    image_size = ref_img.height

    stroke_image_dir = os.path.join(stroke_image_base, str(process_img_id))
    if os.path.exists(stroke_image_dir):
        shutil.rmtree(stroke_image_dir)
    os.makedirs(stroke_image_dir, exist_ok=True)
    stroke_image_dir_prev = os.path.join(stroke_image_base_prev, str(process_img_id))
    if os.path.exists(stroke_image_dir_prev):
        shutil.rmtree(stroke_image_dir_prev)
    os.makedirs(stroke_image_dir_prev, exist_ok=True)
    stroke_image_dir_next = os.path.join(stroke_image_base_next, str(process_img_id))
    if os.path.exists(stroke_image_dir_next):
        shutil.rmtree(stroke_image_dir_next)
    os.makedirs(stroke_image_dir_next, exist_ok=True)

    params_path_tar1 = os.path.join(vector_data_dir, str(process_img_id) + '_ref.jsonl')
    with open(params_path_tar1, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b = item['stroke_params']  # component list => curve list => stroke list (N', 4, 2)
            curves_endpoint_connected_state = item['connect_state']  # component list => curve list => ['0_1_2', None]

    data_base_extra = "outputs/stroke_correspondence_results"
    if gen_time > 0:
        data_base_extra += '-Gen%d' % gen_time
    params_path_tar0 = os.path.join(data_base_extra, 'params', 'tar_pred-' + str(process_img_id) + '.jsonl')
    with open(params_path_tar0, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b_tar0 = item['stroke_params']
            # stroke_data_b: a list of layers
            #   => layer: a list of stroke chains
            #     => stroke chain: a list of strokes
            #       => stroke: (4, 2)

    for c_i in range(len(stroke_data_b)):
        curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
        endpoint_connected_state_list = curves_endpoint_connected_state[c_i]  # list of (2), ['0_1_2', None]

        curve_b_list_tar0 = stroke_data_b_tar0[c_i]  # list of (N', 4, 2)

        for curve_i in range(len(curve_b_list)):
            curve_b_points = curve_b_list[curve_i]  # list (N') of (4, 2)
            endpoint_connected_states = endpoint_connected_state_list[curve_i]  # ['0_1_2', None]

            stroke_num = len(curve_b_points)

            # starting stroke
            outpath = os.path.join(stroke_image_dir, "endpoint_%s_%s_%s.png" % (c_i, curve_i, 0))
            if endpoint_connected_states[0] is None:
                stroke_data_b_single = [[[curve_b_points[0]]]]
            else:
                corr_comp, corr_curve, corr_point = endpoint_connected_states[0].split('_')
                assert int(corr_comp) == c_i
                if int(corr_point) == 0:  # connected to the starting point of a curve
                    corr_stroke = curve_b_list_tar0[int(corr_curve)][int(corr_point)]  # (4, 2)
                else:  # connected to the ending point of a curve
                    corr_stroke = curve_b_list_tar0[int(corr_curve)][int(corr_point) - 1]  # (4, 2)
                stroke_data_b_single = [[[curve_b_points[0], corr_stroke]]]
            draw_sketch_cairo(stroke_data_b_single, outpath, is_bezier=True,
                                side=image_size, line_diameter=line_thickness)

            # intermediate strokes
            for p_i in range(1, stroke_num):
                outpath = os.path.join(stroke_image_dir, "endpoint_%s_%s_%s.png" % (c_i, curve_i, p_i))
                stroke_data_b_single = [[[curve_b_points[p_i - 1], curve_b_points[p_i]]]]
                draw_sketch_cairo(stroke_data_b_single, outpath, is_bezier=True,
                                    side=image_size, line_diameter=line_thickness)

            # last stroke
            outpath = os.path.join(stroke_image_dir, "endpoint_%s_%s_%s.png" % (c_i, curve_i, stroke_num))
            if endpoint_connected_states[1] is None:
                stroke_data_b_single = [[[curve_b_points[stroke_num - 1]]]]
            else:
                corr_comp, corr_curve, corr_point = endpoint_connected_states[1].split('_')
                assert int(corr_comp) == c_i
                if int(corr_point) == 0:  # connected to the starting point of a curve
                    corr_stroke = curve_b_list_tar0[int(corr_curve)][int(corr_point)]  # (4, 2)
                else:  # connected to the ending point of a curve
                    corr_stroke = curve_b_list_tar0[int(corr_curve)][int(corr_point) - 1]  # (4, 2)
                stroke_data_b_single = [[[curve_b_points[stroke_num - 1], corr_stroke]]]
            draw_sketch_cairo(stroke_data_b_single, outpath, is_bezier=True,
                                side=image_size, line_diameter=line_thickness)


if __name__ == '__main__':
    database = test_data_base
    process_img_id = test_img_id

    line_thickness = 3
    connect_threshold = 1

    ###################### Main process ######################

    load_svg(os.path.join(database, '[0inv]'), line_thickness, process_img_id)

    vector_curve_splitting(os.path.join(database, '[0inv]'), line_thickness, process_img_id)

    find_connected_points(os.path.join(database, '[0inv]'), connect_threshold, process_img_id)

    curve_grouping(os.path.join(database, '[0inv]'), example_info_map, line_thickness, process_img_id)

    ## For component transform task
    component_translation_with_optical_flow_main(os.path.join(database, '[0inv]'), line_thickness, None, None, process_img_id,
                                                 data_extra=example_info_map)
    gen_component_images(os.path.join(database, '[0inv]'), line_thickness, process_img_id, data_extra=example_info_map)

    ## For endpoint matching task
    gen_stroke_images(os.path.join(database, '[0inv]'), line_thickness, process_img_id)

    ## For control point matching task
    gen_stroke_images_ctrlpoint(os.path.join(database, '[0inv]'), line_thickness, process_img_id)