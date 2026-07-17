import os
import json
import jsonlines
import cv2
import numpy as np
import copy

from utils.draw_sketch import draw_sketch_stroke_cairo
from configs.example_configs import test_data_base, test_img_id, gen_time


def load_stroke_parameter(vector_data_path):
    curves_endpoint_connected_state = None
    with open(vector_data_path, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b = item['stroke_params']
            if 'connect_state' in item.keys():
                curves_endpoint_connected_state = item['connect_state']  # component list => curve list => ['0_1_2', None]
    return stroke_data_b, curves_endpoint_connected_state


def find_connect_states(c_i, curve_b_list_ref0, curve_b_list_tar1,
                        curves_endpoint_connected_state_ref0, curves_endpoint_connected_state_tar1):
    num_prev_curves = len(curve_b_list_ref0)
    num_new_curves = len(curve_b_list_tar1)
    if curves_endpoint_connected_state_ref0 is None:  # Gen-0
        curves_endpoint_connected_state_comb_comp = [[None, None] for _ in range(num_prev_curves + num_new_curves)]
    else:  # Gen-1
        curves_endpoint_connected_state_comb_comp = copy.deepcopy(curves_endpoint_connected_state_ref0[c_i]) + [[None, None] for _ in range(num_new_curves)]

    endpoint_connected_state_list_tar1 = curves_endpoint_connected_state_tar1[c_i]  # curve list, ['0_1_2', None]
    assert len(curve_b_list_tar1) == len(endpoint_connected_state_list_tar1)

    for curve_i_new in range(len(curve_b_list_tar1)):
        endpoint_connected_states_new = endpoint_connected_state_list_tar1[curve_i_new]  # ['0_1_2', None]

        curve_i_new_comb = num_prev_curves + curve_i_new
        curve_i_new_start_point_idx = 0
        curve_i_new_end_point_idx = len(curve_b_list_tar1[curve_i_new])

        # Store connected states for new strokes (the same)
        assert curves_endpoint_connected_state_comb_comp[curve_i_new_comb][0] is None and \
               curves_endpoint_connected_state_comb_comp[curve_i_new_comb][1] is None
        curves_endpoint_connected_state_comb_comp[curve_i_new_comb] = endpoint_connected_states_new

        # Store connected states for prev strokes (change the curve index)
        if endpoint_connected_states_new[0] is not None:  # starting point of the new curve has connection
            c_i_prev, curve_i_prev, point_i_prev = endpoint_connected_states_new[0].split('_')  # 0, 1, 2
            c_i_prev, curve_i_prev, point_i_prev = int(c_i_prev), int(curve_i_prev), int(point_i_prev)
            assert c_i_prev == c_i
            conn_state_idx_prev = 0 if point_i_prev == 0 else 1  # connected to the start/end point of prev curve

            assert curves_endpoint_connected_state_comb_comp[curve_i_prev][conn_state_idx_prev] is None
            curves_endpoint_connected_state_comb_comp[curve_i_prev][conn_state_idx_prev] = '_'.join([str(c_i),
                                                                                                     str(curve_i_new_comb),
                                                                                                     str(curve_i_new_start_point_idx)])

        if endpoint_connected_states_new[1] is not None:  # ending point of the new curve has connection
            c_i_prev, curve_i_prev, point_i_prev = endpoint_connected_states_new[1].split('_')  # 0, 1, 2
            c_i_prev, curve_i_prev, point_i_prev = int(c_i_prev), int(curve_i_prev), int(point_i_prev)
            assert c_i_prev == c_i
            conn_state_idx_prev = 0 if point_i_prev == 0 else 1  # connected to the start/end point of prev curve

            assert curves_endpoint_connected_state_comb_comp[curve_i_prev][conn_state_idx_prev] is None
            curves_endpoint_connected_state_comb_comp[curve_i_prev][conn_state_idx_prev] = '_'.join([str(c_i),
                                                                                                     str(curve_i_new_comb),
                                                                                                     str(curve_i_new_end_point_idx)])
    return curves_endpoint_connected_state_comb_comp


def combine_single_image_v1(params_path_ref0, params_path_ref1, params_path_tar0, params_path_tar1):
    """
    Visualize colors according to the first and second rounds, without reordering the colors according to components
    """
    assert os.path.exists(params_path_ref0)
    assert os.path.exists(params_path_tar0)
    params_ref0, curves_endpoint_connected_state_ref0 = load_stroke_parameter(params_path_ref0)
    params_tar0, _ = load_stroke_parameter(params_path_tar0)

    if not os.path.exists(params_path_tar1):  # w/o inverse prediction; should be Gen>0?
        assert not os.path.exists(params_path_ref1)
        assert curves_endpoint_connected_state_ref0 is not None
        params_tar1 = [[] for _ in range(len(params_tar0))]
        curves_endpoint_connected_state_tar1 = [[] for _ in range(len(params_tar0))]
    else:
        params_tar1, curves_endpoint_connected_state_tar1 = load_stroke_parameter(params_path_tar1)

    if not os.path.exists(params_path_ref1):
        params_ref1 = copy.deepcopy(params_tar1)
    else:
        params_ref1, _ = load_stroke_parameter(params_path_ref1)

    assert len(params_ref0) == len(params_ref1) == len(params_tar0) == len(params_tar1)

    params_comb_ref = []
    params_comb_tar = []
    params_comb_ref_addi = []
    params_comb_tar_addi = []

    ## Generate connected states
    curves_endpoint_connected_state_comb = []  # component list => curve list => ['0_1_2', None]

    for c_i in range(len(params_ref0)):
        curve_b_list_ref0 = params_ref0[c_i]  # list of curve in (N', 4, 2)
        curve_b_list_ref1 = params_ref1[c_i]  # list of curve in (N', 4, 2)
        curve_b_list_tar0 = params_tar0[c_i]  # list of curve in (N', 4, 2)
        curve_b_list_tar1 = params_tar1[c_i]  # list of curve in (N', 4, 2)

        params_comb_ref.append(curve_b_list_ref0)
        params_comb_tar.append(curve_b_list_tar0)

        params_comb_ref_addi += curve_b_list_ref1
        params_comb_tar_addi += curve_b_list_tar1

        curves_endpoint_connected_state_comb_comp = find_connect_states(c_i, curve_b_list_ref0, curve_b_list_tar1,
                                                                        curves_endpoint_connected_state_ref0,
                                                                        curves_endpoint_connected_state_tar1)
        curves_endpoint_connected_state_comb.append(curves_endpoint_connected_state_comb_comp)

    params_comb_ref.append(params_comb_ref_addi)
    params_comb_tar.append(params_comb_tar_addi)

    assert len(params_comb_ref) == len(params_comb_tar)
    return params_comb_ref, params_comb_tar, curves_endpoint_connected_state_comb


def combine_single_image_v0(params_path_ref0, params_path_ref1, params_path_tar0, params_path_tar1):
    """
    Combine strokes into the same component
    """
    assert os.path.exists(params_path_ref0)
    assert os.path.exists(params_path_tar0)
    params_ref0, curves_endpoint_connected_state_ref0 = load_stroke_parameter(params_path_ref0)
    params_tar0, _ = load_stroke_parameter(params_path_tar0)

    if not os.path.exists(params_path_tar1):  # w/o inverse prediction; should be Gen>0?
        assert not os.path.exists(params_path_ref1)
        assert curves_endpoint_connected_state_ref0 is not None
        params_tar1 = [[] for _ in range(len(params_tar0))]
        curves_endpoint_connected_state_tar1 = [[] for _ in range(len(params_tar0))]
    else:
        params_tar1, curves_endpoint_connected_state_tar1 = load_stroke_parameter(params_path_tar1)

    if not os.path.exists(params_path_ref1):
        params_ref1 = copy.deepcopy(params_tar1)
    else:
        params_ref1, _ = load_stroke_parameter(params_path_ref1)

    assert len(params_ref0) == len(params_ref1) == len(params_tar0) == len(params_tar1)

    params_comb_ref = []
    params_comb_tar = []

    ## Generate connected states
    curves_endpoint_connected_state_comb = []  # component list => curve list => ['0_1_2', None]

    for c_i in range(len(params_ref0)):
        curve_b_list_ref0 = params_ref0[c_i]  # list of curve in (N', 4, 2)
        curve_b_list_ref1 = params_ref1[c_i]  # list of curve in (N', 4, 2)
        ## Used for undrawn some curves (e.g., the bidirectional ones)
        # if len(curve_b_list_ref1) > 0:
        #     for curve_i in range(len(curve_b_list_ref1)):
        #         curve_b_list_ref1[curve_i][0][0][0] = -404.0
        curve_b_list_tar0 = params_tar0[c_i]  # list of curve in (N', 4, 2)
        curve_b_list_tar1 = params_tar1[c_i]  # list of curve in (N', 4, 2)

        params_comb_ref.append(curve_b_list_ref0 + curve_b_list_ref1)
        params_comb_tar.append(curve_b_list_tar0 + curve_b_list_tar1)

        curves_endpoint_connected_state_comb_comp = find_connect_states(c_i, curve_b_list_ref0, curve_b_list_tar1,
                                                                        curves_endpoint_connected_state_ref0,
                                                                        curves_endpoint_connected_state_tar1)
        curves_endpoint_connected_state_comb.append(curves_endpoint_connected_state_comb_comp)

    assert len(params_comb_ref) == len(params_comb_tar)
    return params_comb_ref, params_comb_tar, curves_endpoint_connected_state_comb


def count_stroke_num(stroke_data_b):
    total_stroke_num = 0
    component_stroke_nums = []
    for c_i in range(len(stroke_data_b)):
        curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
        component_stroke_num = 0
        for curve_i in range(len(curve_b_list)):
            curve_b_points = curve_b_list[curve_i]  # list (N') of (4, 2)
            stroke_num = len(curve_b_points)
            total_stroke_num += stroke_num
            component_stroke_num += stroke_num
        component_stroke_nums.append(component_stroke_num)
    return total_stroke_num, component_stroke_nums


def main_real(database, process_img_id, vis_mode='v0', vis_layer=False):
    line_thickness = 3

    data_base_extra = "outputs/stroke_correspondence_results"
    if gen_time > 0:
        data_base_extra += '-[Gen%d]' % gen_time

    params_path_ref0 = os.path.join(database, 'vector-params', str(process_img_id) + '_ref.jsonl')
    params_path_tar0 = os.path.join(data_base_extra, 'params', 'tar_pred-' + str(process_img_id) + '.jsonl')
    params_path_ref1 = os.path.join(data_base_extra, '[0inv]', 'params', 'tar_pred-' + str(process_img_id) + '.jsonl')
    params_path_tar1 = os.path.join(database, '[0inv]', 'vector-params', str(process_img_id) + '_ref.jsonl')

    save_base = os.path.join(data_base_extra, '[1comb]', 'vector-vis-comb')
    save_base_wo_bg = os.path.join(data_base_extra, '[1comb]', 'vector-vis-comb', 'wo-bg')
    save_base_param = os.path.join(data_base_extra, '[1comb]', 'vector-params')
    os.makedirs(save_base, exist_ok=True)
    os.makedirs(save_base_wo_bg, exist_ok=True)
    os.makedirs(save_base_param, exist_ok=True)

    if vis_layer:
        save_base_layer = os.path.join(save_base, 'single_layer', str(process_img_id))
        save_base_wo_bg_layer = os.path.join(save_base_wo_bg, 'single_layer', str(process_img_id))
        os.makedirs(save_base_layer, exist_ok=True)
        os.makedirs(save_base_wo_bg_layer, exist_ok=True)

    if vis_mode == 'v0':
        params_comb_ref, params_comb_tar, curves_endpoint_connected_state_comb = combine_single_image_v0(params_path_ref0, params_path_ref1, params_path_tar0, params_path_tar1)
    elif vis_mode == 'v1':
        params_comb_ref, params_comb_tar, curves_endpoint_connected_state_comb = combine_single_image_v1(params_path_ref0, params_path_ref1, params_path_tar0, params_path_tar1)
    else:
        raise Exception('Unknown vis_mode:', vis_mode)

    total_stroke_num, component_stroke_nums = count_stroke_num(params_comb_ref)
    print('#', process_img_id, 'total_stroke_num:', total_stroke_num)

    if vis_mode == 'v0':
        save_jsonl_path_ref = os.path.join(save_base_param, str(process_img_id) + '_ref.jsonl')
        vector_data_ref = {}
        vector_data_ref['stroke_params'] = params_comb_ref
        with jsonlines.open(save_jsonl_path_ref, mode='w') as json_writer:
            json_writer.write(vector_data_ref)

        save_jsonl_path_tar = os.path.join(save_base_param, str(process_img_id) + '_tar.jsonl')
        vector_data_tar = {}
        vector_data_tar['stroke_params'] = params_comb_tar
        vector_data_tar['connect_state'] = curves_endpoint_connected_state_comb
        with jsonlines.open(save_jsonl_path_tar, mode='w') as json_writer:
            json_writer.write(vector_data_tar)

    # vis
    raster_img_path_ref = os.path.join(database, 'raster_black', str(process_img_id) + "_ref.png")
    raster_img_ref = cv2.imread(raster_img_path_ref)[:, :, 0]  # (H, W), [0-stroke, 255-BG]
    raster_img_path_tar = os.path.join(database, 'raster_black', str(process_img_id) + "_tar.png")
    raster_img_tar = cv2.imread(raster_img_path_tar)[:, :, 0]  # (H, W), [0-stroke, 255-BG]

    img_width = raster_img_ref.shape[0]

    ## with bg
    vector_vis_path = os.path.join(save_base, str(process_img_id) + "_ref.png")
    draw_sketch_stroke_cairo(params_comb_ref, vector_vis_path, is_bezier=True,
                                side=img_width, line_diameter=line_thickness, bg_sketch=raster_img_ref)

    vector_vis_path = os.path.join(save_base, str(process_img_id) + "_tar.png")
    draw_sketch_stroke_cairo(params_comb_tar, vector_vis_path, is_bezier=True,
                                side=img_width, line_diameter=line_thickness, bg_sketch=raster_img_tar)

    # vis single layer
    if vis_layer:
        for c_i in range(len(params_comb_ref)):
            vector_vis_path = os.path.join(save_base_layer, str(c_i) + "_ref.png")
            draw_sketch_stroke_cairo([params_comb_ref[c_i]], vector_vis_path, is_bezier=True,
                                        side=img_width, line_diameter=line_thickness, bg_sketch=raster_img_ref,
                                        max_seq_number=total_stroke_num, color_shift=int(np.sum(component_stroke_nums[:c_i])))

            vector_vis_path = os.path.join(save_base_layer, str(c_i) + "_tar.png")
            draw_sketch_stroke_cairo([params_comb_tar[c_i]], vector_vis_path, is_bezier=True,
                                        side=img_width, line_diameter=line_thickness, bg_sketch=raster_img_tar,
                                        max_seq_number=total_stroke_num, color_shift=int(np.sum(component_stroke_nums[:c_i])))

    ## w/o bg
    vector_vis_path = os.path.join(save_base_wo_bg, str(process_img_id) + "_ref.png")
    draw_sketch_stroke_cairo(params_comb_ref, vector_vis_path, is_bezier=True,
                                side=img_width, line_diameter=line_thickness, bg_sketch=None)

    vector_vis_path = os.path.join(save_base_wo_bg, str(process_img_id) + "_tar.png")
    draw_sketch_stroke_cairo(params_comb_tar, vector_vis_path, is_bezier=True,
                                side=img_width, line_diameter=line_thickness, bg_sketch=None)

    # vis single layer
    if vis_layer:
        for c_i in range(len(params_comb_ref)):
            vector_vis_path = os.path.join(save_base_wo_bg_layer, str(c_i) + "_ref.png")
            draw_sketch_stroke_cairo([params_comb_ref[c_i]], vector_vis_path, is_bezier=True,
                                        side=img_width, line_diameter=line_thickness, bg_sketch=None,
                                        max_seq_number=total_stroke_num, color_shift=int(np.sum(component_stroke_nums[:c_i])))

            vector_vis_path = os.path.join(save_base_wo_bg_layer, str(c_i) + "_tar.png")
            draw_sketch_stroke_cairo([params_comb_tar[c_i]], vector_vis_path, is_bezier=True,
                                        side=img_width, line_diameter=line_thickness, bg_sketch=None,
                                        max_seq_number=total_stroke_num, color_shift=int(np.sum(component_stroke_nums[:c_i])))


if __name__ == '__main__':
    database = test_data_base
    process_img_id = test_img_id

    main_real(database, process_img_id)
