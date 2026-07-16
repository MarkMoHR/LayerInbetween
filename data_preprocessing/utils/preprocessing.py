import numpy as np
from rdp import rdp
import math

from .draw_sketch import draw_sketch_cairo


def load_stroke_data(all_strokes, parts_used, padding=0, nodetail=True, no_initial=False):
    '''
    all_strokes: component list => curve list => stroke list => point list (4), ['current x', 'current y', 'next x', 'next y']
    '''
    max_dim = 512
    vector_image = []  # component list => curve list => stroke list (N, 2)
    parts_used_final = []
    x_max = y_max = 0
    for ci, component in enumerate(all_strokes):
        if no_initial and ci == 0 and parts_used[ci] == 'initial':
            continue
        if nodetail and ci == len(all_strokes) - 1 and parts_used[ci] == 'details':
            continue

        curve_list = []
        for curve in component:
            if len(curve) == 0:  # skip the empty curve
                # component_list.append([])
                continue
            stroke_list = np.array([curve[0][:2]] + [stroke[2:4] for stroke in curve])  # add each curve (N, 2)
            ## add padding
            pad_scaling = max_dim / (max_dim + 2.0 * padding)
            stroke_list_pad = stroke_list * pad_scaling
            stroke_list_pad += padding
            curve_list.append(stroke_list_pad)
            x_max_stroke, y_max_stroke = np.max(stroke_list_pad, 0)
            x_max = max(x_max, x_max_stroke)
            y_max = max(y_max, y_max_stroke)
        vector_image.append(curve_list)  # for each component
        parts_used_final.append(parts_used[ci])

    assert x_max <= max_dim, x_max
    assert y_max <= max_dim, y_max
    return vector_image, parts_used_final


def show_stroke_data_statistics(stroke_data, parts_used):
    print(' >>', len(stroke_data), 'components')
    total_curves = 0
    for ci, component in enumerate(stroke_data):
        print(' >>> components:', parts_used[ci], '|', len(component), 'curves')
        for curve in component:
            total_curves += 1
            print(' >>>>', curve.shape)
    print(' >>>>> total_curves:', total_curves)


def point_dist(pa, pb):
    return math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)


def adaptiva_rdp(curve_ori, epsilon=0.5, min_len=10):
    '''
    curve_ori: (N, 2)
    '''
    curve = rdp(curve_ori, epsilon=epsilon)
    if len(curve) == 2:
        return curve

    start_idx = -1
    new_curve = []
    for i in range(len(curve)):
        curr_p = curve[i]
        if i == len(curve) - 1:
            if start_idx != -1:
                end_idx = i
                sub_curve = curve[start_idx:end_idx + 1]
                start_idx = -1
                new_curve += adaptiva_rdp(sub_curve, epsilon + 0.1, min_len)
            else:
                new_curve.append(curr_p)
        else:
            next_p = curve[i + 1]
            if point_dist(curr_p, next_p) < min_len:
                if start_idx == -1:
                    start_idx = i
            else:
                if start_idx != -1:
                    end_idx = i
                    sub_curve = curve[start_idx:end_idx + 1]
                    start_idx = -1
                    new_curve += adaptiva_rdp(sub_curve, epsilon + 0.1, min_len)
                else:
                    new_curve.append(curr_p)

    assert start_idx == -1
    return new_curve


def stroke_simplification(stroke_data, parts_used, epsilon=0.5, is_adaptive=False):
    # print('###### stroke_simplification')
    # print(' >>', len(stroke_data), 'components')

    vector_image = []
    for ci, component in enumerate(stroke_data):
        # print(' >>> components:', parts_used[ci], '|', len(component), 'curves')

        curve_list = []
        for curve in component:
            if not is_adaptive:
                curve_simp = rdp(curve, epsilon=epsilon)
            else:
                curve_simp = adaptiva_rdp(curve.tolist(), epsilon=epsilon)
            assert len(curve_simp) > 1
            curve_list.append(np.array(curve_simp))
            # print(' >>>>', curve.shape, '=>', curve_simp.shape)
        vector_image.append(curve_list)
    return vector_image


def polyline_to_bezier(stroke_data, parts_used):
    # print('###### polyline_to_bezier')
    # print(' >>', len(stroke_data), 'components')
    vector_image = []
    for ci, component in enumerate(stroke_data):
        # print(' >>> components:', parts_used[ci], '|', len(component), 'curves')
        curve_list = []
        for curve in component:  # curve: (N, 2)
            stroke_list = []  # list of (4, 2)

            total_point = len(curve)
            curr_idx = 0

            while curr_idx <= total_point - 4:
                stroke_list.append(curve[curr_idx: curr_idx+4])
                curr_idx += 3

            if total_point - curr_idx == 1:
                assert len(stroke_list) > 0
            elif total_point - curr_idx == 2:
                if len(stroke_list) > 0:
                    stroke_list[-1][-1] = curve[-1]
                else:
                    start_point = curve[curr_idx]
                    end_point = curve[curr_idx + 1]
                    ctrl_point1 = start_point / 3.0 * 2.0 + end_point / 3.0
                    ctrl_point2 = start_point / 3.0 + end_point / 3.0 * 2.0
                    last_stroke = np.stack([start_point, ctrl_point1, ctrl_point2, end_point], axis=0)
                    stroke_list.append(last_stroke)
            elif total_point - curr_idx == 3:
                start_point = curve[curr_idx]
                intermediate_point = curve[curr_idx+1]
                end_point = curve[curr_idx+2]
                ctrl_point1 = start_point / 3.0 + intermediate_point / 3.0 * 2.0
                ctrl_point2 = intermediate_point / 3.0 * 2.0 + end_point / 3.0
                last_stroke = np.stack([start_point, ctrl_point1, ctrl_point2, end_point], axis=0)
                stroke_list.append(last_stroke)
            else:
                raise Exception('error')

            stroke_list = np.stack(stroke_list, axis=0)  # (N, 4, 2)
            # print(' >>>>', curve.shape, '=>', stroke_list.shape)
            curve_list.append(stroke_list.tolist())
        vector_image.append(curve_list)
    return vector_image


# TODO: can remove this checking finally
def deep_shape_check(original_data, new_data):
    '''
    original_data: component list => curve list => stroke list (N, 2)
    new_data: component list => curve list => stroke list (N, 2)
    '''
    if type(original_data) is list:
        assert type(new_data) is list, type(new_data)
        assert len(original_data) == len(new_data), (len(original_data), len(new_data))
        if len(original_data) == 2 and type(original_data[0]) is float:
            return
        if len(original_data) == 4 and type(original_data[0]) is list and type(original_data[0][0]) is float:
            return
        for i in range(len(original_data)):
            deep_shape_check(original_data[i], new_data[i])
    elif type(original_data) is np.ndarray:
        # assert type(new_data) is np.ndarray, type(new_data)
        assert original_data.shape == np.array(new_data).shape, (original_data.shape, np.array(new_data).shape)
    else:
        raise Exception('Unknown type:', type(original_data))


def cal_line_mask(curve_points, raster_size, line_thickness=3):
    sketch_img = draw_sketch_cairo([[curve_points]], outpath=None, is_bezier=True,
                                   part_label=False, clip_curve=True,
                                   side=raster_size, line_diameter=line_thickness,
                                   bg_color=(1, 1, 1), fg_color=(0, 0, 0))

    line_mask = np.array(sketch_img, dtype=np.float32)  # [0-stroke, 255-BG]
    line_mask = 1.0 - line_mask / 255.0  # (H, W), [0-BG, 1-stroke]
    return line_mask
