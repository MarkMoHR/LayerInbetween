import os
import numpy as np
import jsonlines
from PIL import Image

from .preprocessing import cal_line_mask
from .draw_sketch import draw_sketch_cairo


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

def translate_component(curve_list, flow, raster_size, line_thickness=3):
    '''
    curve_list: list of (N', 4, 2)
    flow: (H, W, 2)
    '''
    curve_list_trans = []
    line_mask_list = []
    masked_flow_list = []

    for curve_i, curve_points in enumerate(curve_list):  # (N', 4, 2)
        line_mask = cal_line_mask(curve_points, raster_size, line_thickness)  # (H, W), [0-BG, 1-stroke]
        masked_flow = flow * np.expand_dims(line_mask, axis=-1)  # (H, W, 2)
        line_mask_list.append(line_mask)
        masked_flow_list.append(masked_flow)

    line_mask_list = np.stack(line_mask_list, axis=0)  # (K', H, W), [0-BG, 1-stroke]
    masked_flow_list = np.stack(masked_flow_list, axis=0)  # (K', H, W, 2)

    avg_offset = np.sum(masked_flow_list, axis=(0, 1, 2)) / np.sum(line_mask_list)  # (2), [dx, dy]
    for curve_i, curve_points in enumerate(curve_list):  # (N', 4, 2), [x, y]
        curve_points_shift = np.array(curve_points) + np.expand_dims(np.expand_dims(avg_offset, axis=0), axis=0)
        curve_list_trans.append(curve_points_shift)

    return curve_list_trans, avg_offset


def translate_curve(stroke_list, flow, raster_size, line_thickness=3):
    '''
    stroke_list: (N', 4, 2)
    flow: (H, W, 2)
    '''
    line_mask = cal_line_mask(stroke_list, raster_size, line_thickness)  # (H, W), [0-BG, 1-stroke]
    masked_flow = flow * np.expand_dims(line_mask, axis=-1)  # (H, W, 2)

    avg_offset = np.sum(masked_flow, axis=(0, 1)) / np.sum(line_mask)  # (2), [dx, dy]
    stroke_list_trans = np.array(stroke_list) + np.expand_dims(np.expand_dims(avg_offset, axis=0), axis=0)
    return stroke_list_trans, avg_offset


def component_translation_with_optical_flow(data_base, do_real, line_thickness, optical_flow_method, use_distance_transform, process_img_id,
                                            data_extra=None, gen_time=None):
    black_threshold = 200

    vector_data_dir = os.path.join(data_base, 'vector-params')

    if data_extra is not None:
        optical_flow_method = data_extra[str(process_img_id)]["optical_flow_method"]
        use_distance_transform = data_extra[str(process_img_id)]["use_distance_transform"]
    else:
        assert optical_flow_method is not None
        assert use_distance_transform is not None
    optical_flow_dir = optical_flow_data_map[optical_flow_method]['optical_flow_dir']
    distance_transform_factor = optical_flow_data_map[optical_flow_method]['distance_transform_factor']
    if use_distance_transform:
        optical_flow_dir += '-[DT-' + str(distance_transform_factor) + ']'
    optical_flow_base = os.path.join(data_base, 'optical_flow', optical_flow_dir, 'flow')

    translated_save_based = os.path.join(data_base, 'optical_flow', optical_flow_dir, 'warped_vector')
    translated_param_save_based = os.path.join(data_base, 'optical_flow', optical_flow_dir, 'component_offset')

    os.makedirs(translated_save_based, exist_ok=True)
    os.makedirs(translated_param_save_based, exist_ok=True)

    vector_data_path_ref = os.path.join(vector_data_dir, str(process_img_id) + '_ref.jsonl')
    with open(vector_data_path_ref, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b = item['stroke_params']  # component list => curve list => stroke list (N', 4, 2)
            # parts_data = item['component_part']

    if data_extra is not None:
        data_base_extra = "outputs/stroke_correspondence_results"
        if gen_time > 0:
            data_base_extra += '-Gen%d' % gen_time
        vector_data_path_ref_extra = os.path.join(data_base_extra, 'params', 'tar_pred-' + str(process_img_id) + '.jsonl')
        with open(vector_data_path_ref_extra, "r+") as f:
            for item in jsonlines.Reader(f):
                stroke_data_b_extra = item['stroke_params']  # component list => curve list => stroke list (N', 4, 2)

    if not do_real:
        ref_img_path = os.path.join(data_base, 'raster_black', 'sketch_' + str(process_img_id) + '_bezier-ref.png')
        tar_img_path = os.path.join(data_base, 'raster_black', 'sketch_' + str(process_img_id) + '_bezier-tar.png')
    else:
        ref_img_path = os.path.join(data_base, 'raster_black', str(process_img_id) + '_ref.png')
        tar_img_path = os.path.join(data_base, 'raster_black', str(process_img_id) + '_tar.png')
    flow_path = os.path.join(optical_flow_base, 'flow-' + str(process_img_id) + '.npz')

    reference_img = Image.open(ref_img_path).convert('RGB')
    image_size = reference_img.height

    npz = np.load(flow_path, encoding='latin1', allow_pickle=True)
    flow = npz['flow_mat']  # (H, W, 2)

    translated_params_save_path = os.path.join(translated_param_save_based, str(process_img_id) + '.jsonl')
    if os.path.exists(translated_params_save_path):
        os.remove(translated_params_save_path)

    stroke_data_b_trans = []  # component list => curve list => stroke list (N', 4, 2)
    for c_i in range(len(stroke_data_b)):
        # component_name = parts_data[c_i]
        curve_list = stroke_data_b[c_i]  # list of (N', 4, 2)
        if data_extra is not None:
            curve_list += stroke_data_b_extra[c_i]
        curve_list_trans, component_offset = translate_component(curve_list, flow, image_size)
        # component_offset: (2), [dx, dy], in image size
        stroke_data_b_trans.append(curve_list_trans)

        transform_params_data = {}
        transform_params_data['component_index'] = c_i
        transform_params_data['component_offset'] = component_offset.tolist()
        with jsonlines.open(translated_params_save_path, mode='a') as json_writer:
            json_writer.write(transform_params_data)

    warped_img = draw_sketch_cairo(stroke_data_b_trans, None, is_bezier=True,
                                    part_label=False,
                                    side=image_size, line_diameter=line_thickness,
                                    bg_color=(1, 1, 1), fg_color=(0, 0, 0))
    warped_img = np.array(warped_img, dtype=np.float32)  # (H, W), [0-stroke, 255-BG]
    warped_img = np.tile(np.expand_dims(warped_img, axis=-1), (1, 1, 3))

    target_img = Image.open(tar_img_path).convert('RGB')
    target_img = np.array(target_img, dtype=np.uint8)[:, :, 0]  # (H, W), [0-stroke, 255-BG]

    target_trans_mask = (warped_img < black_threshold).any(-1)
    warped_img_bg = np.expand_dims(target_img, axis=-1).astype(np.float32)
    warped_img_bg = np.concatenate(
        [np.ones_like(warped_img_bg) * 255,
            warped_img_bg,
            np.ones_like(warped_img_bg) * 255], axis=-1)
    warped_img_bg = 255 - (255 - warped_img_bg) * 0.7
    warped_img_bg[target_trans_mask] = warped_img[target_trans_mask]

    warped_img_bg_png = Image.fromarray(warped_img_bg.astype(np.uint8), 'RGB')
    save_path = os.path.join(translated_save_based, 'sketch_' + str(process_img_id) + '.png')
    warped_img_bg_png.save(save_path, 'PNG')


def curve_translation_with_optical_flow(data_base, line_thickness, optical_flow_method, use_distance_transform, process_img_id):
    black_threshold = 200

    optical_flow_dir = optical_flow_data_map[optical_flow_method]['optical_flow_dir']
    # use_distance_transform = optical_flow_data_map[optical_flow_method]['use_distance_transform']
    distance_transform_factor = optical_flow_data_map[optical_flow_method]['distance_transform_factor']
    if use_distance_transform:
        optical_flow_dir += '-[DT-' + str(distance_transform_factor) + ']'
    optical_flow_base = os.path.join(data_base, 'optical_flow', optical_flow_dir, 'flow')

    translated_save_based = os.path.join(data_base, 'optical_flow', optical_flow_dir, 'warped_vector_curve')
    os.makedirs(translated_save_based, exist_ok=True)

    vector_data_dir = os.path.join(data_base, 'vector-params-split')
    vector_data_path_ref = os.path.join(vector_data_dir, str(process_img_id) + '_ref.jsonl')
    with open(vector_data_path_ref, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b = item['stroke_params']  # curve list => stroke list (N', 4, 2)

    ref_img_path = os.path.join(data_base, 'raster_black', str(process_img_id) + '_ref.png')
    tar_img_path = os.path.join(data_base, 'raster_black', str(process_img_id) + '_tar.png')
    flow_path = os.path.join(optical_flow_base, 'flow-' + str(process_img_id) + '.npz')

    npz = np.load(flow_path, encoding='latin1', allow_pickle=True)
    flow = npz['flow_mat']  # (H, W, 2)

    ref_img = Image.open(ref_img_path)
    image_size = ref_img.height

    stroke_data_b_trans = []  # curve list => stroke list (N', 4, 2)
    for curve_i in range(len(stroke_data_b)):
        stroke_list = stroke_data_b[curve_i]  # (N', 4, 2)
        stroke_list_trans, curve_offset = translate_curve(stroke_list, flow, image_size)
        # curve_offset: (2), [dx, dy], in image size
        stroke_data_b_trans.append(stroke_list_trans)

    warped_img = draw_sketch_cairo([stroke_data_b_trans], None, is_bezier=True,
                                    part_label=False,
                                    side=image_size, line_diameter=line_thickness,
                                    bg_color=(1, 1, 1), fg_color=(0, 0, 0))
    warped_img = np.array(warped_img, dtype=np.float32)  # (H, W), [0-stroke, 255-BG]
    warped_img = np.tile(np.expand_dims(warped_img, axis=-1), (1, 1, 3))

    target_img = Image.open(tar_img_path).convert('RGB')
    target_img = np.array(target_img, dtype=np.uint8)[:, :, 0]  # (H, W), [0-stroke, 255-BG]

    target_trans_mask = (warped_img < black_threshold).any(-1)
    warped_img_bg = np.expand_dims(target_img, axis=-1).astype(np.float32)
    warped_img_bg = np.concatenate(
        [np.ones_like(warped_img_bg) * 255,
            warped_img_bg,
            np.ones_like(warped_img_bg) * 255], axis=-1)
    warped_img_bg = 255 - (255 - warped_img_bg) * 0.7
    warped_img_bg[target_trans_mask] = warped_img[target_trans_mask]

    warped_img_bg_png = Image.fromarray(warped_img_bg.astype(np.uint8), 'RGB')
    save_path = os.path.join(translated_save_based, 'sketch_' + str(process_img_id) + '.png')
    warped_img_bg_png.save(save_path, 'PNG')
