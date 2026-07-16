import os
import random
import numpy as np
from PIL import Image
import jsonlines

import torch

# import pydiffvg

from hparam import HParams
from dataset_utils.common import load_txt_ids


def copy_hparams(hparams):
    """Return a copy of an HParams instance."""
    return HParams(**hparams.values())


class LineDataLoader(object):
    def __init__(self,
                 dataset_base,
                 batch_size,
                 window_size_scaling,
                 window_size_min,
                 window_size_scaling_comp,
                 window_size_min_comp,
                 use_real_endpoints,
                 transform_model_name,
                 use_optical_flow,
                 is_train):
        self.dataset_base = dataset_base
        self.batch_size = batch_size
        self.window_size_scaling = window_size_scaling
        self.window_size_min = window_size_min
        self.window_size_scaling_comp = window_size_scaling_comp
        self.window_size_min_comp = window_size_min_comp
        self.use_real_endpoints = use_real_endpoints
        self.transform_model_name = transform_model_name
        self.use_optical_flow = use_optical_flow
        self.is_train = is_train
        self.dataset_names = ['creature', 'bird']
        self.ref_tar_split_names = ['ref', 'tar']
        self.dataset_split = 'train' if is_train else 'val'

        self.img_ids = self.get_img_ids()
        self.example_num = len(self.img_ids)
        print('Loaded', self.dataset_split, ':', self.example_num)

        ## Load invalid component ids
        outsider_img_comp_ids_list_path = os.path.join(self.dataset_base, 'transform_invalid_comp_ids', 'out-of-bound',
                                                       self.dataset_split + '-win=' + str(self.window_size_scaling_comp) + '-min=' + str(self.window_size_min_comp) + '.txt')
        outsider_img_comp_ids_list = load_txt_ids(outsider_img_comp_ids_list_path)
        invalid_occ_img_comp_ids_list_path = os.path.join(self.dataset_base, 'transform_invalid_comp_ids', 'occlusion',
                                                          self.dataset_split + '_invalid.txt')
        invalid_occ_img_comp_ids_list = load_txt_ids(invalid_occ_img_comp_ids_list_path)
        single_stroke_comp_ids_list_path = os.path.join(self.dataset_base, 'transform_invalid_comp_ids', 'single-stroke-component',
                                                        self.dataset_split + '_invalid.txt')
        single_stroke_comp_ids_list = load_txt_ids(single_stroke_comp_ids_list_path)
        invalid_img_comp_ids_list = outsider_img_comp_ids_list + invalid_occ_img_comp_ids_list + single_stroke_comp_ids_list
        self.invalid_img_comp_ids_list = list(set(invalid_img_comp_ids_list))

        self.valid_endpoint_index_buffer = []

    def get_img_ids(self):
        img_ids = []
        for dataset_name in self.dataset_names:
            vector_data_dir = os.path.join(self.dataset_base, dataset_name + '_512', self.dataset_split, 'vector-params')
            all_files = os.listdir(vector_data_dir)
            all_files = [item for item in all_files if '_ref.jsonl' in item]

            for filename in all_files:
                img_index = filename[:filename.find('_')]
                img_ids.append(dataset_name + '-' + img_index)
        img_ids.sort()
        return img_ids

    def get_valid_img_endpoints(self, dataset_name, image_index, reference_stroke_data, occluded_only=False):
        ## TODO: For eval with a common dataset
        transform_model_names = ['FAD3-T12-2.0x-51-min=64']  # FAD-2.0x-51 / FAD2-2.0x-51-v3 / FAD3-2.0x-51-min=64 / FAD3-T12-2.0x-51-min=64

        if self.is_train:
            out_of_bound_txt_path = os.path.join(self.dataset_base, dataset_name + '_512', self.dataset_split, 'endpoint_type_ids',
                                                 'win=' + str(self.window_size_scaling) + '-min=' + str(self.window_size_min),
                                                 str(image_index), 'out_of_bound_with_transform-[' + self.transform_model_name + '].txt')
            out_of_bound_endpoint_ids = load_txt_ids(out_of_bound_txt_path)
            stroke_outside_txt_path = os.path.join(self.dataset_base, dataset_name + '_512', self.dataset_split, 'endpoint_type_ids',
                                                   'win=' + str(self.window_size_scaling) + '-min=' + str(self.window_size_min),
                                                   str(image_index), 'stroke_outside_with_transform-[' + self.transform_model_name + '].txt')
            stroke_outside_endpoint_ids = load_txt_ids(stroke_outside_txt_path)
        else:
            out_of_bound_endpoint_ids = []
            stroke_outside_endpoint_ids = []
            for transform_model_name in transform_model_names:
                out_of_bound_txt_path = os.path.join(self.dataset_base, dataset_name + '_512', self.dataset_split, 'endpoint_type_ids',
                                                     'win=' + str(self.window_size_scaling) + '-min=' + str(self.window_size_min), str(image_index),
                                                     'out_of_bound_with_transform-[' + transform_model_name + '].txt')
                out_of_bound_endpoint_ids += load_txt_ids(out_of_bound_txt_path)
                stroke_outside_txt_path = os.path.join(self.dataset_base, dataset_name + '_512', self.dataset_split, 'endpoint_type_ids',
                                                       'win=' + str(self.window_size_scaling) + '-min=' + str(self.window_size_min), str(image_index),
                                                       'stroke_outside_with_transform-[' + transform_model_name + '].txt')
                stroke_outside_endpoint_ids += load_txt_ids(stroke_outside_txt_path)
            out_of_bound_endpoint_ids = list(set(out_of_bound_endpoint_ids))
            stroke_outside_endpoint_ids = list(set(stroke_outside_endpoint_ids))

        invalid_occlusion_txt_path = os.path.join(self.dataset_base, dataset_name + '_512', self.dataset_split, 'endpoint_type_ids',
                                                  'win=-1', str(image_index), 'invalid_occlusion.txt')
        invalid_occlusion_endpoint_ids = load_txt_ids(invalid_occlusion_txt_path)
        valid_occlusion_txt_path = os.path.join(self.dataset_base, dataset_name + '_512', self.dataset_split, 'endpoint_type_ids',
                                                'win=-1', str(image_index), 'valid_occlusion.txt')
        valid_occlusion_endpoint_ids = load_txt_ids(valid_occlusion_txt_path)

        img_id = dataset_name + '-' + str(image_index) + '-'
        invalid_img_comp_ids = [item for item in self.invalid_img_comp_ids_list if img_id in item]
        invalid_comp_indices = [int(item[item.find(img_id) + len(img_id):]) for item in invalid_img_comp_ids]

        valid_img_endpoint_ids = []
        for c_i in range(len(reference_stroke_data)):
            if c_i in invalid_comp_indices:
                continue

            curve_b_list = reference_stroke_data[c_i]  # list of (N', 4, 2)
            for curve_i in range(len(curve_b_list)):
                curve_b_points = curve_b_list[curve_i]  # list (N') of (4, 2)
                stroke_num = len(curve_b_points)
                for point_index in range(stroke_num + 1):
                    if self.use_real_endpoints and int(point_index) != 0 and int(point_index) != stroke_num:
                        continue

                    endpoint_id = "%s_%s_%s" % (c_i, curve_i, point_index)
                    if endpoint_id in out_of_bound_endpoint_ids or \
                            endpoint_id in stroke_outside_endpoint_ids or \
                            endpoint_id in invalid_occlusion_endpoint_ids:
                        continue
                    if occluded_only and endpoint_id not in valid_occlusion_endpoint_ids:
                        continue
                    valid_img_endpoint_ids.append(endpoint_id)

        valid_img_endpoint_ids.sort()
        return valid_img_endpoint_ids

    def load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.float32)  # (H, W, 3), [0.0-strokes, 255.0-BG]
        image = image[:, :, 0] / 255.0  # (H, W), [0.0-strokes, 1.0-BG]
        return image

    def load_stroke_parameter(self, vector_data_path):
        stroke_data_b_list = []
        parts_data_list = []
        with open(vector_data_path, "r+") as f:
            for item in jsonlines.Reader(f):
                stroke_data_b = item['stroke_params']
                parts_data = item['component_part']
                stroke_data_b_list.append(stroke_data_b)
                parts_data_list.append(parts_data)
        assert len(stroke_data_b_list) == 1
        assert len(parts_data_list) == 1
        return stroke_data_b_list[0]

    def load_transform_parameter(self, transform_params_path):
        transform_params_data = {}
        with open(transform_params_path, "r+") as f:
            for item in jsonlines.Reader(f):
                c_idx = item['component_index']
                transform_params_data[c_idx] = {}
                transform_params_data[c_idx]['component_center'] = item['component_center']  # (2), [0.0, 1.0], relative to image size
                transform_params_data[c_idx]['component_win_size'] = item['component_win_size']  # (2), in image size
                transform_params_data[c_idx]['pred_cursor'] = item['pred_cursor']  # (2), [0.0, 1.0], relative to image size
                transform_params_data[c_idx]['pred_window_size'] = item['pred_window_size']  # (2), in image size
                transform_params_data[c_idx]['pred_rotate_angle'] = item['pred_rotate_angle']  # (), [-180.0, 180.0]
                transform_params_data[c_idx]['pred_shear_x_angle'] = item['pred_shear_x_angle']  # (), [-90.0, 90.0]
                transform_params_data[c_idx]['pred_shear_y_angle'] = item['pred_shear_y_angle']  # (), [-90.0, 90.0]
        return transform_params_data

    def process_stroke_parameter(self, parameters_ref, parameters_tar, comp_index, curve_index, stroke_index, image_size):
        '''
        parameters_ref / parameters_tar: component list => curve list => stroke list (N', 4, 2)
        '''
        curve_points_ref = parameters_ref[comp_index][curve_index]  # (N', 4, 2)
        curve_points_tar = parameters_tar[comp_index][curve_index]  # (N', 4, 2)

        if stroke_index == 0:
            p_prev = curve_points_ref[stroke_index][0]  # (2)
            p_curr = curve_points_ref[stroke_index][0]
            p_curr_tar = curve_points_tar[stroke_index][0]
            p_next = curve_points_ref[stroke_index][-1]
        elif stroke_index >= len(curve_points_ref):
            p_prev = curve_points_ref[stroke_index - 1][0]  # (2)
            p_curr = curve_points_ref[stroke_index - 1][-1]
            p_curr_tar = curve_points_tar[stroke_index - 1][-1]
            p_next = curve_points_ref[stroke_index - 1][-1]
        else:
            p_prev = curve_points_ref[stroke_index - 1][0]  # (2)
            p_curr = curve_points_ref[stroke_index - 1][-1]
            p_curr_tar = curve_points_tar[stroke_index - 1][-1]
            p_next = curve_points_ref[stroke_index][-1]

        window_size_dist1 = np.abs(np.array(p_prev) - np.array(p_curr))  # (2), full size
        window_size_dist2 = np.abs(np.array(p_curr) - np.array(p_next))  # (2), full size
        window_size_dist = np.concatenate([window_size_dist1, window_size_dist2], axis=-1)  # (4), full size
        window_size = np.max(window_size_dist, axis=-1) * 2.0  # (), full size
        window_size_norm = window_size / float(image_size)  # (), [0.0, 1.0]

        window_size_scaled = window_size * self.window_size_scaling
        window_size_scaled = min(max(window_size_scaled, self.window_size_min), image_size * 1.5)

        centerpoint_ref = np.array(p_curr, dtype=np.float32)  # (2), full size
        centerpoint_tar = np.array(p_curr_tar, dtype=np.float32)  # (2), full size

        centerpoint_ref_norm = centerpoint_ref / float(image_size)  # (2), [0.0, 1.0]
        endpoint_tar_rel = (centerpoint_tar - centerpoint_ref) / (window_size_scaled / 2.0)  # (2), [-1.0, 1.0]

        return centerpoint_ref_norm, endpoint_tar_rel, window_size_norm

    def get_batch(self, use_cuda, batch_idx=None, all_example=False, batch_idx_offset=0, occluded_only=False):
        reference_image_batch = []
        reference_component_batch = []
        reference_stroke_batch = []
        target_image_batch = []
        reference_centerpoints_batch = []
        reference_centerpoints_offset_batch = []
        target_endpoint_offset_gt_batch = []
        base_window_size_batch = []
        image_id_batch = []
        endpoint_id_batch = []

        component_centerpoints_batch = []
        component_win_size_batch = []
        target_transform_cursor_batch = []
        target_transform_win_size_batch = []
        target_transform_angle_batch = []
        target_transform_shear_x_batch = []
        target_transform_shear_y_batch = []

        if self.is_train:
            selected_indices = np.random.choice(np.arange(self.example_num), size=self.batch_size, replace=False)
        else:
            selected_indices = [self.batch_size * batch_idx + i + batch_idx_offset for i in range(self.batch_size)]

        for batch_i in range(len(selected_indices)):
            selected_id = self.img_ids[selected_indices[batch_i]]  # 'bird-1' or 'creature-230'
            selected_dataset_name = selected_id[:selected_id.find('-')]
            selected_index = selected_id[selected_id.find('-') + 1:]
            image_id_batch.append(selected_id)

            reference_image_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split, 'raster_black', 'sketch_' + str(selected_index) + '_bezier-' + self.ref_tar_split_names[0] + '.png')
            reference_stroke_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split, 'vector-params', str(selected_index) + '_' + self.ref_tar_split_names[0] + '.jsonl')
            target_image_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split, 'raster_black', 'sketch_' + str(selected_index) + '_bezier-' + self.ref_tar_split_names[1] + '.png')
            target_stroke_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split, 'vector-params', str(selected_index) + '_' + self.ref_tar_split_names[1] + '.jsonl')

            reference_image = self.load_image(reference_image_path)  # (H, W), [0.0-strokes, 1.0-BG]
            target_image = self.load_image(target_image_path)  # (H, W), [0.0-strokes, 1.0-BG]
            reference_stroke_data = self.load_stroke_parameter(reference_stroke_path)
            target_stroke_data = self.load_stroke_parameter(target_stroke_path)
            # reference_stroke_data / target_stroke_data: component list => curve list => stroke list (N', 4, 2)

            image_size = reference_image.shape[0]

            valid_endpoint_ids = self.get_valid_img_endpoints(selected_dataset_name, selected_index, reference_stroke_data,
                                                              occluded_only=occluded_only)
            if not occluded_only:
                assert len(valid_endpoint_ids) > 0
            else:
                if len(valid_endpoint_ids) == 0:
                    return None

            transform_global_model_name_plus = self.transform_model_name + '-[c_min=' + str(self.window_size_min_comp) + ']'
            if self.use_optical_flow:
                transform_global_model_name_plus += '-[optical]'
            transform_params_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split,
                                                 'component_transform_params', transform_global_model_name_plus, selected_index + '.jsonl')
            transform_params_data = self.load_transform_parameter(transform_params_path)

            if self.is_train:
                random.shuffle(valid_endpoint_ids)
                random_endpoint_ids = [valid_endpoint_ids[0]]
            else:
                if not all_example:
                    if len(self.valid_endpoint_index_buffer) <= batch_idx:
                        random.shuffle(valid_endpoint_ids)
                        random_endpoint_ids = [valid_endpoint_ids[0]]
                        self.valid_endpoint_index_buffer.append(random_endpoint_ids[0])
                    else:
                        random_endpoint_ids = [self.valid_endpoint_index_buffer[batch_idx]]
                else:
                    random_endpoint_ids = [item for item in valid_endpoint_ids]

            for random_endpoint_id in random_endpoint_ids:
                comp_curve_point = random_endpoint_id.split('_')
                c_i, curve_i, point_index = comp_curve_point
                endpoint_id_batch.append(random_endpoint_id)

                reference_component_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split, 'raster_black_component',
                                                        str(selected_index), 'component_' + str(c_i) + '-' + self.ref_tar_split_names[0] + '.png')
                reference_component_image = self.load_image(reference_component_path)  # (H, W), [0.0-strokes, 1.0-BG]

                reference_stroke_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split, 'raster_black_endpoint_stroke',
                                                     str(selected_index), 'endpoint_' + random_endpoint_id + '.png')
                reference_stroke_image = self.load_image(reference_stroke_path)  # (H, W), [0.0-strokes, 1.0-BG]

                centerpoint, endpoint_offset_gt, window_size = self.process_stroke_parameter(reference_stroke_data, target_stroke_data, int(c_i), int(curve_i), int(point_index), image_size)
                # centerpoints: (2), [0.0, 1.0]
                # endpoint_offset_gt: (2), [-1.0, 1.0]
                # window_sizes: (), [0.0, 1.0]

                centerpoint_offset = np.maximum(np.minimum(centerpoint, 1.0), 0.0)

                reference_image_batch.append(reference_image)
                reference_component_batch.append(reference_component_image)
                reference_stroke_batch.append(reference_stroke_image)
                target_image_batch.append(target_image)
                reference_centerpoints_batch.append(centerpoint)
                reference_centerpoints_offset_batch.append(centerpoint_offset)
                target_endpoint_offset_gt_batch.append(endpoint_offset_gt)
                base_window_size_batch.append(window_size)
                component_centerpoints_batch.append(transform_params_data[int(c_i)]['component_center'])
                component_win_size_batch.append(transform_params_data[int(c_i)]['component_win_size'])
                target_transform_cursor_batch.append(transform_params_data[int(c_i)]['pred_cursor'])
                target_transform_win_size_batch.append(transform_params_data[int(c_i)]['pred_window_size'])
                target_transform_angle_batch.append(transform_params_data[int(c_i)]['pred_rotate_angle'])
                target_transform_shear_x_batch.append(transform_params_data[int(c_i)]['pred_shear_x_angle'])
                target_transform_shear_y_batch.append(transform_params_data[int(c_i)]['pred_shear_y_angle'])

        reference_image_batch = np.expand_dims(np.stack(reference_image_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        reference_component_batch = np.expand_dims(np.stack(reference_component_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        reference_stroke_batch = np.expand_dims(np.stack(reference_stroke_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        target_image_batch = np.expand_dims(np.stack(target_image_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        reference_centerpoints_batch = np.expand_dims(np.stack(reference_centerpoints_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0]
        reference_centerpoints_offset_batch = np.expand_dims(np.stack(reference_centerpoints_offset_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0]
        target_endpoint_offset_gt_batch = np.expand_dims(np.stack(target_endpoint_offset_gt_batch, axis=0), axis=1)  # (N, 1, 2), [-1.0, 1.0]
        base_window_size_batch = np.expand_dims(np.stack(base_window_size_batch, axis=0), axis=-1)  # (N, 1), [0.0, 1.0]
        component_centerpoints_batch = np.expand_dims(np.stack(component_centerpoints_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0], relative to image size
        component_win_size_batch = np.expand_dims(np.stack(component_win_size_batch, axis=0), axis=1)  # (N, 1, 2), in image size
        target_transform_cursor_batch = np.expand_dims(np.stack(target_transform_cursor_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0], relative to image size
        target_transform_win_size_batch = np.expand_dims(np.stack(target_transform_win_size_batch, axis=0), axis=1)  # (N, 1, 2), in image size
        target_transform_angle_batch = np.expand_dims(np.stack(target_transform_angle_batch, axis=0), axis=1)  # (N, 1), [-180.0, 180.0]
        target_transform_shear_x_batch = np.expand_dims(np.stack(target_transform_shear_x_batch, axis=0), axis=1)  # (N, 1), [-90.0, 90.0]
        target_transform_shear_y_batch = np.expand_dims(np.stack(target_transform_shear_y_batch, axis=0), axis=1)  # (N, 1), [-90.0, 90.0]

        ## convert to tensor
        reference_image_batch = torch.tensor(reference_image_batch).float()
        reference_component_batch = torch.tensor(reference_component_batch).float()
        reference_stroke_batch = torch.tensor(reference_stroke_batch).float()
        target_image_batch = torch.tensor(target_image_batch).float()
        reference_centerpoints_batch = torch.tensor(reference_centerpoints_batch).float()
        reference_centerpoints_offset_batch = torch.tensor(reference_centerpoints_offset_batch).float()
        target_endpoint_offset_gt_batch = torch.tensor(target_endpoint_offset_gt_batch).float()
        base_window_size_batch = torch.tensor(base_window_size_batch).float()
        component_centerpoints_batch = torch.tensor(component_centerpoints_batch).float()
        component_win_size_batch = torch.tensor(component_win_size_batch).float()
        target_transform_cursor_batch = torch.tensor(target_transform_cursor_batch).float()
        target_transform_win_size_batch = torch.tensor(target_transform_win_size_batch).float()
        target_transform_angle_batch = torch.tensor(target_transform_angle_batch).float()
        target_transform_shear_x_batch = torch.tensor(target_transform_shear_x_batch).float()
        target_transform_shear_y_batch = torch.tensor(target_transform_shear_y_batch).float()

        if use_cuda:
            reference_image_batch = reference_image_batch.cuda()
            reference_component_batch = reference_component_batch.cuda()
            reference_stroke_batch = reference_stroke_batch.cuda()
            target_image_batch = target_image_batch.cuda()
            reference_centerpoints_batch = reference_centerpoints_batch.cuda()
            reference_centerpoints_offset_batch = reference_centerpoints_offset_batch.cuda()
            target_endpoint_offset_gt_batch = target_endpoint_offset_gt_batch.cuda()
            base_window_size_batch = base_window_size_batch.cuda()
            component_centerpoints_batch = component_centerpoints_batch.cuda()
            component_win_size_batch = component_win_size_batch.cuda()
            target_transform_cursor_batch = target_transform_cursor_batch.cuda()
            target_transform_win_size_batch = target_transform_win_size_batch.cuda()
            target_transform_angle_batch = target_transform_angle_batch.cuda()
            target_transform_shear_x_batch = target_transform_shear_x_batch.cuda()
            target_transform_shear_y_batch = target_transform_shear_y_batch.cuda()

        return reference_image_batch, reference_component_batch, reference_stroke_batch, target_image_batch, \
               reference_centerpoints_batch, reference_centerpoints_offset_batch, target_endpoint_offset_gt_batch, \
               base_window_size_batch, image_id_batch, endpoint_id_batch, \
               component_centerpoints_batch, component_win_size_batch, \
               target_transform_cursor_batch, target_transform_win_size_batch, target_transform_angle_batch, \
               target_transform_shear_x_batch, target_transform_shear_y_batch


def load_dataset(model_params, test_only=False, stroke_fixing=False):
    data_base = model_params.dataset_base

    valid_model_params = copy_hparams(model_params)
    valid_model_params.batch_size = 1  # only sample one at a time

    if not test_only:
        train_set = LineDataLoader(dataset_base=data_base, batch_size=model_params.batch_size,
                                   window_size_scaling=model_params.window_size_scaling_ref,
                                   window_size_min=model_params.window_size_min,
                                   window_size_scaling_comp=model_params.window_size_scaling_ref_comp,
                                   window_size_min_comp=model_params.window_size_min_comp,
                                   use_real_endpoints=model_params.use_real_endpoints,
                                   transform_model_name=model_params.transform_model_name,
                                   use_optical_flow=model_params.use_optical_flow,
                                   is_train=True)
    else:
        train_set = None
    val_set = LineDataLoader(dataset_base=data_base, batch_size=valid_model_params.batch_size,
                             window_size_scaling=valid_model_params.window_size_scaling_ref,
                             window_size_min=valid_model_params.window_size_min,
                             window_size_scaling_comp=valid_model_params.window_size_scaling_ref_comp,
                             window_size_min_comp=valid_model_params.window_size_min_comp,
                             use_real_endpoints=valid_model_params.use_real_endpoints,
                             transform_model_name=valid_model_params.transform_model_name,
                             use_optical_flow=valid_model_params.use_optical_flow,
                             is_train=False)

    result = [train_set, val_set, model_params, valid_model_params]
    return result
