import os
import random
import numpy as np
from PIL import Image
import jsonlines

import torch

# import pydiffvg

from hparam import HParams
from dataset_utils.common import load_txt_ids, load_txt_ids_info
from image_utils.image_processing import disturb_endpoint


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
                 transform_model_name,
                 transform_local_model_name,
                 use_optical_flow,
                 training_with_endpoint_disturb,
                 do_dataset_filtering,
                 is_train):
        self.dataset_base = dataset_base
        self.batch_size = batch_size
        self.window_size_scaling = window_size_scaling
        self.window_size_min = window_size_min
        self.window_size_scaling_comp = window_size_scaling_comp
        self.window_size_min_comp = window_size_min_comp
        self.transform_model_name = transform_model_name
        self.transform_local_model_name = transform_local_model_name
        self.use_optical_flow = use_optical_flow
        self.training_with_endpoint_disturb = training_with_endpoint_disturb
        self.do_dataset_filtering = do_dataset_filtering
        self.is_train = is_train
        self.dataset_names = ['creature', 'bird']
        self.ref_tar_split_names = ['ref', 'tar']
        self.dataset_split = 'train' if is_train else 'val'

        self.img_ids = self.get_img_ids()
        self.example_num = len(self.img_ids)
        print('Loaded', self.dataset_split, ':', self.example_num)

        if self.do_dataset_filtering:
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
        else:
            self.invalid_img_comp_ids_list = []

        self.valid_stroke_index_buffer = []

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

    def get_valid_img_ctrlpoints(self, dataset_name, image_index, reference_stroke_data, occluded_only=False):
        ## TODO: For eval with a common dataset
        if self.do_dataset_filtering:
            out_of_bound_txt_path = os.path.join(self.dataset_base, dataset_name + '_512', self.dataset_split, 'ctrlpoint_type_ids',
                                                 'win=' + str(self.window_size_scaling) + '-min=' + str(self.window_size_min),
                                                 str(image_index), 'out_of_bound.txt')
            out_of_bound_stroke_ids = load_txt_ids(out_of_bound_txt_path)
        else:
            out_of_bound_stroke_ids = []

        short_stroke_txt_path = os.path.join(self.dataset_base, dataset_name + '_512', self.dataset_split, 'ctrlpoint_type_ids',
                                             'win=-1', str(image_index), 'short_stroke.txt')
        short_stroke_ids = load_txt_ids(short_stroke_txt_path)

        valid_occlusion_state_txt_path = os.path.join(self.dataset_base, dataset_name + '_512', self.dataset_split, 'ctrlpoint_type_ids',
                                                'win=-1', str(image_index), 'valid_occlusion_state.txt')
        valid_occlusion_stroke_ids, valid_occlusion_state_map = load_txt_ids_info(valid_occlusion_state_txt_path)

        img_id = dataset_name + '-' + str(image_index) + '-'
        invalid_img_comp_ids = [item for item in self.invalid_img_comp_ids_list if img_id in item]
        invalid_comp_indices = [int(item[item.find(img_id) + len(img_id):]) for item in invalid_img_comp_ids]

        valid_img_stroke_ids = []
        valid_img_stroke_occlusion_state_map = {}
        for c_i in range(len(reference_stroke_data)):
            if c_i in invalid_comp_indices:
                continue

            curve_b_list = reference_stroke_data[c_i]  # list of (N', 4, 2)
            for curve_i in range(len(curve_b_list)):
                curve_b_points = curve_b_list[curve_i]  # list (N') of (4, 2)
                stroke_num = len(curve_b_points)
                for stroke_index in range(stroke_num):
                    stroke_id = "%s_%s_%s" % (c_i, curve_i, stroke_index)
                    if stroke_id in out_of_bound_stroke_ids or stroke_id in short_stroke_ids:
                        continue
                    if occluded_only and stroke_id not in valid_occlusion_stroke_ids:
                        continue
                    valid_img_stroke_ids.append(stroke_id)

                    if stroke_id not in valid_occlusion_stroke_ids:
                        valid_img_stroke_occlusion_state_map[stroke_id] = 0
                    else:
                        assert stroke_id in valid_occlusion_state_map.keys()
                        valid_img_stroke_occlusion_state_map[stroke_id] = valid_occlusion_state_map[stroke_id]

        valid_img_stroke_ids.sort()
        assert len(valid_img_stroke_ids) == len(valid_img_stroke_occlusion_state_map.keys())
        return valid_img_stroke_ids, valid_img_stroke_occlusion_state_map

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

    def load_transform_local_parameter(self, transform_params_path):
        transform_params_data = {}
        with open(transform_params_path, "r+") as f:
            for item in jsonlines.Reader(f):
                transform_params_data['pred_translate'] = item['pred_translate']  # (2), [-1.0, 1.0], relative to target trans0 window
                transform_params_data['pred_scaling_times'] = item['pred_scaling_times']  # (2), [0.2, 2.0], relative to target trans0 window
                transform_params_data['pred_rotate_angle'] = item['pred_rotate_angle']  # (), [-180.0, 180.0]
                transform_params_data['pred_shear_x_angle'] = item['pred_shear_x_angle']  # (), [-90.0, 90.0]
                transform_params_data['pred_shear_y_angle'] = item['pred_shear_y_angle']  # (), [-90.0, 90.0]
        return transform_params_data

    def process_stroke_parameter(self, parameters_ref, parameters_tar, comp_index, curve_index, stroke_index, image_size,
                                 stroke_endpoint_occlusion_state, occluded_mask):
        '''
        parameters_ref / parameters_tar: component list => curve list => stroke list (N', 4, 2)
        occluded_mask: (H, W), [0-occluded, 1-visible]
        '''
        curve_points_ref = parameters_ref[comp_index][curve_index]  # (N', 4, 2)
        curve_points_tar = parameters_tar[comp_index][curve_index]  # (N', 4, 2)

        if stroke_index == 0:
            p_prev = curve_points_ref[stroke_index][0]  # (2)
            p_curr = curve_points_ref[stroke_index][0]
            p_next = curve_points_ref[stroke_index][-1]
        else:
            p_prev = curve_points_ref[stroke_index - 1][0]  # (2)
            p_curr = curve_points_ref[stroke_index - 1][-1]
            p_next = curve_points_ref[stroke_index][-1]

        window_size_dist1 = np.abs(np.array(p_prev) - np.array(p_curr))  # (2), full size
        window_size_dist2 = np.abs(np.array(p_curr) - np.array(p_next))  # (2), full size
        window_size_dist = np.concatenate([window_size_dist1, window_size_dist2], axis=-1)  # (4), full size
        window_size = np.max(window_size_dist, axis=-1) * 2.0  # (), full size
        window_size_single = np.max(window_size_dist2, axis=-1) * 2.0  # (), full size
        window_size_norm = window_size / float(image_size)  # (), [0.0, 1.0]
        window_size_single_norm = window_size_single / float(image_size)  # (), [0.0, 1.0]

        window_size_scaled = window_size * self.window_size_scaling
        window_size_scaled = min(max(window_size_scaled, self.window_size_min), image_size * 1.5)

        centerpoint_ref = np.array(p_curr, dtype=np.float32)  # (2), full size
        end_ctrl_tar = np.array(curve_points_tar[stroke_index], dtype=np.float32)  # (4, 2), full size

        centerpoint_ref_norm = centerpoint_ref / float(image_size)  # (2), [0.0, 1.0]
        end_ctrl_tar_rel = (end_ctrl_tar - np.expand_dims(centerpoint_ref, axis=0)) / (window_size_scaled / 2.0)  # (4, 2), [-1.0, 1.0]

        if stroke_endpoint_occlusion_state in [0, 3]:
            end_ctrl_tar_rel_dist = np.copy(end_ctrl_tar_rel)
        else:
            # disturb target endpoint
            end_ctrl_tar_dist = disturb_endpoint(end_ctrl_tar, stroke_endpoint_occlusion_state == 1, occluded_mask,
                                                 image_size)  # (4, 2), in full size
            end_ctrl_tar_rel_dist = (end_ctrl_tar_dist - np.expand_dims(centerpoint_ref, axis=0)) / (window_size_scaled / 2.0)  # (4, 2), [-1.0, 1.0]

        end_ctrl_tar_rel_flatten = end_ctrl_tar_rel.flatten()
        end_ctrl_tar_rel_dist_flatten = end_ctrl_tar_rel_dist.flatten()
        return centerpoint_ref_norm, end_ctrl_tar_rel_dist_flatten, end_ctrl_tar_rel_flatten, window_size_norm, window_size_single_norm

    def get_batch(self, use_cuda, batch_idx=None, all_example=False, batch_idx_offset=0, occluded_only=False):
        reference_image_batch = []
        reference_stroke_batch = []
        reference_stroke_ctrl_batch = []
        target_image_batch = []
        reference_centerpoints_batch = []
        reference_centerpoints_offset_batch = []
        target_end_ctrl_offset_gt_batch = []
        target_end_ctrl_offset_gt_non_dist_batch = []
        target_occluded_mask_batch = []
        base_window_size_batch = []
        base_window_size_single_batch = []
        image_id_batch = []
        stroke_id_batch = []

        component_centerpoints_batch = []
        component_win_size_batch = []
        target_transform_cursor_batch = []
        target_transform_win_size_batch = []
        target_transform_angle_batch = []
        target_transform_shear_x_batch = []
        target_transform_shear_y_batch = []

        target_transform1_translate_batch = []
        target_transform1_scaling_batch = []
        target_transform1_angle_batch = []
        target_transform1_shear_x_batch = []
        target_transform1_shear_y_batch = []

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

            valid_stroke_ids, valid_stroke_occlusion_state_map = self.get_valid_img_ctrlpoints(
                selected_dataset_name, selected_index, reference_stroke_data, occluded_only=occluded_only)
            if not occluded_only:
                assert len(valid_stroke_ids) > 0
            else:
                if len(valid_stroke_ids) == 0:
                    return None

            transform_global_model_name_plus = self.transform_model_name
            if self.use_optical_flow:
                transform_global_model_name_plus += '-[optical]'
            transform_params_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split,
                                                 'component_transform_params', transform_global_model_name_plus, selected_index + '.jsonl')
            transform_params_data = self.load_transform_parameter(transform_params_path)

            if self.is_train:
                random.shuffle(valid_stroke_ids)
                random_stroke_ids = [valid_stroke_ids[0]]
            else:
                if not all_example:
                    if len(self.valid_stroke_index_buffer) <= batch_idx:
                        random.shuffle(valid_stroke_ids)
                        random_stroke_ids = [valid_stroke_ids[0]]
                        self.valid_stroke_index_buffer.append(random_stroke_ids[0])
                    else:
                        random_stroke_ids = [self.valid_stroke_index_buffer[batch_idx]]
                else:
                    random_stroke_ids = [item for item in valid_stroke_ids]

            for random_stroke_id in random_stroke_ids:
                comp_curve_point = random_stroke_id.split('_')
                c_i, curve_i, stroke_index = comp_curve_point
                stroke_id_batch.append(random_stroke_id)

                reference_stroke_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split, 'raster_black_endpoint_stroke',
                                                     str(selected_index), 'endpoint_' + random_stroke_id + '.png')
                reference_stroke_image = self.load_image(reference_stroke_path)  # (H, W), [0.0-strokes, 1.0-BG]
                reference_stroke_ctrl_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split, 'raster_black_ctrlpoint_stroke_ref',
                                                     str(selected_index), 'stroke_' + random_stroke_id + '.png')
                reference_stroke_ctrl_image = self.load_image(reference_stroke_ctrl_path)  # (H, W), [0.0-strokes, 1.0-BG]

                occluded_mask_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split,
                                                  'occluded_mask', str(selected_index), 'component_%s-tar.png' % c_i)
                if not os.path.exists(occluded_mask_path):
                    return None
                occluded_mask_image = self.load_image(occluded_mask_path)  # (H, W), [0-occluded, 1-visible]

                stroke_endpoint_occlusion_state = valid_stroke_occlusion_state_map[random_stroke_id]
                if not self.training_with_endpoint_disturb or not self.is_train:
                    stroke_endpoint_occlusion_state = 0

                centerpoint, end_ctrl_offset_gt, end_ctrl_offset_gt_non_dist, window_size, window_size_single = self.process_stroke_parameter(
                    reference_stroke_data, target_stroke_data, int(c_i), int(curve_i), int(stroke_index), image_size,
                    stroke_endpoint_occlusion_state, occluded_mask_image)
                # centerpoints: (2), [0.0, 1.0]
                # end_ctrl_offset_gt / end_ctrl_offset_gt_non_dist: (8), [-1.0, 1.0]
                # window_sizes / window_size_single: (), [0.0, 1.0]

                centerpoint_offset = np.maximum(np.minimum(centerpoint, 1.0), 0.0)

                transform_local_params_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split,
                                                           'component_local_transform_params')
                transform_models_name_plus = '[' + self.transform_model_name + ']-[' + self.transform_local_model_name + ']'
                if self.use_optical_flow:
                    transform_models_name_plus += '-[optical]'
                transform_local_params_path = os.path.join(transform_local_params_path, transform_models_name_plus,
                                                           str(selected_index), random_stroke_id + '.jsonl')
                transform_local_params_data = self.load_transform_local_parameter(transform_local_params_path)

                reference_image_batch.append(reference_image)
                reference_stroke_batch.append(reference_stroke_image)
                reference_stroke_ctrl_batch.append(reference_stroke_ctrl_image)
                target_image_batch.append(target_image)
                reference_centerpoints_batch.append(centerpoint)
                reference_centerpoints_offset_batch.append(centerpoint_offset)
                target_end_ctrl_offset_gt_batch.append(end_ctrl_offset_gt)
                target_end_ctrl_offset_gt_non_dist_batch.append(end_ctrl_offset_gt_non_dist)
                target_occluded_mask_batch.append(occluded_mask_image)
                base_window_size_batch.append(window_size)
                base_window_size_single_batch.append(window_size_single)
                component_centerpoints_batch.append(transform_params_data[int(c_i)]['component_center'])
                component_win_size_batch.append(transform_params_data[int(c_i)]['component_win_size'])
                target_transform_cursor_batch.append(transform_params_data[int(c_i)]['pred_cursor'])
                target_transform_win_size_batch.append(transform_params_data[int(c_i)]['pred_window_size'])
                target_transform_angle_batch.append(transform_params_data[int(c_i)]['pred_rotate_angle'])
                target_transform_shear_x_batch.append(transform_params_data[int(c_i)]['pred_shear_x_angle'])
                target_transform_shear_y_batch.append(transform_params_data[int(c_i)]['pred_shear_y_angle'])
                target_transform1_translate_batch.append(transform_local_params_data['pred_translate'])
                target_transform1_scaling_batch.append(transform_local_params_data['pred_scaling_times'])
                target_transform1_angle_batch.append(transform_local_params_data['pred_rotate_angle'])
                target_transform1_shear_x_batch.append(transform_local_params_data['pred_shear_x_angle'])
                target_transform1_shear_y_batch.append(transform_local_params_data['pred_shear_y_angle'])

        reference_image_batch = np.expand_dims(np.stack(reference_image_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        reference_stroke_batch = np.expand_dims(np.stack(reference_stroke_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        reference_stroke_ctrl_batch = np.expand_dims(np.stack(reference_stroke_ctrl_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        target_image_batch = np.expand_dims(np.stack(target_image_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        reference_centerpoints_batch = np.expand_dims(np.stack(reference_centerpoints_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0]
        reference_centerpoints_offset_batch = np.expand_dims(np.stack(reference_centerpoints_offset_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0]
        target_end_ctrl_offset_gt_batch = np.expand_dims(np.stack(target_end_ctrl_offset_gt_batch, axis=0), axis=1)  # (N, 1, 8), [-1.0, 1.0]
        target_end_ctrl_offset_gt_non_dist_batch = np.expand_dims(np.stack(target_end_ctrl_offset_gt_non_dist_batch, axis=0), axis=1)  # (N, 1, 8), [-1.0, 1.0]
        target_occluded_mask_batch = np.expand_dims(np.stack(target_occluded_mask_batch, axis=0), axis=-1)  # (N, H, W, 1), [0-occluded, 1-visible]
        base_window_size_batch = np.expand_dims(np.stack(base_window_size_batch, axis=0), axis=-1)  # (N, 1), [0.0, 1.0]
        base_window_size_single_batch = np.expand_dims(np.stack(base_window_size_single_batch, axis=0), axis=-1)  # (N, 1), [0.0, 1.0]
        component_centerpoints_batch = np.expand_dims(np.stack(component_centerpoints_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0], relative to image size
        component_win_size_batch = np.expand_dims(np.stack(component_win_size_batch, axis=0), axis=1)  # (N, 1, 2), in image size
        target_transform_cursor_batch = np.expand_dims(np.stack(target_transform_cursor_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0], relative to image size
        target_transform_win_size_batch = np.expand_dims(np.stack(target_transform_win_size_batch, axis=0), axis=1)  # (N, 1, 2), in image size
        target_transform_angle_batch = np.expand_dims(np.stack(target_transform_angle_batch, axis=0), axis=1)  # (N, 1), [-180.0, 180.0]
        target_transform_shear_x_batch = np.expand_dims(np.stack(target_transform_shear_x_batch, axis=0), axis=1)  # (N, 1), [-90.0, 90.0]
        target_transform_shear_y_batch = np.expand_dims(np.stack(target_transform_shear_y_batch, axis=0), axis=1)  # (N, 1), [-90.0, 90.0]
        target_transform1_translate_batch = np.expand_dims(np.stack(target_transform1_translate_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0], relative to image size
        target_transform1_scaling_batch = np.expand_dims(np.stack(target_transform1_scaling_batch, axis=0), axis=1)  # (N, 1, 2), in image size
        target_transform1_angle_batch = np.expand_dims(np.stack(target_transform1_angle_batch, axis=0), axis=1)  # (N, 1), [-180.0, 180.0]
        target_transform1_shear_x_batch = np.expand_dims(np.stack(target_transform1_shear_x_batch, axis=0), axis=1)  # (N, 1), [-90.0, 90.0]
        target_transform1_shear_y_batch = np.expand_dims(np.stack(target_transform1_shear_y_batch, axis=0), axis=1)  # (N, 1), [-90.0, 90.0]

        ## convert to tensor
        reference_image_batch = torch.tensor(reference_image_batch).float()
        reference_stroke_batch = torch.tensor(reference_stroke_batch).float()
        reference_stroke_ctrl_batch = torch.tensor(reference_stroke_ctrl_batch).float()
        target_image_batch = torch.tensor(target_image_batch).float()
        reference_centerpoints_batch = torch.tensor(reference_centerpoints_batch).float()
        reference_centerpoints_offset_batch = torch.tensor(reference_centerpoints_offset_batch).float()
        target_end_ctrl_offset_gt_batch = torch.tensor(target_end_ctrl_offset_gt_batch).float()
        target_end_ctrl_offset_gt_non_dist_batch = torch.tensor(target_end_ctrl_offset_gt_non_dist_batch).float()
        target_occluded_mask_batch = torch.tensor(target_occluded_mask_batch).float()
        base_window_size_batch = torch.tensor(base_window_size_batch).float()
        base_window_size_single_batch = torch.tensor(base_window_size_single_batch).float()
        component_centerpoints_batch = torch.tensor(component_centerpoints_batch).float()
        component_win_size_batch = torch.tensor(component_win_size_batch).float()
        target_transform_cursor_batch = torch.tensor(target_transform_cursor_batch).float()
        target_transform_win_size_batch = torch.tensor(target_transform_win_size_batch).float()
        target_transform_angle_batch = torch.tensor(target_transform_angle_batch).float()
        target_transform_shear_x_batch = torch.tensor(target_transform_shear_x_batch).float()
        target_transform_shear_y_batch = torch.tensor(target_transform_shear_y_batch).float()
        target_transform1_translate_batch = torch.tensor(target_transform1_translate_batch).float()
        target_transform1_scaling_batch = torch.tensor(target_transform1_scaling_batch).float()
        target_transform1_angle_batch = torch.tensor(target_transform1_angle_batch).float()
        target_transform1_shear_x_batch = torch.tensor(target_transform1_shear_x_batch).float()
        target_transform1_shear_y_batch = torch.tensor(target_transform1_shear_y_batch).float()

        if use_cuda:
            reference_image_batch = reference_image_batch.cuda()
            reference_stroke_batch = reference_stroke_batch.cuda()
            reference_stroke_ctrl_batch = reference_stroke_ctrl_batch.cuda()
            target_image_batch = target_image_batch.cuda()
            reference_centerpoints_batch = reference_centerpoints_batch.cuda()
            reference_centerpoints_offset_batch = reference_centerpoints_offset_batch.cuda()
            target_end_ctrl_offset_gt_batch = target_end_ctrl_offset_gt_batch.cuda()
            target_end_ctrl_offset_gt_non_dist_batch = target_end_ctrl_offset_gt_non_dist_batch.cuda()
            target_occluded_mask_batch = target_occluded_mask_batch.cuda()
            base_window_size_batch = base_window_size_batch.cuda()
            base_window_size_single_batch = base_window_size_single_batch.cuda()
            component_centerpoints_batch = component_centerpoints_batch.cuda()
            component_win_size_batch = component_win_size_batch.cuda()
            target_transform_cursor_batch = target_transform_cursor_batch.cuda()
            target_transform_win_size_batch = target_transform_win_size_batch.cuda()
            target_transform_angle_batch = target_transform_angle_batch.cuda()
            target_transform_shear_x_batch = target_transform_shear_x_batch.cuda()
            target_transform_shear_y_batch = target_transform_shear_y_batch.cuda()
            target_transform1_translate_batch = target_transform1_translate_batch.cuda()
            target_transform1_scaling_batch = target_transform1_scaling_batch.cuda()
            target_transform1_angle_batch = target_transform1_angle_batch.cuda()
            target_transform1_shear_x_batch = target_transform1_shear_x_batch.cuda()
            target_transform1_shear_y_batch = target_transform1_shear_y_batch.cuda()

        return reference_image_batch, reference_stroke_batch, reference_stroke_ctrl_batch, target_image_batch, \
               reference_centerpoints_batch, reference_centerpoints_offset_batch, \
               target_end_ctrl_offset_gt_batch, target_end_ctrl_offset_gt_non_dist_batch, \
               target_occluded_mask_batch, \
               base_window_size_batch, base_window_size_single_batch, image_id_batch, stroke_id_batch, \
               component_centerpoints_batch, component_win_size_batch, \
               target_transform_cursor_batch, target_transform_win_size_batch, target_transform_angle_batch, \
               target_transform_shear_x_batch, target_transform_shear_y_batch, \
               target_transform1_translate_batch, target_transform1_scaling_batch, target_transform1_angle_batch, \
               target_transform1_shear_x_batch, target_transform1_shear_y_batch


def load_dataset(model_params, test_only=False):
    data_base = model_params.dataset_base

    valid_model_params = copy_hparams(model_params)
    valid_model_params.batch_size = 1  # only sample one at a time

    if not test_only:
        train_set = LineDataLoader(dataset_base=data_base, batch_size=model_params.batch_size,
                                   window_size_scaling=model_params.window_size_scaling_ref,
                                   window_size_min=model_params.window_size_min,
                                   window_size_scaling_comp=model_params.window_size_scaling_ref_comp,
                                   window_size_min_comp=model_params.window_size_min_comp,
                                   transform_model_name=model_params.transform_model_name,
                                   transform_local_model_name=model_params.transform_local_model_name,
                                   use_optical_flow=model_params.use_optical_flow,
                                   training_with_endpoint_disturb=model_params.training_with_endpoint_disturb,
                                   do_dataset_filtering=model_params.do_dataset_filtering,
                                   is_train=True)
    else:
        train_set = None
    val_set = LineDataLoader(dataset_base=data_base, batch_size=valid_model_params.batch_size,
                             window_size_scaling=valid_model_params.window_size_scaling_ref,
                             window_size_min=valid_model_params.window_size_min,
                             window_size_scaling_comp=valid_model_params.window_size_scaling_ref_comp,
                             window_size_min_comp=valid_model_params.window_size_min_comp,
                             transform_model_name=valid_model_params.transform_model_name,
                             transform_local_model_name=valid_model_params.transform_local_model_name,
                             use_optical_flow=valid_model_params.use_optical_flow,
                             training_with_endpoint_disturb=valid_model_params.training_with_endpoint_disturb,
                             do_dataset_filtering=valid_model_params.do_dataset_filtering,
                             is_train=False)

    result = [train_set, val_set, model_params, valid_model_params]
    return result
