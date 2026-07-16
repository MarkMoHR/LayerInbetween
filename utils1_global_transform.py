import os
import random

import numpy as np
from PIL import Image
import jsonlines
import copy
from glob import glob

import torch

# import pydiffvg

from hparam import HParams
from dataset_utils.common import load_txt_ids, OPTICAL_FLOW_DIR_MAP


def copy_hparams(hparams):
    """Return a copy of an HParams instance."""
    return HParams(**hparams.values())


class LineDataLoader(object):
    def __init__(self,
                 dataset_base,
                 batch_size,
                 window_size_scaling,
                 window_size_min,
                 stroke_thickness,
                 use_optical_flow,
                 optical_flow_method,
                 use_distance_transform,
                 do_dataset_filtering,
                 is_train):
        self.dataset_base = dataset_base
        self.batch_size = batch_size
        self.window_size_scaling = window_size_scaling
        self.window_size_min = window_size_min
        self.stroke_thickness = stroke_thickness
        self.use_optical_flow = use_optical_flow
        self.optical_flow_method = optical_flow_method
        self.use_distance_transform = use_distance_transform
        self.do_dataset_filtering = do_dataset_filtering
        self.is_train = is_train
        self.dataset_names = ['creature', 'bird']
        self.ref_tar_split_names = ['ref', 'tar']
        self.dataset_split = 'train' if is_train else 'val'

        self.img_ids = self.get_img_ids()
        example_num = len(self.img_ids)
        print('Loaded', self.dataset_split, ':', example_num)

        self.example_num = example_num
        self.num_batches = example_num // self.batch_size
        print('batch_size', batch_size, ', num_batches', self.num_batches)

        self.valid_component_index_buffer = []
        self.invalid_img_comp_map, self.valid_occlusion_img_comp_map = self.get_img_comp_map()

    def get_img_comp_map(self):
        invalid_img_comp_map = {}
        valid_occlusion_img_comp_map = {}

        if self.do_dataset_filtering:
            outsider_img_comp_ids_list_path = os.path.join(self.dataset_base, 'transform_invalid_comp_ids', 'out-of-bound',
                                                          self.dataset_split + '-win=' + str(self.window_size_scaling) + '-min=' + str(self.window_size_min) + '.txt')
            outsider_img_comp_ids_list = load_txt_ids(outsider_img_comp_ids_list_path)

            invalid_occ_img_comp_ids_list_path = os.path.join(self.dataset_base, 'transform_invalid_comp_ids', 'occlusion',
                                                              self.dataset_split + '_invalid.txt')
            invalid_occ_img_comp_ids_list = load_txt_ids(invalid_occ_img_comp_ids_list_path)
            single_stroke_comp_ids_list_path = os.path.join(self.dataset_base, 'transform_invalid_comp_ids', 'single-stroke-component',
                                                            self.dataset_split + '_invalid.txt')
            single_stroke_comp_ids_list = load_txt_ids(single_stroke_comp_ids_list_path)
            invalid_img_comp_ids_list = outsider_img_comp_ids_list + invalid_occ_img_comp_ids_list + single_stroke_comp_ids_list
            invalid_img_comp_ids_list = list(set(invalid_img_comp_ids_list))
            print('invalid_img_comp_ids_list:', len(invalid_img_comp_ids_list))
        else:
            invalid_img_comp_ids_list = []

        valid_occ_img_comp_ids_list_path = os.path.join(self.dataset_base, 'transform_invalid_comp_ids', 'occlusion',
                                                        self.dataset_split + '_valid.txt')
        valid_occ_img_comp_ids_list = load_txt_ids(valid_occ_img_comp_ids_list_path)

        for item in invalid_img_comp_ids_list:
            img_id = item[:item.rfind('-')]
            comp_id = item[item.rfind('-')+1:]
            if img_id not in invalid_img_comp_map.keys():
                invalid_img_comp_map[img_id] = [int(comp_id)]
            else:
                invalid_img_comp_map[img_id].append(int(comp_id))

        for item in valid_occ_img_comp_ids_list:
            if item in invalid_img_comp_ids_list:
                continue
            img_id = item[:item.rfind('-')]
            comp_id = item[item.rfind('-') + 1:]
            if img_id not in valid_occlusion_img_comp_map.keys():
                valid_occlusion_img_comp_map[img_id] = [int(comp_id)]
            else:
                valid_occlusion_img_comp_map[img_id].append(int(comp_id))

        return invalid_img_comp_map, valid_occlusion_img_comp_map

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

    def get_valid_component_ids(self, image_index, reference_stroke_data, occluded_only=False):
        component_num = len(reference_stroke_data)
        if not occluded_only:
            valid_component_ids = [ii for ii in range(component_num)]
        else:
            if image_index in self.valid_occlusion_img_comp_map.keys():
                valid_occlusion_component_ids = self.valid_occlusion_img_comp_map[image_index]
                valid_component_ids = [item for item in valid_occlusion_component_ids]
            else:
                valid_component_ids = []

        if image_index in self.invalid_img_comp_map.keys():
            invalid_component_ids = self.invalid_img_comp_map[image_index]
            assert len(invalid_component_ids) < component_num
            for item in invalid_component_ids:
                # assert item in valid_component_ids
                if item in valid_component_ids:
                    valid_component_ids.remove(item)
        return valid_component_ids

    def load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.float32)  # (H, W, 3), [0.0-strokes, 255.0-BG]
        image = image[:, :, 0] / 255.0  # (H, W), [0.0-strokes, 1.0-BG]
        return image

    def load_stroke_parameter(self, vector_data_path):
        stroke_data_b_list = []
        # parts_data_list = []
        with open(vector_data_path, "r+") as f:
            for item in jsonlines.Reader(f):
                stroke_data_b = item['stroke_params']
                # parts_data = item['component_part']
                stroke_data_b_list.append(stroke_data_b)
                # parts_data_list.append(parts_data)
        assert len(stroke_data_b_list) == 1
        # assert len(parts_data_list) == 1
        return stroke_data_b_list[0]

    def load_translate_parameter(self, translate_params_path, component_index):
        with open(translate_params_path, "r+") as f:
            for item in jsonlines.Reader(f):
                c_idx = item['component_index']
                component_offset = item['component_offset']  # (2), [dx, dy], in image size
                if component_index == c_idx:
                    return component_offset
        raise Exception('No such component_index?')

    def process_stroke_parameter_ref(self, parameters, selected_component_index, image_size):
        '''
        parameters: component list => curve list => stroke list (N', 4, 2)
        '''
        curve_list = parameters[selected_component_index]  # list of (N', 4, 2)
        stroke_list = np.concatenate(curve_list, axis=0)  # (N^, 4, 2)
        endpoints = np.concatenate([stroke_list[:, 0, :], stroke_list[:, -1, :]], axis=0)  # (N^ * 2, 2)
        x_min, x_max = np.min(endpoints[:, 0]), np.max(endpoints[:, 0])
        y_min, y_max = np.min(endpoints[:, 1]), np.max(endpoints[:, 1])
        centerpoint = np.array([(x_min + x_max) / 2.0,
                                (y_min + y_max) / 2.0], dtype=np.float32) / float(image_size)  # (2), [0.0, 1.0]

        window_size = max(x_max - x_min, y_max - y_min) / float(image_size)  # (), [0.0, 1.0]
        return centerpoint, window_size

    def get_batch(self, use_cuda, batch_idx=None, all_example=False, batch_idx_offset=0, occluded_only=False):
        reference_image_batch = []
        reference_component_batch = []
        target_image_batch = []
        target_component_batch = []
        reference_centerpoints_batch = []
        reference_centerpoints_offset_batch = []
        base_window_size_batch = []
        image_id_batch = []
        component_id_batch = []

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

            reference_image = self.load_image(reference_image_path)  # (H, W), [0.0-strokes, 1.0-BG]
            target_image = self.load_image(target_image_path)  # (H, W), [0.0-strokes, 1.0-BG]
            reference_stroke_data = self.load_stroke_parameter(reference_stroke_path)
            # reference_stroke_data: component list => curve list => stroke list (N', 4, 2)

            image_size = reference_image.shape[0]

            valid_component_ids = self.get_valid_component_ids(selected_id, reference_stroke_data, occluded_only=occluded_only)
            if not occluded_only:
                assert len(valid_component_ids) > 0
            else:
                if len(valid_component_ids) == 0:
                    return None

            if self.is_train:
                random.shuffle(valid_component_ids)
                random_component_indices = [valid_component_ids[0]]
            else:
                if not all_example:
                    if len(self.valid_component_index_buffer) <= batch_idx:
                        random.shuffle(valid_component_ids)
                        random_component_indices = [valid_component_ids[0]]
                        self.valid_component_index_buffer.append(random_component_indices[0])
                    else:
                        random_component_indices = [self.valid_component_index_buffer[batch_idx]]
                else:
                    random_component_indices = [item for item in valid_component_ids]
            component_id_batch += random_component_indices

            for random_component_index in random_component_indices:
                reference_component_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split, 'raster_black_component', str(selected_index),
                                                        'component_' + str(random_component_index) + '-' + self.ref_tar_split_names[0] + '.png')
                target_component_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split, 'raster_black_component', str(selected_index),
                                                     'component_' + str(random_component_index) + '-' + self.ref_tar_split_names[1] + '.png')

                reference_component_image = self.load_image(reference_component_path)  # (H, W), [0.0-strokes, 1.0-BG]
                target_component_image = self.load_image(target_component_path)  # (H, W), [0.0-strokes, 1.0-BG]

                centerpoint, window_size = self.process_stroke_parameter_ref(reference_stroke_data, random_component_index, image_size)
                # centerpoints: (2), [0.0, 1.0]
                # window_sizes: (), [0.0, 1.0]

                # add offset when using optical flow
                if self.use_optical_flow:
                    optical_flow_dir = OPTICAL_FLOW_DIR_MAP[self.optical_flow_method]
                    if self.use_distance_transform:
                        optical_flow_dir += '-[DT-10]'
                    translate_params_path = os.path.join(self.dataset_base, selected_dataset_name + '_512', self.dataset_split,
                                                         'optical_flow', optical_flow_dir, 'component_offset', str(selected_index) + '.jsonl')
                    component_offset = self.load_translate_parameter(translate_params_path, random_component_index)
                    # (2), [dx, dy], in image size
                    centerpoint_offset = (centerpoint * float(image_size) + np.array(component_offset, dtype=np.float32)) / float(image_size)  # (2), [0.0, 1.0]
                else:
                    centerpoint_offset = np.maximum(np.minimum(centerpoint, 1.0), 0.0)

                reference_image_batch.append(reference_image)
                reference_component_batch.append(reference_component_image)
                target_image_batch.append(target_image)
                target_component_batch.append(target_component_image)
                reference_centerpoints_batch.append(centerpoint)
                reference_centerpoints_offset_batch.append(centerpoint_offset)
                base_window_size_batch.append(window_size)

        reference_image_batch = np.expand_dims(np.stack(reference_image_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        reference_component_batch = np.expand_dims(np.stack(reference_component_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        target_image_batch = np.expand_dims(np.stack(target_image_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        target_component_batch = np.expand_dims(np.stack(target_component_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        reference_centerpoints_batch = np.expand_dims(np.stack(reference_centerpoints_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0]
        reference_centerpoints_offset_batch = np.expand_dims(np.stack(reference_centerpoints_offset_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0]
        base_window_size_batch = np.expand_dims(np.stack(base_window_size_batch, axis=0), axis=-1)  # (N, 1), [0.0, 1.0]

        ## convert to tensor
        reference_image_batch = torch.tensor(reference_image_batch).float()
        reference_component_batch = torch.tensor(reference_component_batch).float()
        target_image_batch = torch.tensor(target_image_batch).float()
        target_component_batch = torch.tensor(target_component_batch).float()
        reference_centerpoints_batch = torch.tensor(reference_centerpoints_batch).float()
        reference_centerpoints_offset_batch = torch.tensor(reference_centerpoints_offset_batch).float()
        base_window_size_batch = torch.tensor(base_window_size_batch).float()

        if use_cuda:
            reference_image_batch = reference_image_batch.cuda()
            reference_component_batch = reference_component_batch.cuda()
            target_image_batch = target_image_batch.cuda()
            target_component_batch = target_component_batch.cuda()
            reference_centerpoints_batch = reference_centerpoints_batch.cuda()
            reference_centerpoints_offset_batch = reference_centerpoints_offset_batch.cuda()
            base_window_size_batch = base_window_size_batch.cuda()

        return reference_image_batch, reference_component_batch, target_image_batch, target_component_batch, \
               reference_centerpoints_batch, reference_centerpoints_offset_batch, \
               base_window_size_batch, image_id_batch, component_id_batch


class RealLineDataLoader(object):
    def __init__(self,
                 dataset_base,
                 dataset_base_extra,
                 batch_size,
                 window_size_scaling,
                 window_size_min,
                 stroke_thickness,
                 use_optical_flow,
                 optical_flow_method,
                 use_distance_transform,
                 use_target_layer,
                 target_layer_method):
        self.dataset_base = dataset_base
        self.dataset_base_extra = dataset_base_extra
        self.batch_size = batch_size
        self.window_size_scaling = window_size_scaling
        self.window_size_min = window_size_min
        self.stroke_thickness = stroke_thickness
        self.use_optical_flow = use_optical_flow
        self.optical_flow_method = optical_flow_method
        self.use_distance_transform = use_distance_transform
        self.use_target_layer = use_target_layer
        self.target_layer_method = target_layer_method
        self.ref_tar_split_names = ['ref', 'tar']

    def get_valid_component_ids(self, reference_stroke_data):
        component_num = len(reference_stroke_data)
        valid_component_ids = [ii for ii in range(component_num)]
        return valid_component_ids

    def load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.float32)  # (H, W, 3), [0.0-strokes, 255.0-BG]
        image = image[:, :, 0] / 255.0  # (H, W), [0.0-strokes, 1.0-BG]
        return image

    def load_stroke_parameter(self, vector_data_path, vector_data_path_extra=None):
        with open(vector_data_path, "r+") as f:
            for item in jsonlines.Reader(f):
                stroke_data_b = item['stroke_params']

        if vector_data_path_extra is None:
            return stroke_data_b
        else:
            with open(vector_data_path_extra, "r+") as f:
                for item in jsonlines.Reader(f):
                    stroke_data_b_extra = item['stroke_params']
            assert len(stroke_data_b_extra) == len(stroke_data_b)

            stroke_data_b_comb = []  # component list => curve list => stroke list (N', 4, 2)
            for c_i in range(len(stroke_data_b)):
                curve_list = copy.deepcopy(stroke_data_b[c_i])  # list of (N', 4, 2)
                curve_list += copy.deepcopy(stroke_data_b_extra[c_i])
                stroke_data_b_comb.append(curve_list)
            return stroke_data_b_comb

    def load_translate_parameter(self, translate_params_path, component_index):
        with open(translate_params_path, "r+") as f:
            for item in jsonlines.Reader(f):
                c_idx = item['component_index']
                component_offset = item['component_offset']  # (2), [dx, dy], in image size
                if component_index == c_idx:
                    return component_offset
        raise Exception('No such component_index?')

    def process_stroke_parameter_ref(self, parameters, selected_component_index, image_size):
        '''
        parameters: component list => curve list => stroke list (N', 4, 2)
        '''
        curve_list = parameters[selected_component_index]  # list of (N', 4, 2)
        stroke_list = np.concatenate(curve_list, axis=0)  # (N^, 4, 2)
        endpoints = np.concatenate([stroke_list[:, 0, :], stroke_list[:, -1, :]], axis=0)  # (N^ * 2, 2)
        x_min, x_max = np.min(endpoints[:, 0]), np.max(endpoints[:, 0])
        y_min, y_max = np.min(endpoints[:, 1]), np.max(endpoints[:, 1])
        centerpoint = np.array([(x_min + x_max) / 2.0,
                                (y_min + y_max) / 2.0], dtype=np.float32) / float(image_size)  # (2), [0.0, 1.0]

        window_size = max(x_max - x_min, y_max - y_min) / float(image_size)  # (), [0.0, 1.0]
        return centerpoint, window_size

    def get_batch(self, use_cuda, test_img_id):
        reference_image_batch = []
        reference_component_batch = []
        target_image_batch = []
        reference_centerpoints_batch = []
        reference_centerpoints_offset_batch = []
        base_window_size_batch = []
        image_id_batch = []
        component_id_batch = []

        selected_indices = [test_img_id]

        for batch_i in range(len(selected_indices)):
            selected_index = str(selected_indices[batch_i])
            image_id_batch.append(selected_index)

            reference_image_path = os.path.join(self.dataset_base, 'raster_black', str(selected_index) + '_' + self.ref_tar_split_names[0] + '.png')
            reference_stroke_path = os.path.join(self.dataset_base, 'vector-params', str(selected_index) + '_' + self.ref_tar_split_names[0] + '.jsonl')
            reference_stroke_path_extra = os.path.join(self.dataset_base_extra, 'params', 'tar_pred-' + str(selected_index) + '.jsonl') if self.dataset_base_extra is not None else None
            target_image_path = os.path.join(self.dataset_base, 'raster_black', str(selected_index) + '_' + self.ref_tar_split_names[1] + '.png')

            reference_image = self.load_image(reference_image_path)  # (H, W), [0.0-strokes, 1.0-BG]
            target_image = self.load_image(target_image_path)  # (H, W), [0.0-strokes, 1.0-BG]
            reference_stroke_data = self.load_stroke_parameter(reference_stroke_path, reference_stroke_path_extra)
            # reference_stroke_data: component list => curve list => stroke list (N', 4, 2)

            image_size = reference_image.shape[0]

            valid_component_ids = self.get_valid_component_ids(reference_stroke_data)
            assert len(valid_component_ids) > 0

            random_component_indices = [item for item in valid_component_ids]
            component_id_batch += random_component_indices

            for random_component_index in random_component_indices:
                reference_component_path = os.path.join(self.dataset_base, 'raster_black_component', str(selected_index),
                                                        'component_' + str(random_component_index) + '-' + self.ref_tar_split_names[0] + '.png')
                reference_component_image = self.load_image(reference_component_path)  # (H, W), [0.0-strokes, 1.0-BG]

                centerpoint, window_size = self.process_stroke_parameter_ref(reference_stroke_data, random_component_index, image_size)
                # centerpoints: (2), [0.0, 1.0]
                # window_sizes: (), [0.0, 1.0]

                # add offset when using optical flow
                if self.use_optical_flow:
                    optical_flow_dir = OPTICAL_FLOW_DIR_MAP[self.optical_flow_method]
                    if self.use_distance_transform:
                        optical_flow_dir += '-[DT-10]'
                    translate_params_path = os.path.join(self.dataset_base,
                                                         'optical_flow', optical_flow_dir, 'component_offset', str(selected_index) + '.jsonl')
                    component_offset = self.load_translate_parameter(translate_params_path, random_component_index)
                    # (2), [dx, dy], in image size
                    centerpoint_offset = (centerpoint * float(image_size) + np.array(component_offset, dtype=np.float32)) / float(image_size)  # (2), [0.0, 1.0]
                else:
                    centerpoint_offset = np.maximum(np.minimum(centerpoint, 1.0), 0.0)

                if self.use_target_layer:
                    if self.target_layer_method == 'box_depth_ol':
                        target_layer_dir = '[box]-[depth_overlap]'
                    elif self.target_layer_method == 'box_depth':
                        target_layer_dir = '[box]-[depth]'
                    elif self.target_layer_method == 'mask_line':
                        target_layer_dir = '[mask]-[linearts]'
                    elif self.target_layer_method == 'box_depth+mask_line' or self.target_layer_method == 'box_depth_ol+mask_line':
                        target_layer_dir = '[both]'
                    else:
                        raise Exception('Unknown target_layer_method:', self.target_layer_method)

                    reference_image_batch.append(reference_image)

                    target_layer_image_path = os.path.join(self.dataset_base, 'layers', target_layer_dir, 'image',
                                                           str(selected_index), str(random_component_index) + '_tar.png')
                    target_layer_image = self.load_image(target_layer_image_path)  # (H, W), [0.0-strokes, 1.0-BG]
                    target_image_batch.append(target_layer_image)
                else:
                    reference_image_batch.append(reference_image)
                    target_image_batch.append(target_image)

                reference_component_batch.append(reference_component_image)
                reference_centerpoints_batch.append(centerpoint)
                reference_centerpoints_offset_batch.append(centerpoint_offset)
                base_window_size_batch.append(window_size)

        reference_image_batch = np.expand_dims(np.stack(reference_image_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        reference_component_batch = np.expand_dims(np.stack(reference_component_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        target_image_batch = np.expand_dims(np.stack(target_image_batch, axis=0), axis=-1)  # (N, H, W, 1), [0.0-strokes, 1.0-BG]
        reference_centerpoints_batch = np.expand_dims(np.stack(reference_centerpoints_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0]
        reference_centerpoints_offset_batch = np.expand_dims(np.stack(reference_centerpoints_offset_batch, axis=0), axis=1)  # (N, 1, 2), [0.0, 1.0]
        base_window_size_batch = np.expand_dims(np.stack(base_window_size_batch, axis=0), axis=-1)  # (N, 1), [0.0, 1.0]

        ## convert to tensor
        reference_image_batch = torch.tensor(reference_image_batch).float()
        reference_component_batch = torch.tensor(reference_component_batch).float()
        target_image_batch = torch.tensor(target_image_batch).float()
        reference_centerpoints_batch = torch.tensor(reference_centerpoints_batch).float()
        reference_centerpoints_offset_batch = torch.tensor(reference_centerpoints_offset_batch).float()
        base_window_size_batch = torch.tensor(base_window_size_batch).float()

        if use_cuda:
            reference_image_batch = reference_image_batch.cuda()
            reference_component_batch = reference_component_batch.cuda()
            target_image_batch = target_image_batch.cuda()
            reference_centerpoints_batch = reference_centerpoints_batch.cuda()
            reference_centerpoints_offset_batch = reference_centerpoints_offset_batch.cuda()
            base_window_size_batch = base_window_size_batch.cuda()

        return reference_image_batch, reference_component_batch, target_image_batch, \
               reference_centerpoints_batch, reference_centerpoints_offset_batch, \
               base_window_size_batch, image_id_batch, component_id_batch


def load_dataset(model_params, test_only=False):
    data_base = model_params.dataset_base

    valid_model_params = copy_hparams(model_params)
    valid_model_params.batch_size = 1  # only sample one at a time

    if not test_only:
        train_set = LineDataLoader(dataset_base=data_base, batch_size=model_params.batch_size,
                                   stroke_thickness=model_params.stroke_thickness,
                                   window_size_scaling=model_params.window_size_scaling_ref,
                                   window_size_min=model_params.window_size_min,
                                   use_optical_flow=model_params.use_optical_flow,
                                   optical_flow_method=model_params.optical_flow_method,
                                   use_distance_transform=model_params.use_distance_transform,
                                   do_dataset_filtering=model_params.do_dataset_filtering,
                                   is_train=True)
    else:
        train_set = None
    val_set = LineDataLoader(dataset_base=data_base, batch_size=valid_model_params.batch_size,
                             stroke_thickness=valid_model_params.stroke_thickness,
                             window_size_scaling=valid_model_params.window_size_scaling_ref,
                             window_size_min=valid_model_params.window_size_min,
                             use_optical_flow=valid_model_params.use_optical_flow,
                             optical_flow_method=valid_model_params.optical_flow_method,
                             use_distance_transform=valid_model_params.use_distance_transform,
                             do_dataset_filtering=valid_model_params.do_dataset_filtering,
                             is_train=False)

    result = [train_set, val_set, model_params, valid_model_params]
    return result


def load_real_dataset(model_params, data_base, data_base_extra=None):
    valid_model_params = copy_hparams(model_params)
    valid_model_params.batch_size = 1  # only sample one at a time

    val_set = RealLineDataLoader(dataset_base=data_base,
                                 dataset_base_extra=data_base_extra,
                                 batch_size=valid_model_params.batch_size,
                                 stroke_thickness=valid_model_params.stroke_thickness,
                                 window_size_scaling=valid_model_params.window_size_scaling_ref,
                                 window_size_min=valid_model_params.window_size_min,
                                 use_optical_flow=valid_model_params.use_optical_flow,
                                 use_distance_transform=valid_model_params.use_distance_transform,
                                 optical_flow_method=valid_model_params.optical_flow_method,
                                 use_target_layer=valid_model_params.use_target_layer,
                                 target_layer_method=valid_model_params.target_layer_method)

    result = [val_set, model_params, valid_model_params]
    return result
