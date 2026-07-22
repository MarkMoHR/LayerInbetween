import argparse
import os

from configs.example_configs import gpu_id, test_data_base, test_img_id, example_info_map, gen_time
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from PIL import Image
import cv2
import jsonlines
import shutil

from build_sam import build_sam2_video_predictor
from sam2_common import cal_mask_bbox, generate_colors2, mask_dilate


def png_to_jpg(data_dir):
    save_dir = os.path.join(data_dir, 'video_frames')
    os.makedirs(save_dir, exist_ok=True)

    all_files = os.listdir(data_dir)
    all_files = [item for item in all_files if '.png' in item]
    file_num = len(all_files)

    for file_i in range(file_num):
        img_path = os.path.join(data_dir, 'frame' + str(file_i) + '.png')
        img_path_c = os.path.join(save_dir, str(file_i) + ".jpg")
        img = Image.open(img_path).convert('RGB')
        img.save(img_path_c)

    return save_dir


def remove_inclusion_and_obtain_layers(out_frame_idx, sorted_layer_mask_list, lineart_path, save_base_layer,
                                       verbose=True, recolor_mask=False):
    save_base_layer_raw_mask = os.path.join(save_base_layer, 'mask_raw')
    save_base_layer_proc_mask = os.path.join(save_base_layer, 'mask_proc')
    os.makedirs(save_base_layer_raw_mask, exist_ok=True)
    os.makedirs(save_base_layer_proc_mask, exist_ok=True)
    if verbose:
        save_base_layer_image = os.path.join(save_base_layer, 'image')
        save_base_layer_image_vis = os.path.join(save_base_layer, 'image_vis')
        os.makedirs(save_base_layer_image, exist_ok=True)
        os.makedirs(save_base_layer_image_vis, exist_ok=True)
    if recolor_mask:
        save_base_layer_proc_mask_color = os.path.join(save_base_layer_proc_mask, 'recolor')
        os.makedirs(save_base_layer_proc_mask_color, exist_ok=True)
        recolors = [  # in front is in front
            # [255, 185, 0],  # orange
            # [0, 255, 92],  # green
            # # [255, 0, 232],  # pink
            # [0, 0, 230],  # dark blue

            [16, 190, 255],  # light blue
            [255, 137, 0],  # orange
            [2, 42, 255],  # dark blue
            [0, 255, 35],  # green
            [231, 0, 255],  # pink

        ]

    ## Step-5: remove inclusion
    num_mask = len(sorted_layer_mask_list)

    if verbose:
        lineart = Image.open(lineart_path).convert("L")
        lineart = np.array(lineart, dtype=np.uint8)

        layer_vis = np.ones_like(lineart, dtype=np.float32) * 255.0
        layer_vis = np.tile(np.expand_dims(layer_vis, axis=-1), (1, 1, 3))
        colors = generate_colors2(num_mask)  # list of (3), in [0., 1.]

    for m_i in range(num_mask):
        obj_id_i = sorted_layer_mask_list[m_i]['obj_id']
        mask_i = sorted_layer_mask_list[m_i]['mask']
        bbox_i = sorted_layer_mask_list[m_i]['bbox']
        mask_i_proc = np.ones_like(mask_i)
        if recolor_mask:
            proc_mask_recolor = np.ones(shape=(mask_i_proc.shape[0], mask_i_proc.shape[1], 3), dtype=np.float32) * 255

        for m_j in range(m_i-1, -1, -1):  # iterate all smaller masks
            mask_j = sorted_layer_mask_list[m_j]['mask']
            bbox_j = sorted_layer_mask_list[m_j]['bbox']
            area_j = sorted_layer_mask_list[m_j]['area']
            if bbox_j is None:
                continue

            mask_i_proc = np.logical_and(mask_i_proc, np.logical_not(mask_j))
            if recolor_mask:
                proc_mask_recolor[mask_j] = recolors[m_j]

        ## Step-6: filter lineart image to obtain layers
        if verbose:
            layer = np.copy(lineart)
            layer[np.logical_not(mask_i_proc)] = 255

            layer_rbg = 255.0 - np.tile(np.expand_dims(layer, axis=-1), (1, 1, 3)).astype(np.float32)
            layer_rbg *= 1.0 - np.expand_dims(np.expand_dims(np.array(colors[obj_id_i])[::-1], axis=0), axis=0)
            layer_rbg = 255.0 - layer_rbg

            layer_vis[mask_i_proc] = layer_rbg[mask_i_proc]

        save_path_raw_mask = os.path.join(save_base_layer_raw_mask, str(obj_id_i) + "_fra=" + str(out_frame_idx) + ".png")
        save_path_proc_mask = os.path.join(save_base_layer_proc_mask, str(obj_id_i) + "_fra=" + str(out_frame_idx) + ".png")

        raw_mask = np.zeros_like(mask_i, dtype=np.uint8)
        raw_mask[mask_i] = 255
        raw_mask = Image.fromarray(raw_mask, 'L')
        raw_mask.save(save_path_raw_mask, 'PNG')

        proc_mask = np.zeros_like(mask_i_proc, dtype=np.uint8)
        proc_mask[mask_i_proc] = 255
        proc_mask = Image.fromarray(proc_mask, 'L')
        proc_mask.save(save_path_proc_mask, 'PNG')

        if recolor_mask:
            save_path_proc_mask = os.path.join(save_base_layer_proc_mask_color, str(obj_id_i) + "_fra=" + str(out_frame_idx) + ".png")
            proc_mask_recolor = proc_mask_recolor.astype(np.uint8)
            proc_mask_recolor = Image.fromarray(proc_mask_recolor, 'RGB')
            proc_mask_recolor.save(save_path_proc_mask, 'PNG')

        if verbose:
            save_path_layer_img = os.path.join(save_base_layer_image, str(obj_id_i) + "_fra=" + str(out_frame_idx) + ".png")
            layer = Image.fromarray(layer, 'L')
            layer.save(save_path_layer_img, 'PNG')

    if verbose:
        save_path_layer_vis = os.path.join(save_base_layer_image_vis, "fra=" + str(out_frame_idx) + ".png")
        layer_vis = Image.fromarray(layer_vis.astype(np.uint8), 'RGB')
        layer_vis.save(save_path_layer_vis, 'PNG')


def single_prompt_inference(raw_data_base, correspondence_result_base, inbetween_result_base, test_id, image_type, prompt_type, example_info_map,
                            verbose=False):
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    model_config_map = {'sam2.1_hiera_large.pt': 'sam2.1_hiera_l.yaml',
                        'sam2.1_hiera_base_plus.pt': 'sam2.1_hiera_b+.yaml',
                        'sam2.1_hiera_small.pt': 'sam2.1_hiera_s.yaml',
                        'sam2.1_hiera_tiny.pt': 'sam2.1_hiera_t.yaml',
                        }

    model_base = os.path.join(os.path.dirname(__file__), "models", "SAM2.1")

    sam2_checkpoint_name = 'sam2.1_hiera_large.pt'

    sam2_checkpoint = os.path.join(model_base, sam2_checkpoint_name)
    model_cfg = "configs/sam2.1/" + model_config_map[sam2_checkpoint_name]

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    if image_type == "linearts":
        data_dir = 'linearts_layers'
    else:
        raise Exception('Unknown image_type:', image_type)

    if example_info_map is not None:
        mask_dilate_size = example_info_map[str(test_id)]["inbetweening_configs"]["mask_dilate_size"]
        order_type = example_info_map[str(test_id)]["inbetweening_configs"]["order_type"]  # [area, manual]
        manual_order = example_info_map[str(test_id)]["inbetweening_configs"]["manual_order"]  # set to None if not used
    else:
        mask_dilate_size = -8
        order_type = 'area'
        manual_order = None
    if prompt_type == "line_mask":
        mask_dilate_size = -2
    assert order_type in ['area', 'manual']
    if order_type == 'manual':
        assert manual_order is not None

    save_base_layer = os.path.join(inbetween_result_base, str(test_id), "layers", "[" + prompt_type + "]-[" + image_type + "]")
    os.makedirs(save_base_layer, exist_ok=True)

    vector_data_path = os.path.join(correspondence_result_base, "[1comb]", "vector-params", str(test_id) + "_ref.jsonl")
    with open(vector_data_path, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b = item['stroke_params']
            # stroke_data_b: a list of layers
            #   => layer: a list of stroke chains
            #     => stroke chain: a list of strokes
            #       => stroke: (4, 2)

    n_layers = len(stroke_data_b)
    layer_mask_lists = []  # len = n_frame

    for c_i, component in enumerate(stroke_data_b):
        curve_points = []
        for curve_i, curve in enumerate(component):  # (N, 4, 2)
            curve_points.append(curve)
        curve_points = np.concatenate(curve_points, axis=0)  # (N', 4, 2)
        curve_points = np.reshape(curve_points, (-1, 2))  # (N' * 4, 2)
        min_x, min_y = np.min(curve_points[:, 0]), np.min(curve_points[:, 1])
        max_x, max_y = np.max(curve_points[:, 0]), np.max(curve_points[:, 1])
        component_box = np.array([min_x, min_y, max_x, max_y], dtype=np.float32)

        component_img_path = os.path.join(inbetween_result_base, str(test_id), "linearts_layers", "layer%s" % c_i, "frame0.png")
        component_img = Image.open(component_img_path).convert('L')
        component_img = 1.0 - np.array(component_img, dtype=np.float32) / 255.0
        component_line_mask = component_img

        component_mask_path = os.path.join(raw_data_base, "layers/[both]/mask_raw", str(test_id), "%s_ref.png" % c_i)
        component_mask_img = Image.open(component_mask_path).convert('L')
        component_mask_img = np.array(component_mask_img, dtype=np.float32) / 255.0
        component_region_mask = component_mask_img

        ############################# Inference for each layer #############################
        # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
        video_dir = png_to_jpg(os.path.join(inbetween_result_base, str(test_id), data_dir, 'layer' + str(c_i)))

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        prompts = {}  # hold all the clicks we add for visualization
        inference_state = predictor.init_state(video_path=video_dir)

        input_box = component_box  # xyxy format
        input_line_mask = component_line_mask  # (H, W), [0-BG, 1-FG]
        input_region_mask = component_region_mask  # (H, W), [0-BG, 1-FG]

        ## Step-1: Add a first prompt on a frame

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = c_i  # give a unique id to each object we interact with (it can be any integers)

        if prompt_type == "box":
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=input_box,
            )
        elif prompt_type == "line_mask":
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                mask=input_line_mask,
            )
        elif prompt_type == "region_mask":
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                mask=input_region_mask,
            )
        else:
            raise Exception("Unknown prompt_type:", prompt_type)
        prompts[ann_obj_id] = input_box

        ## Step-2: Propagate the prompts to get the masklet across the video

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        assert len(video_segments) == len(frame_names)
        for out_frame_idx, video_segments_split in enumerate(video_segments):
            video_segments_split = video_segments[out_frame_idx]
            assert len(video_segments_split.items()) == 1

            ## Step-3: calculate bbox for each layer mask
            out_obj_id = ann_obj_id
            out_mask = video_segments_split[out_obj_id]
            layer_mask_info = {}
            layer_mask_info['obj_id'] = out_obj_id
            out_mask_proc = out_mask[0]
            if mask_dilate_size != 0:
                out_mask_proc = mask_dilate(out_mask_proc, mask_dilate_size)

            layer_mask_info['mask'] = out_mask_proc
            bbox, box_area = cal_mask_bbox(out_mask_proc)
            layer_mask_info['bbox'] = bbox
            layer_mask_info['area'] = box_area

            if order_type == 'manual':
                layer_mask_info['manual'] = manual_order[c_i]

            if len(layer_mask_lists) <= out_frame_idx:
                layer_mask_lists.append([layer_mask_info])
            else:
                layer_mask_lists[out_frame_idx].append(layer_mask_info)

    for out_frame_idx in range(len(layer_mask_lists)):
        layer_mask_list = layer_mask_lists[out_frame_idx]
        assert len(layer_mask_list) == n_layers
        ## Step-4: sort each mask
        sorted_layer_mask_list = sorted(layer_mask_list, key=(lambda x: x[order_type]))
        # smaller area is in the front

        ## Step-5: remove inclusion and obtain layers
        lineart_path = os.path.join(inbetween_result_base, str(test_id), 'linearts', str(out_frame_idx) + '.png')
        remove_inclusion_and_obtain_layers(out_frame_idx, sorted_layer_mask_list, lineart_path, save_base_layer,
                                            verbose=verbose, recolor_mask=False)


def multi_prompt_inference(correspondence_result_base, inbetween_result_base, test_id, image_prompt_types, example_info_map,
                           remove_intermediate_files=False, verbose=False):
    save_dir_name = "[both]" if len(image_prompt_types) == 2 else "[all]"

    if example_info_map is not None:
        order_type = example_info_map[str(test_id)]["inbetweening_configs"]["order_type"]  # [area, manual]
        manual_order = example_info_map[str(test_id)]["inbetweening_configs"]["manual_order"]  # set to None if not used
    else:
        order_type = 'area'
        manual_order = None
    assert order_type in ['area', 'manual']
    if order_type == 'manual':
        assert manual_order is not None

    save_base_layer = os.path.join(inbetween_result_base, str(test_id), "layers", save_dir_name)

    lineart_dir = os.path.join(inbetween_result_base, str(test_id), 'linearts')
    all_files = os.listdir(lineart_dir)
    all_files = [item for item in all_files if '.png' in item]
    file_num = len(all_files)

    vector_data_path = os.path.join(correspondence_result_base, "[1comb]", "vector-params", str(test_id) + "_ref.jsonl")
    with open(vector_data_path, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b = item['stroke_params']
            # stroke_data_b: a list of layers
            #   => layer: a list of stroke chains
            #     => stroke chain: a list of strokes
            #       => stroke: (4, 2)

    for out_frame_idx in range(file_num):
        ## Merge masks
        layer_mask_list = []
        for c_i, component in enumerate(stroke_data_b):
            layer_mask_info = {}
            layer_mask_info['obj_id'] = c_i

            for p_i, (image_type_, prompt_type_) in enumerate(image_prompt_types):
                raw_mask_path = os.path.join(inbetween_result_base, str(test_id), "layers", "[" + prompt_type_ + "]-[" + image_type_ + "]", "mask_raw",
                                                str(c_i) + "_fra=" + str(out_frame_idx) + ".png")
                raw_mask = Image.open(raw_mask_path).convert('L')
                raw_mask = np.array(raw_mask) > 0

                if p_i == 0:
                    out_mask_proc = raw_mask
                else:
                    out_mask_proc = np.logical_or(out_mask_proc, raw_mask)

            layer_mask_info['mask'] = out_mask_proc
            bbox, box_area = cal_mask_bbox(out_mask_proc)
            layer_mask_info['bbox'] = bbox
            layer_mask_info['area'] = box_area

            if order_type == 'manual':
                layer_mask_info['manual'] = manual_order[c_i]
            layer_mask_list.append(layer_mask_info)

        ## Step-4: sort each mask
        sorted_layer_mask_list = sorted(layer_mask_list, key=(lambda x: x[order_type]))
        # smaller area is in the front

        ## Step-5: remove inclusion and obtain layers
        lineart_path = os.path.join(inbetween_result_base, str(test_id), 'linearts', str(out_frame_idx) + '.png')
        remove_inclusion_and_obtain_layers(out_frame_idx, sorted_layer_mask_list, lineart_path, save_base_layer,
                                            verbose=verbose)

    if remove_intermediate_files:
        for p_i, (image_type_, prompt_type_) in enumerate(image_prompt_types):
            intermediate_files_dir = os.path.join(inbetween_result_base, str(test_id), "layers", "[" + prompt_type_ + "]-[" + image_type_ + "]")
            assert os.path.isdir(intermediate_files_dir)
            shutil.rmtree(intermediate_files_dir)

        intermediate_files_dir = os.path.join(save_base_layer, 'mask_raw')
        assert os.path.isdir(intermediate_files_dir)
        shutil.rmtree(intermediate_files_dir)


def inference_multi_img_main(raw_data_base, image_id):
    correspondence_result_base = "outputs/stroke_correspondence_results"
    inbetween_result_base = "outputs/inbetweening_results"
    if gen_time > 0:
        correspondence_result_base += '-Gen%d' % gen_time
        inbetween_result_base += '-Gen%d' % gen_time

    layering_method = example_info_map[str(image_id)]["inbetweening_configs"]["layering_method"]
    if "+" not in layering_method:
        image_prompt_types_group = [
            [("linearts", layering_method)]
        ]
    else:
        image_prompt_types_group = []
        image_prompt_types_group_comb = []
        layering_method_items = layering_method.split("+")
        for item in layering_method_items:
            image_prompt_types_group.append([("linearts", item)])
            image_prompt_types_group_comb.append(("linearts", item))
        image_prompt_types_group.append(image_prompt_types_group_comb)

    for image_prompt_types in image_prompt_types_group:
        if len(image_prompt_types) == 1:
            image_type = image_prompt_types[0][0]
            prompt_type = image_prompt_types[0][1]
            single_prompt_inference(raw_data_base, correspondence_result_base, inbetween_result_base, image_id, image_type, prompt_type, example_info_map)
        else:
            multi_prompt_inference(correspondence_result_base, inbetween_result_base, image_id, image_prompt_types, example_info_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_base', default=test_data_base, type=str, help="data base path")
    parser.add_argument('--image_id', default=test_img_id, type=int, help='image ID for evaluation')
    args = parser.parse_args()

    inference_multi_img_main(args.data_base, args.image_id)
    