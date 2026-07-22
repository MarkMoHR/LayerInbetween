import argparse
import os

from configs.example_configs import gpu_id, test_data_base, test_img_id, example_info_map
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from PIL import Image
import jsonlines
import shutil

from build_sam import build_sam2_video_predictor
from sam2_common import cal_mask_bbox, cal_bbox_inclusion_area, generate_colors2, mask_dilate


def png_to_jpg(data_dir, img_id, img_path_format):
    save_dir = os.path.join(data_dir, 'video_frames', str(img_id))
    os.makedirs(save_dir, exist_ok=True)

    for di, data_split in enumerate(["ref", "tar"]):
        img_path = os.path.join(data_dir, img_path_format % (str(img_id), data_split))
        img_path_c = os.path.join(save_dir, str(di) + ".jpg")
        if not os.path.exists(img_path_c):
            img = Image.open(img_path).convert('RGB')
            img.save(img_path_c)
    return save_dir


def remove_inclusion_and_obtain_layers(split_type, sorted_layer_mask_list, lineart_path, test_id, save_base_layer,
                                       inclusion_rate_threshold, verbose=True):
    save_base_layer_raw_mask = os.path.join(save_base_layer, 'mask_raw', str(test_id))
    save_base_layer_proc_mask = os.path.join(save_base_layer, 'mask_proc', str(test_id))
    os.makedirs(save_base_layer_raw_mask, exist_ok=True)
    os.makedirs(save_base_layer_proc_mask, exist_ok=True)
    if verbose:
        save_base_layer_image = os.path.join(save_base_layer, 'image', str(test_id))
        save_base_layer_image_vis = os.path.join(save_base_layer, 'image_vis')
        os.makedirs(save_base_layer_image, exist_ok=True)
        os.makedirs(save_base_layer_image_vis, exist_ok=True)

    ## Step-5: remove inclusion
    lineart = Image.open(lineart_path).convert("L")
    lineart = np.array(lineart, dtype=np.uint8)

    num_mask = len(sorted_layer_mask_list)

    layer_vis = np.ones_like(lineart, dtype=np.float32) * 255.0
    layer_vis = np.tile(np.expand_dims(layer_vis, axis=-1), (1, 1, 3))
    colors = generate_colors2(num_mask)  # list of (3), in [0., 1.]

    for m_i in range(num_mask):
        obj_id_i = sorted_layer_mask_list[m_i]['obj_id']
        mask_i = sorted_layer_mask_list[m_i]['mask']
        bbox_i = sorted_layer_mask_list[m_i]['bbox']
        mask_i_proc = np.copy(mask_i)

        for m_j in range(m_i):  # iterate all smaller masks
            mask_j = sorted_layer_mask_list[m_j]['mask']
            bbox_j = sorted_layer_mask_list[m_j]['bbox']
            area_j = sorted_layer_mask_list[m_j]['area']
            if bbox_j is None:
                continue

            inclusion_area = cal_bbox_inclusion_area(bbox_i, bbox_j)
            assert inclusion_area <= area_j
            inclusion_rate = float(inclusion_area) / float(area_j)

            if inclusion_rate > inclusion_rate_threshold:
                mask_i_proc = np.logical_and(mask_i_proc, np.logical_not(mask_j))

        ## Step-6: filter lineart image to obtain layers
        layer = np.copy(lineart)
        layer[np.logical_not(mask_i_proc)] = 255

        layer_rbg = 255.0 - np.tile(np.expand_dims(layer, axis=-1), (1, 1, 3)).astype(np.float32)
        layer_rbg *= 1.0 - np.expand_dims(np.expand_dims(np.array(colors[obj_id_i])[::-1], axis=0), axis=0)
        layer_rbg = 255.0 - layer_rbg

        layer_vis[mask_i_proc] = layer_rbg[mask_i_proc]

        save_path_raw_mask = os.path.join(save_base_layer_raw_mask, str(obj_id_i) + "_" + split_type + ".png")
        raw_mask = np.zeros_like(mask_i, dtype=np.uint8)
        raw_mask[mask_i] = 255
        raw_mask = Image.fromarray(raw_mask, 'L')
        raw_mask.save(save_path_raw_mask, 'PNG')

        save_path_proc_mask = os.path.join(save_base_layer_proc_mask, str(obj_id_i) + "_" + split_type + ".png")
        proc_mask = np.zeros_like(mask_i_proc, dtype=np.uint8)
        proc_mask[mask_i_proc] = 255
        proc_mask = Image.fromarray(proc_mask, 'L')
        proc_mask.save(save_path_proc_mask, 'PNG')

        if verbose:
            save_path_layer_img = os.path.join(save_base_layer_image, str(obj_id_i) + "_" + split_type + ".png")
            layer = Image.fromarray(layer, 'L')
            layer.save(save_path_layer_img, 'PNG')

    if verbose:
        save_path_layer_vis = os.path.join(save_base_layer_image_vis, str(test_id) + "_" + split_type + ".png")
        layer_vis = Image.fromarray(layer_vis.astype(np.uint8), 'RGB')
        layer_vis.save(save_path_layer_vis, 'PNG')


def single_prompt_inference(data_base, test_ids, image_type, prompt_type, inclusion_rate_threshold, data_type,
                            data_base_video=None, verbose=True):
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

    save_base_layer = os.path.join(data_base, 'layers', "[" + prompt_type + "]-[" + image_type + "]")

    # save_root_vis = "outputs_video/prompt_" + prompt_type
    # save_base_vis = os.path.join(save_root_vis, sam2_checkpoint_name[:sam2_checkpoint_name.rfind('.')], image_type)
    # os.makedirs(save_base_vis, exist_ok=True)

    for i, test_id in enumerate(test_ids):
        if image_type == "linearts":
            data_dir = 'raster_black'
        elif image_type == "depth":
            data_dir = 'depth/vis'
        elif image_type == "depth_overlap":
            data_dir = 'depth/vis/overlap'

        # print('processing', i, '/', len(test_ids), ':', test_id)

        mask_dilate_size = example_info_map[str(test_id)]["target_layer_gen_configs"]["mask_dilate_size"]

        # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
        if data_base_video is None:
            data_base_video = data_base

        if data_type != 'real' and image_type == "linearts":
            img_path_format = "sketch_%s_bezier-%s.png"  # % (str(test_id), split_type)
        else:
            img_path_format = "%s_%s.png"  # % (str(test_id), split_type)
        video_dir = png_to_jpg(os.path.join(data_base_video, data_dir), test_id, img_path_format)

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # frame_names_linearts = [str(test_id) + "_ref.png", str(test_id) + "_tar.png"]

        vector_data_path = os.path.join(data_base, "vector-params", str(test_id) + "_ref.jsonl")
        with open(vector_data_path, "r+") as f:
            for item in jsonlines.Reader(f):
                stroke_data_b = item['stroke_params']
                # stroke_data_b: a list of layers
                #   => layer: a list of stroke chains
                #     => stroke chain: a list of strokes
                #       => stroke: (4, 2)

        component_boxes = []
        component_centers = []
        component_masks = []
        for c_i, component in enumerate(stroke_data_b):
            curve_points = []
            for curve_i, curve in enumerate(component):  # (N, 4, 2)
                curve_points.append(curve)
            curve_points = np.concatenate(curve_points, axis=0)  # (N', 4, 2)
            curve_points = np.reshape(curve_points, (-1, 2))  # (N' * 4, 2)
            min_x, min_y = np.min(curve_points[:, 0]), np.min(curve_points[:, 1])
            max_x, max_y = np.max(curve_points[:, 0]), np.max(curve_points[:, 1])
            center = [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0]
            component_boxes.append(np.array([min_x, min_y, max_x, max_y], dtype=np.float32))
            component_centers.append(np.array(center))

            component_img_path = os.path.join(data_base, "raster_black_component", str(test_id), "component_%s-ref.png" % c_i)
            component_img = Image.open(component_img_path).convert('L')
            component_img = 1.0 - np.array(component_img, dtype=np.float32) / 255.0
            component_masks.append(component_img)

        prompts = {}  # hold all the clicks we add for visualization
        inference_state = predictor.init_state(video_path=video_dir)

        for c_i in range(len(component_boxes)):
            input_box = component_boxes[c_i]  # xyxy format
            input_mask = component_masks[c_i]  # (H, W), [0-BG, 1-FG]

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
            elif prompt_type == "mask":
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    mask=input_mask,
                )
            else:
                raise Exception("Unknown prompt_type:", prompt_type)
            prompts[ann_obj_id] = input_box

            # # show the results on the current (interacted) frame
            # plt.figure(figsize=(9, 6))
            # plt.title(f"frame {ann_frame_idx}")
            # plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
            # show_box(input_box, plt.gca())
            # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
            # plt.show()

        ## Step-2: Propagate the prompts to get the masklet across the video

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        assert len(video_segments) == 2
        for f_i, video_segments_split in enumerate(video_segments):
            split_type = "ref" if f_i == 0 else "tar"
            video_segments_split = video_segments[f_i]

            ## Step-3: calculate bbox for each layer mask
            layer_mask_list = []
            for out_obj_id, out_mask in video_segments_split.items():
                layer_mask_info = {}
                layer_mask_info['obj_id'] = out_obj_id
                out_mask_proc = out_mask[0]
                if mask_dilate_size != 0:
                    out_mask_proc = mask_dilate(out_mask_proc, mask_dilate_size)

                layer_mask_info['mask'] = out_mask_proc
                bbox, box_area = cal_mask_bbox(out_mask_proc)
                layer_mask_info['bbox'] = bbox
                layer_mask_info['area'] = box_area
                layer_mask_list.append(layer_mask_info)

            ## Step-4: sort each mask according to the area
            sorted_layer_mask_list = sorted(layer_mask_list, key=(lambda x: x['area']))
            # smaller area is in the front

            ## Step-5: remove inclusion and obtain layers
            if data_type == 'real':
                lineart_path = os.path.join(data_base, 'raster_black', str(test_id) + "_" + split_type + ".png")
            else:
                lineart_path = os.path.join(data_base, 'raster_black', "sketch_" + str(test_id) + "_bezier-" + split_type + ".png")
            remove_inclusion_and_obtain_layers(split_type, sorted_layer_mask_list, lineart_path, test_id, save_base_layer,
                                               inclusion_rate_threshold, verbose=verbose)


def multi_prompt_inference(data_base, test_ids, image_prompt_types, inclusion_rate_threshold, data_type,
                           remove_intermediate_files=False, verbose=True):
    save_base_layer = os.path.join(data_base, 'layers', "[both]")

    for i, test_id in enumerate(test_ids):
        # print('processing', test_id)

        vector_data_path = os.path.join(data_base, "vector-params", str(test_id) + "_ref.jsonl")
        with open(vector_data_path, "r+") as f:
            for item in jsonlines.Reader(f):
                stroke_data_b = item['stroke_params']
                # stroke_data_b: a list of layers
                #   => layer: a list of stroke chains
                #     => stroke chain: a list of strokes
                #       => stroke: (4, 2)

        for split_type in ["ref", "tar"]:
            ## Merge masks
            layer_mask_list = []
            for c_i, component in enumerate(stroke_data_b):
                layer_mask_info = {}
                layer_mask_info['obj_id'] = c_i

                for p_i, (image_type_, prompt_type_) in enumerate(image_prompt_types):
                    raw_mask_path = os.path.join(data_base, 'layers', "[" + prompt_type_ + "]-[" + image_type_ + "]", "mask_raw",
                                                 str(test_id), str(c_i) + "_" + split_type + ".png")
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
                layer_mask_list.append(layer_mask_info)

            ## Step-4: sort each mask according to the area
            sorted_layer_mask_list = sorted(layer_mask_list, key=(lambda x: x['area']))
            # smaller area is in the front

            ## Step-5: remove inclusion and obtain layers
            if data_type == 'real':
                lineart_path = os.path.join(data_base, 'raster_black', str(test_id) + "_" + split_type + ".png")
            else:
                lineart_path = os.path.join(data_base, 'raster_black', "sketch_" + str(test_id) + "_bezier-" + split_type + ".png")
            remove_inclusion_and_obtain_layers(split_type, sorted_layer_mask_list, lineart_path, test_id, save_base_layer,
                                               inclusion_rate_threshold, verbose=verbose)

    if remove_intermediate_files:
        for p_i, (image_type_, prompt_type_) in enumerate(image_prompt_types):
            intermediate_files_dir = os.path.join(data_base, 'layers', "[" + prompt_type_ + "]-[" + image_type_ + "]")
            assert os.path.isdir(intermediate_files_dir)
            shutil.rmtree(intermediate_files_dir)

        # intermediate_files_dir = os.path.join(save_base_layer, 'mask_raw')
        # assert os.path.isdir(intermediate_files_dir)
        # shutil.rmtree(intermediate_files_dir)


def inference_single_img_main(data_base, image_id, inclusion_rate_threshold=0.8):
    target_layer_method = example_info_map[str(image_id)]["target_layer_method"]
    if target_layer_method == "box_depth":
        image_prompt_types_group = [
            [("depth", "box")]
        ]
    elif target_layer_method == "box_depth_ol":
        image_prompt_types_group = [
            [("depth_overlap", "box")]
        ]
    elif target_layer_method == "mask_line":
        image_prompt_types_group = [
            [("linearts", "mask")]
        ]
    elif target_layer_method == "box_depth+mask_line":
        image_prompt_types_group = [
            [("depth", "box")],
            [("linearts", "mask")],
            [("depth", "box"), ("linearts", "mask")]
        ]
    elif target_layer_method == "box_depth_ol+mask_line":
        image_prompt_types_group = [
            [("depth_overlap", "box")],
            [("linearts", "mask")],
            [("depth_overlap", "box"), ("linearts", "mask")]
        ]
    else:
        raise Exception('Unknown target_layer_method:', target_layer_method)

    for image_prompt_types in image_prompt_types_group:
        if len(image_prompt_types) == 1:
            image_type = image_prompt_types[0][0]
            prompt_type = image_prompt_types[0][1]
            single_prompt_inference(data_base, [image_id], image_type, prompt_type, inclusion_rate_threshold, data_type='real')
        else:
            multi_prompt_inference(data_base, [image_id], image_prompt_types, inclusion_rate_threshold, data_type='real')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_base', default=test_data_base, type=str, help="data base path")
    parser.add_argument('--image_id', default=test_img_id, type=int, help='image ID for evaluation')
    args = parser.parse_args()

    inference_single_img_main(args.data_base, args.image_id)