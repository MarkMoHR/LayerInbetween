import argparse
import os

import cv2
import jsonlines
import numpy as np

from configs.example_configs import test_data_base, test_img_id, gen_time, example_info_map


def main(raw_data_base, process_img_id, vis_single_layer=False):
    correspondence_result_base = "outputs/stroke_correspondence_results"
    inbetween_result_base = "outputs/inbetweening_results"
    if gen_time > 0:
        correspondence_result_base += '-Gen%d' % gen_time
        inbetween_result_base += '-Gen%d' % gen_time

    layering_method = example_info_map[str(process_img_id)]["inbetweening_configs"]["layering_method"]

    if "+" not in layering_method:
        layer_mask_dir = '[%s]-[linearts]' % layering_method
    else:
        items = layering_method.split("+")
        layer_mask_dir = "[both]" if len(items) == 2 else "[all]"

    layer_lineart_base = os.path.join(inbetween_result_base, str(process_img_id), 'linearts_layers')

    save_base = os.path.join(inbetween_result_base, str(process_img_id), 'linearts_occ_resolve')
    save_base_mask = os.path.join(inbetween_result_base, str(process_img_id), 'linearts_occ_resolve_mask')
    os.makedirs(save_base, exist_ok=True)
    os.makedirs(save_base_mask, exist_ok=True)
    if vis_single_layer:
        save_base_single = os.path.join(save_base, 'single')
        os.makedirs(save_base_single, exist_ok=True)

    all_files = os.listdir(os.path.join(layer_lineart_base, 'layer0'))
    all_files = [item for item in all_files if '.png' in item]

    vector_data_path = os.path.join(correspondence_result_base, "[1comb]", "vector-params", str(process_img_id) + "_ref.jsonl")
    with open(vector_data_path, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b = item['stroke_params']
            # stroke_data_b: a list of layers
            #   => layer: a list of stroke chains
            #     => stroke chain: a list of strokes
            #       => stroke: (4, 2)

    input_img_path = os.path.join(raw_data_base, "raster_black", str(process_img_id) + "_ref.png")
    input_img = cv2.imread(input_img_path)
    image_size = input_img.shape[0]

    for filename in all_files:
        img_name= filename[:-4]
        frame_idx = img_name[5:]

        frame_canvas = np.zeros(shape=(image_size, image_size), dtype=np.float32)  # (H, W), [0-BG, 255-stroke]

        for c_i in range(len(stroke_data_b)):
            layer_lineart_dir = os.path.join(layer_lineart_base, 'layer' + str(c_i))
            component_img_path = os.path.join(layer_lineart_dir, 'frame' + str(frame_idx) + '.png')
            component_img = cv2.imread(component_img_path)
            component_img = component_img[:, :, 0]  # (H, W), [0-stroke, 255-BG]

            ## resolve occlusion
            layer_mask_img = os.path.join(inbetween_result_base, str(process_img_id), 'layers', layer_mask_dir, 'mask_proc',
                                          str(c_i) + "_fra=" + str(frame_idx) + ".png")
            layer_mask_img = cv2.imread(layer_mask_img)
            layer_mask = layer_mask_img[:, :, 0].astype(np.float32) / 255.0  # (H, W), [0-BG, 1-FG]

            component_img_resolved = (255.0 - component_img.astype(np.float32)) * layer_mask
            if vis_single_layer:
                component_img_resolved_vis = 255.0 - np.clip(component_img_resolved, 0.0, 255.0)  # (H, W), [0-stroke, 255-BG]
                component_img_resolved_vis = component_img_resolved_vis.astype(np.uint8)
                save_path = os.path.join(save_base_single, filename[:-4] + '-' + str(c_i) +  '.png')
                cv2.imwrite(save_path, component_img_resolved_vis)

            save_base_mask_ci = os.path.join(save_base_mask, 'layer' + str(c_i))
            os.makedirs(save_base_mask_ci, exist_ok=True)
            component_img_resolved_mask = (255.0 - component_img.astype(np.float32)) * (1.0 - layer_mask)  # (H, W), [0-BG, mask]
            component_img_resolved_mask = component_img_resolved_mask.astype(np.uint8)
            save_path = os.path.join(save_base_mask_ci, frame_idx + '.png')
            cv2.imwrite(save_path, component_img_resolved_mask)

            frame_canvas += component_img_resolved

        frame_canvas = 255.0 - np.clip(frame_canvas, 0.0, 255.0)  # (H, W), [0-stroke, 255-BG]
        frame_canvas = frame_canvas.astype(np.uint8)

        save_path = os.path.join(save_base, frame_idx + '.png')
        cv2.imwrite(save_path, frame_canvas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_base', default=test_data_base, type=str, help="data base path")
    parser.add_argument('--image_id', default=test_img_id, type=int, help='image ID for evaluation')
    args = parser.parse_args()

    main(args.data_base, args.image_id)
