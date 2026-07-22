import os
import argparse
import jsonlines
from PIL import Image
import numpy as np

from data_preprocessing.utils.draw_sketch import draw_sketch_cairo
from configs.example_configs import test_data_base, test_img_id, gen_time


def single_stroke_interpolation(ref_stroke_, tar_stroke_, num_intermediate=3):
    """
    Interpolate between two strokes (ref_stroke and tar_stroke) to generate intermediate strokes.
    Each stroke is represented as a 4x2 array (for cubic Bezier curves).
    """
    ref_stroke = np.array(ref_stroke_)  # shape: (4, 2)
    tar_stroke = np.array(tar_stroke_)  # shape: (4, 2)

    intermediate_strokes = []  # list of intermediate strokes, each of shape (4, 2)
    for i in range(1, num_intermediate + 1):
        t = i / (num_intermediate + 1)
        intermediate_stroke = (1 - t) * ref_stroke + t * tar_stroke
        intermediate_strokes.append(intermediate_stroke.tolist())

    return intermediate_strokes


def main(database, process_img_id, num_intermediate, line_thickness=3):
    output_base = "outputs/stroke_correspondence_results"
    result_base = "outputs/inbetweening_results"
    if gen_time > 0:
        output_base += '-Gen%d' % gen_time
        result_base += '-Gen%d' % gen_time
    result_base = os.path.join(result_base, str(process_img_id))

    vector_params_base = os.path.join(output_base, "[1comb]", "vector-params")
    ref_jsonl_path = os.path.join(vector_params_base, str(process_img_id) + "_ref.jsonl")
    tar_jsonl_path = os.path.join(vector_params_base, str(process_img_id) + "_tar.jsonl")

    with open(ref_jsonl_path, "r+") as f:
        for item in jsonlines.Reader(f):
            ref_stroke_data_b = item['stroke_params']
            # stroke_data_b: a list of layers
            #   => layer: a list of stroke chains
            #     => stroke chain: a list of strokes
            #       => stroke: (4, 2)
    
    with open(tar_jsonl_path, "r+") as f:
        for item in jsonlines.Reader(f):
            tar_stroke_data_b = item['stroke_params']
            # stroke_data_b: a list of layers
            #   => layer: a list of stroke chains
            #     => stroke chain: a list of strokes
            #       => stroke: (4, 2)
    
    assert len(ref_stroke_data_b) == len(tar_stroke_data_b)

    intermediate_stroke_data_b = [[] for _ in range(num_intermediate)]  # list of sketches
    for c_i in range(len(ref_stroke_data_b)):
        ref_component = ref_stroke_data_b[c_i]
        tar_component = tar_stroke_data_b[c_i]
        assert len(ref_component) == len(tar_component)

        intermediate_component = [[] for _ in range(num_intermediate)]  # list of components
        for curve_i in range(len(ref_component)):
            ref_curve = ref_component[curve_i]  # (N, 4, 2) for bezier
            tar_curve = tar_component[curve_i]
            assert len(ref_curve) == len(tar_curve)

            intermediate_curve = [[] for _ in range(num_intermediate)]  # list of (N, 4, 2)
            for stroke_i in range(len(ref_curve)):
                ref_stroke = ref_curve[stroke_i]  # (4, 2)
                tar_stroke = tar_curve[stroke_i]
                assert len(ref_stroke) == len(tar_stroke)

                intermediate_strokes = single_stroke_interpolation(ref_stroke, tar_stroke, num_intermediate=num_intermediate)
                # list of intermediate strokes, K * (4, 2)
                for k in range(num_intermediate):
                    intermediate_curve[k].append(intermediate_strokes[k])

            for k in range(num_intermediate):
                intermediate_component[k].append(intermediate_curve[k])

        for k in range(num_intermediate):
            intermediate_stroke_data_b[k].append(intermediate_component[k])

    full_stroke_data_b = [ref_stroke_data_b] + intermediate_stroke_data_b + [tar_stroke_data_b]  # list of sketches

    input_img_path = os.path.join(database, "raster_black", str(process_img_id) + '_ref.png')
    img = Image.open(input_img_path).convert('RGB')
    img_width, img_height = img.width, img.height
    assert img_width == img_height
    
    result_params_base = os.path.join(result_base, "vector-params")
    result_linearts_base = os.path.join(result_base, "linearts")
    result_linearts_layer_base = os.path.join(result_base, "linearts_layers")
    os.makedirs(result_params_base, exist_ok=True)
    os.makedirs(result_linearts_base, exist_ok=True)
    os.makedirs(result_linearts_layer_base, exist_ok=True)

    for sketch_i, sketch_stroke_data in enumerate(full_stroke_data_b):
        vector_data_save_path = os.path.join(result_params_base, str(sketch_i) + '.jsonl')
        vector_data = {}
        vector_data['stroke_params'] = sketch_stroke_data
        with jsonlines.open(vector_data_save_path, mode='w') as json_writer:
            json_writer.write(vector_data)

        # Draw the sketch and save as an image
        vector_vis_path = os.path.join(result_linearts_base, str(sketch_i) + ".png")
        draw_sketch_cairo(sketch_stroke_data, vector_vis_path, is_bezier=True,
                          side=img_width, line_diameter=line_thickness)
        
        # Draw each layer separately and save as images
        for c_i in range(len(sketch_stroke_data)):
            layer_dir = os.path.join(result_linearts_layer_base, 'layer' + str(c_i))
            os.makedirs(layer_dir, exist_ok=True)

            component = sketch_stroke_data[c_i]
            vector_component_vis_path = os.path.join(layer_dir, "frame" + str(sketch_i) + ".png")
            draw_sketch_cairo([component], vector_component_vis_path, is_bezier=True,
                              side=img_width, line_diameter=line_thickness)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_base', default=test_data_base, type=str, help="data base path")
    parser.add_argument('--image_id', default=test_img_id, type=int, help='image ID for evaluation')
    parser.add_argument('--num_intermediate', default=15, type=int, help='number of intermediate strokes to generate')
    parser.add_argument('--line_thickness', default=3, type=int, help='thickness of the lines in the output image')
    args = parser.parse_args()
    
    main(args.data_base, args.image_id, args.num_intermediate, args.line_thickness)
