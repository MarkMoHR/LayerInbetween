import os
import numpy as np
import shutil
import jsonlines
from PIL import Image

from utils.svg_parse import parse_svg
from utils.draw_sketch import draw_sketch_cairo, draw_sketch_stroke_chain_cairo, draw_sketch_stroke_cairo
from utils.curve_grouping import curve_grouping_flow, curve_grouping_depth, get_optical_flow_dir_name
from utils.curve_splitting import curve_splitting
from utils.optical_transform import component_translation_with_optical_flow, curve_translation_with_optical_flow

from configs.example_configs import test_data_base, test_img_id, gen_time, example_info_map


def load_svg(database, line_thickness, process_img_ids=None, img_size_default=768):
    svg_base = os.path.join(database, 'svg')
    img_base = os.path.join(database, 'raster_black')

    save_base_parameter = os.path.join(database, 'vector-params-raw')
    save_base_color_separate = os.path.join(database, 'vector-vis-raw')
    # save_base_color_separate_chain = os.path.join(database, 'vector-vis-chain')
    os.makedirs(save_base_parameter, exist_ok=True)
    os.makedirs(save_base_color_separate, exist_ok=True)
    # os.makedirs(save_base_color_separate_chain, exist_ok=True)

    for process_img_id in process_img_ids:
        file_name = str(process_img_id) + '.svg'
        svg_file_path = os.path.join(svg_base, file_name)
        img_file_path = os.path.join(img_base, file_name[:-4] + '_ref.png')
        if os.path.exists(img_file_path):
            img = Image.open(img_file_path).convert('RGB')
            img_width, img_height = img.width, img.height
            assert img_width == img_height
        else:
            img_width, img_height = img_size_default, img_size_default

        view_sizes, curves_list = parse_svg(svg_file_path)
        # curves_list: list of (N_point, 2), in view size
        assert view_sizes[0] == view_sizes[1]
        view_size = view_sizes[0]

        stroke_data_new_b = []  # list of (N, 4, 2), in image size
        total_stroke_num = 0
        for single_curve in curves_list:
            single_curve_norm = single_curve / float(view_size) * float(img_width)  # (N_point, 2)
            stroke_num = (len(single_curve_norm) - 1) // 3
            total_stroke_num += stroke_num

            single_curve_b_norm = []  # (N, 4, 2)
            for s_i in range(stroke_num):
                single_stroke_norm = single_curve_norm[s_i * 3: s_i * 3 + 4, :]  # (4, 2)
                single_curve_b_norm.append(single_stroke_norm.tolist())
            stroke_data_new_b.append(single_curve_b_norm)

        # print('total_stroke_num', total_stroke_num)

        save_jsonl_path = os.path.join(save_base_parameter, file_name[:-4] + '_ref.jsonl')
        vector_data_ref = {}
        vector_data_ref['stroke_params'] = stroke_data_new_b
        with jsonlines.open(save_jsonl_path, mode='w') as json_writer:
            json_writer.write(vector_data_ref)

        vector_vis_path = os.path.join(img_base, file_name[:-4] + "_ref_rendered.png")
        draw_sketch_cairo([stroke_data_new_b], vector_vis_path, is_bezier=True,
                          side=img_width, line_diameter=line_thickness)

        vector_vis_path = os.path.join(save_base_color_separate, file_name[:-4] + "_ref.png")
        draw_sketch_stroke_cairo([stroke_data_new_b], vector_vis_path, is_bezier=True,
                                 side=img_width, line_diameter=line_thickness)

        # vector_vis_path = os.path.join(save_base_color_separate_chain, file_name[:-4] + "_ref.png")
        # stroke_data_new_b_ = [[item] for item in stroke_data_new_b]
        # draw_sketch_stroke_chain_cairo(stroke_data_new_b_, vector_vis_path, is_bezier=True,
        #                                side=img_width, line_diameter=line_thickness)


def gen_stroke_chain_images(database, line_thickness, process_img_ids=None):
    print('============== gen_stroke_chain_images ==============')

    vector_data_dir = os.path.join(database, 'vector-params-split')
    postfix = '_ref.jsonl'
    all_files = os.listdir(vector_data_dir)
    all_files = [item for item in all_files if postfix in item]
    all_files.sort()

    stroke_image_base = os.path.join(database, 'raster_black_stroke_chain')
    os.makedirs(stroke_image_base, exist_ok=True)

    for fi, filename in enumerate(all_files):
        img_index = filename[:filename.find('_')]
        img_index = int(img_index)
        if process_img_ids is not None and img_index not in process_img_ids:
            continue

        # print('Processing', fi, '/', len(all_files), ':', img_index)

        stroke_image_dir = os.path.join(stroke_image_base, str(img_index))
        if os.path.exists(stroke_image_dir):
            shutil.rmtree(stroke_image_dir)
        os.makedirs(stroke_image_dir, exist_ok=True)

        vector_data_path = os.path.join(vector_data_dir, filename)
        with open(vector_data_path, "r+") as f:
            for item in jsonlines.Reader(f):
                curve_b_list = item['stroke_params']  # curve list => stroke list (N', 4, 2)

        ref_img_path = os.path.join(database, 'raster_black', str(img_index) + '_ref.png')
        if not os.path.exists(ref_img_path):
            continue

        ref_img = Image.open(ref_img_path)
        image_size = ref_img.height

        for curve_i in range(len(curve_b_list)):
            curve_b_points = curve_b_list[curve_i]  # list (N') of (4, 2)

            outpath = os.path.join(stroke_image_dir, "curve_%s.png" % curve_i)
            stroke_data_b_single = [[curve_b_points]]
            draw_sketch_cairo(stroke_data_b_single, outpath, is_bezier=True,
                              side=image_size, line_diameter=line_thickness)


def vector_curve_splitting(database, line_thickness, process_img_ids=None):
    print('============== vector_curve_splitting ==============')

    vector_data_dir = os.path.join(database, 'vector-params-raw')
    img_dir = os.path.join(database, 'raster_black')
    postfix = '_ref.jsonl'
    all_files = os.listdir(vector_data_dir)
    all_files = [item for item in all_files if postfix in item]
    all_files.sort()

    save_base_parameter = os.path.join(database, 'vector-params-split')
    save_base_color_separate = os.path.join(database, 'vector-vis-split')
    save_base_color_separate_chain = os.path.join(database, 'vector-vis-chain')
    os.makedirs(save_base_parameter, exist_ok=True)
    os.makedirs(save_base_color_separate, exist_ok=True)
    os.makedirs(save_base_color_separate_chain, exist_ok=True)

    for fi, filename in enumerate(all_files):
        img_index = filename[:filename.find('_')]
        img_index = int(img_index)
        if process_img_ids is not None and img_index not in process_img_ids:
            continue

        # print('Processing', fi, '/', len(all_files), ':', img_index)

        img_file_path = os.path.join(img_dir, str(img_index) + '_ref.png')
        img = Image.open(img_file_path).convert('RGB')
        img_width, img_height = img.width, img.height
        assert img_width == img_height

        vector_data_path_ref = os.path.join(vector_data_dir, str(img_index) + '_ref.jsonl')
        with open(vector_data_path_ref, "r+") as f:
            for item in jsonlines.Reader(f):
                stroke_data_b_raw = item['stroke_params']  # curve list => stroke list (N', 4, 2)

        stroke_data_b_split = []  # list of (N', 4, 2), in image size
        total_stroke_num_new = 0
        for curve_i in range(len(stroke_data_b_raw)):
            stroke_list = stroke_data_b_raw[curve_i]  # (N, 4, 2)
            split_stroke_list, _, _ = curve_splitting(stroke_list)  # list (N') of (4, 2)
            stroke_data_b_split.append(split_stroke_list)
            total_stroke_num_new += len(split_stroke_list)

        print('total_stroke_num_new', total_stroke_num_new)

        vector_data_save_path = os.path.join(save_base_parameter, str(img_index) + '_ref.jsonl')
        vector_data = {}
        vector_data['stroke_params'] = stroke_data_b_split
        with jsonlines.open(vector_data_save_path, mode='w') as json_writer:
            json_writer.write(vector_data)

        vector_vis_path = os.path.join(save_base_color_separate, str(img_index) + "_ref.png")
        draw_sketch_stroke_cairo([stroke_data_b_split], vector_vis_path, is_bezier=True,
                                 side=img_width, line_diameter=line_thickness)

        vector_vis_path = os.path.join(save_base_color_separate_chain, str(img_index) + "_ref.png")
        stroke_data_new_b_ = [[item] for item in stroke_data_b_split]
        draw_sketch_stroke_chain_cairo(stroke_data_new_b_, vector_vis_path, is_bezier=True,
                                       side=img_width, line_diameter=line_thickness)


def curve_grouping(database, line_thickness, optical_flow_method, use_distance_transform, should_merge_single_curve, process_img_ids=None):
    print('============== curve_grouping ==============')
    vector_data_dir = os.path.join(database, 'vector-params-split')
    postfix = '_ref.jsonl'
    all_files = os.listdir(vector_data_dir)
    all_files = [item for item in all_files if postfix in item]
    all_files.sort()

    save_base_parameter = os.path.join(database, 'vector-params')
    os.makedirs(save_base_parameter, exist_ok=True)

    for fi, filename in enumerate(all_files):
        img_index = filename[:filename.find('_')]
        img_index = int(img_index)
        if process_img_ids is not None and img_index not in process_img_ids:
            continue

        # print('Processing', fi, '/', len(all_files), ':', img_index)

        vector_data_path_ref = os.path.join(vector_data_dir, str(img_index) + '_ref.jsonl')
        with open(vector_data_path_ref, "r+") as f:
            for item in jsonlines.Reader(f):
                stroke_data_b_raw = item['stroke_params']  # curve list => stroke list (N', 4, 2)

        ref_img_path = os.path.join(database, 'raster_black', str(img_index) + '_ref.png')
        assert os.path.exists(ref_img_path)
        ref_img = Image.open(ref_img_path)
        image_size = ref_img.height

        # curve_grouping_depth(database, stroke_data_b_raw, img_index)
        group_labels, optical_flow_dir = curve_grouping_flow(database, stroke_data_b_raw, image_size, img_index,
                                                             optical_flow_method, use_distance_transform, should_merge_single_curve, verbose=True)
        max_group_num = np.max(group_labels) + 1

        stroke_data_b = [[] for _ in range(max_group_num)]  # component list => curve list => stroke list (N', 4, 2)
        for curve_i in range(len(group_labels)):
            group_label = group_labels[curve_i]
            stroke_data_b[group_label].append(stroke_data_b_raw[curve_i])

        save_base_grouping = os.path.join(database, 'optical_flow', optical_flow_dir, 'vector-vis-group')
        os.makedirs(save_base_grouping, exist_ok=True)
        vector_vis_path = os.path.join(save_base_grouping, str(img_index) + "_ref.png")
        draw_sketch_stroke_chain_cairo(stroke_data_b, vector_vis_path, is_bezier=True,
                                       side=image_size, line_diameter=line_thickness)

        # save layer-level parameters
        save_jsonl_path = os.path.join(save_base_parameter, str(img_index) + '_ref.jsonl')
        vector_data_ref = {}
        vector_data_ref['stroke_params'] = stroke_data_b
        with jsonlines.open(save_jsonl_path, mode='w') as json_writer:
            json_writer.write(vector_data_ref)

        save_base_grouping_label = os.path.join(database, 'optical_flow', optical_flow_dir, 'group_labels')
        os.makedirs(save_base_grouping_label, exist_ok=True)
        save_jsonl_path = os.path.join(save_base_grouping_label, str(img_index) + '_ref.jsonl')
        group_labels_data_ref = {}
        group_labels_data_ref['group_labels'] = group_labels
        with jsonlines.open(save_jsonl_path, mode='w') as json_writer:
            json_writer.write(group_labels_data_ref)


def curve_grouping_vis(database, line_thickness, optical_flow_method, use_distance_transform, process_img_ids=None):
    print('============== curve_grouping_vis ==============')
    vector_data_dir = os.path.join(database, 'vector-params')
    postfix = '_ref.jsonl'
    all_files = os.listdir(vector_data_dir)
    all_files = [item for item in all_files if postfix in item]
    all_files.sort()

    save_base_color_separate_chain = os.path.join(database, 'vector-vis-chain')
    optical_flow_dir = get_optical_flow_dir_name(optical_flow_method, use_distance_transform)
    save_base_grouping = os.path.join(database, 'optical_flow', optical_flow_dir, 'vector-vis-group')
    os.makedirs(save_base_color_separate_chain, exist_ok=True)
    os.makedirs(save_base_grouping, exist_ok=True)

    for fi, filename in enumerate(all_files):
        img_index = filename[:filename.find('_')]
        img_index = int(img_index)
        if process_img_ids is not None and img_index not in process_img_ids:
            continue

        # print('Processing', fi, '/', len(all_files), ':', img_index)

        vector_data_path_ref = os.path.join(vector_data_dir, str(img_index) + '_ref.jsonl')
        with open(vector_data_path_ref, "r+") as f:
            for item in jsonlines.Reader(f):
                stroke_data_b = item['stroke_params']  # component list => curve list => stroke list (N', 4, 2)

        ref_img_path = os.path.join(database, 'raster_black', str(img_index) + '_ref.png')
        assert os.path.exists(ref_img_path)
        ref_img = Image.open(ref_img_path)
        image_size = ref_img.height

        vector_vis_path = os.path.join(save_base_color_separate_chain, str(img_index) + "_ref.png")
        stroke_data_b_ = []
        for c_i in range(len(stroke_data_b)):
            curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
            for item in curve_b_list:
                stroke_data_b_.append([item])
        draw_sketch_stroke_chain_cairo(stroke_data_b_, vector_vis_path, is_bezier=True,
                                       side=image_size, line_diameter=line_thickness)

        vector_vis_path = os.path.join(save_base_grouping, str(img_index) + "_ref.png")
        draw_sketch_stroke_chain_cairo(stroke_data_b, vector_vis_path, is_bezier=True,
                                       side=image_size, line_diameter=line_thickness)


def curve_translation_with_optical_flow_main(database, line_thickness, optical_flow_method, use_distance_transform, process_img_ids=None):
    print('============== curve_translation_with_optical_flow_main ==============')
    curve_translation_with_optical_flow(database, line_thickness, optical_flow_method, use_distance_transform, process_img_ids)


def component_translation_with_optical_flow_main(database, line_thickness, optical_flow_method, use_distance_transform, process_img_ids=None,
                                                 data_extra=None):
    print('============== component_translation_with_optical_flow_main ==============')
    component_translation_with_optical_flow(database, True, line_thickness, optical_flow_method, use_distance_transform, process_img_ids,
                                            data_extra=data_extra, gen_time=gen_time)


def gen_component_images(database, line_thickness, process_img_ids=None, data_extra=None):
    print('============== gen_component_images ==============')

    vector_data_dir = os.path.join(database, 'vector-params')
    postfix = '_ref.jsonl'
    all_files = os.listdir(vector_data_dir)
    all_files = [item for item in all_files if postfix in item]
    all_files.sort()

    component_image_base = os.path.join(database, 'raster_black_component')
    os.makedirs(component_image_base, exist_ok=True)

    for fi, filename in enumerate(all_files):
        img_index = filename[:filename.find('_')]
        img_index = int(img_index)
        if process_img_ids is not None and img_index not in process_img_ids:
            continue

        # print('Processing', fi, '/', len(all_files), ':', img_index)

        ref_img_path = os.path.join(database, 'raster_black', str(img_index) + '_ref.png')
        assert os.path.exists(ref_img_path)
        ref_img = Image.open(ref_img_path)
        image_size = ref_img.height

        component_image_dir = os.path.join(component_image_base, str(img_index))
        if os.path.exists(component_image_dir):
            shutil.rmtree(component_image_dir)
        os.makedirs(component_image_dir, exist_ok=True)

        vector_data_path = os.path.join(vector_data_dir, filename)
        with open(vector_data_path, "r+") as f:
            for item in jsonlines.Reader(f):
                stroke_data_b = item['stroke_params']  # component list => curve list => stroke list (N', 4, 2)

        if data_extra is not None:
            data_base_extra = "outputs/stroke_correspondence_results"
            if gen_time > 0:
                data_base_extra += '-[Gen%d]' % gen_time
            vector_data_path_extra = os.path.join(data_base_extra, 'params', 'tar_pred-' + str(img_index) + '.jsonl')
            with open(vector_data_path_extra, "r+") as f:
                for item in jsonlines.Reader(f):
                    stroke_data_b_extra = item['stroke_params']  # component list => curve list => stroke list (N', 4, 2)

        for c_i in range(len(stroke_data_b)):
            curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
            if data_extra is not None:
                curve_b_list += stroke_data_b_extra[c_i]
            outpath = os.path.join(component_image_dir, 'component_' + str(c_i) + '-ref.png')
            draw_sketch_cairo([curve_b_list], outpath, is_bezier=True,
                              side=image_size, line_diameter=line_thickness)


def gen_stroke_images(database, line_thickness, process_img_ids=None):
    print('============== gen_stroke_images ==============')

    vector_data_dir = os.path.join(database, 'vector-params')
    postfix = '_ref.jsonl'
    all_files = os.listdir(vector_data_dir)
    all_files = [item for item in all_files if postfix in item]
    all_files.sort()

    stroke_image_base = os.path.join(database, 'raster_black_endpoint_stroke')
    stroke_image_base_prev = os.path.join(database, 'raster_black_endpoint_stroke_prev')
    stroke_image_base_next = os.path.join(database, 'raster_black_endpoint_stroke_next')
    os.makedirs(stroke_image_base, exist_ok=True)
    os.makedirs(stroke_image_base_prev, exist_ok=True)
    os.makedirs(stroke_image_base_next, exist_ok=True)

    for fi, filename in enumerate(all_files):
        img_index = filename[:filename.find('_')]
        img_index = int(img_index)
        if process_img_ids is not None and img_index not in process_img_ids:
            continue

        # print('Processing', fi, '/', len(all_files), ':', img_index)

        ref_img_path = os.path.join(database, 'raster_black', str(img_index) + '_ref.png')
        assert os.path.exists(ref_img_path)
        ref_img = Image.open(ref_img_path)
        image_size = ref_img.height

        stroke_image_dir = os.path.join(stroke_image_base, str(img_index))
        if os.path.exists(stroke_image_dir):
            shutil.rmtree(stroke_image_dir)
        os.makedirs(stroke_image_dir, exist_ok=True)
        stroke_image_dir_prev = os.path.join(stroke_image_base_prev, str(img_index))
        if os.path.exists(stroke_image_dir_prev):
            shutil.rmtree(stroke_image_dir_prev)
        os.makedirs(stroke_image_dir_prev, exist_ok=True)
        stroke_image_dir_next = os.path.join(stroke_image_base_next, str(img_index))
        if os.path.exists(stroke_image_dir_next):
            shutil.rmtree(stroke_image_dir_next)
        os.makedirs(stroke_image_dir_next, exist_ok=True)

        vector_data_path = os.path.join(vector_data_dir, filename)
        with open(vector_data_path, "r+") as f:
            for item in jsonlines.Reader(f):
                stroke_data_b = item['stroke_params']  # component list => curve list => stroke list (N', 4, 2)

                for c_i in range(len(stroke_data_b)):
                    curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
                    for curve_i in range(len(curve_b_list)):
                        curve_b_points = curve_b_list[curve_i]  # list (N') of (4, 2)
                        stroke_num = len(curve_b_points)

                        # starting stroke
                        outpath = os.path.join(stroke_image_dir, "endpoint_%s_%s_%s.png" % (c_i, curve_i, 0))
                        stroke_data_b_single = [[[curve_b_points[0]]]]
                        draw_sketch_cairo(stroke_data_b_single, outpath, is_bezier=True,
                                          side=image_size, line_diameter=line_thickness)

                        # intermediate strokes
                        for p_i in range(1, stroke_num):
                            outpath = os.path.join(stroke_image_dir, "endpoint_%s_%s_%s.png" % (c_i, curve_i, p_i))
                            stroke_data_b_single = [[[curve_b_points[p_i - 1], curve_b_points[p_i]]]]
                            draw_sketch_cairo(stroke_data_b_single, outpath, is_bezier=True,
                                              side=image_size, line_diameter=line_thickness)

                        # last stroke
                        outpath = os.path.join(stroke_image_dir, "endpoint_%s_%s_%s.png" % (c_i, curve_i, stroke_num))
                        stroke_data_b_single = [[[curve_b_points[stroke_num - 1]]]]
                        draw_sketch_cairo(stroke_data_b_single, outpath, is_bezier=True,
                                          side=image_size, line_diameter=line_thickness)

                        ## draw prev stroke
                        for p_i in range(stroke_num + 1):
                            outpath = os.path.join(stroke_image_dir_prev, "endpoint_%s_%s_%s.png" % (c_i, curve_i, p_i))
                            stroke_data_b_single = [[[curve_b_points[max(0, p_i - 1)]]]]
                            draw_sketch_cairo(stroke_data_b_single, outpath, is_bezier=True,
                                              side=image_size, line_diameter=line_thickness)

                        ## draw next stroke
                        for p_i in range(stroke_num + 1):
                            outpath = os.path.join(stroke_image_dir_next, "endpoint_%s_%s_%s.png" % (c_i, curve_i, p_i))
                            stroke_data_b_single = [[[curve_b_points[min(stroke_num - 1, p_i)]]]]
                            draw_sketch_cairo(stroke_data_b_single, outpath, is_bezier=True,
                                              side=image_size, line_diameter=line_thickness)


def gen_stroke_images_ctrlpoint(database, line_thickness, process_img_ids=None):
    print('============== gen_stroke_images_ctrlpoint ==============')

    vector_data_dir = os.path.join(database, 'vector-params')
    postfix = '_ref.jsonl'
    all_files = os.listdir(vector_data_dir)
    all_files = [item for item in all_files if postfix in item]
    all_files.sort()

    stroke_image_base = os.path.join(database, 'raster_black_ctrlpoint_stroke')
    stroke_image_base += '_ref'
    os.makedirs(stroke_image_base, exist_ok=True)

    for fi, filename in enumerate(all_files):
        img_index = filename[:filename.find('_')]
        img_index = int(img_index)
        if process_img_ids is not None and img_index not in process_img_ids:
            continue

        # print('Processing', fi, '/', len(all_files), ':', img_index)

        ref_img_path = os.path.join(database, 'raster_black', str(img_index) + '_ref.png')
        assert os.path.exists(ref_img_path)
        ref_img = Image.open(ref_img_path)
        image_size = ref_img.height

        stroke_image_dir = os.path.join(stroke_image_base, str(img_index))
        if os.path.exists(stroke_image_dir):
            shutil.rmtree(stroke_image_dir)
        os.makedirs(stroke_image_dir, exist_ok=True)

        vector_data_path = os.path.join(vector_data_dir, filename)
        with open(vector_data_path, "r+") as f:
            for item in jsonlines.Reader(f):
                stroke_data_b = item['stroke_params']  # component list => curve list => stroke list (N', 4, 2)

                for c_i in range(len(stroke_data_b)):
                    curve_b_list = stroke_data_b[c_i]  # list of (N', 4, 2)
                    for curve_i in range(len(curve_b_list)):
                        curve_b_points = curve_b_list[curve_i]  # list (N') of (4, 2)
                        stroke_num = len(curve_b_points)
                        for s_i in range(stroke_num):
                            outpath = os.path.join(stroke_image_dir, "stroke_%s_%s_%s.png" % (c_i, curve_i, s_i))
                            stroke_data_b_single = [[[curve_b_points[s_i]]]]
                            draw_sketch_cairo(stroke_data_b_single, outpath, is_bezier=True,
                                              side=image_size, line_diameter=line_thickness)


if __name__ == '__main__':
    database = test_data_base
    process_img_ids = [test_img_id]

    ## Parameters for curve_grouping
    optical_flow_method = example_info_map[str(test_img_id)]['optical_flow_method']
    use_distance_transform = example_info_map[str(test_img_id)]['use_distance_transform']
    should_merge_single_curve = True

    line_thickness = 3

    ###################### Main process ######################

    if '-Gen' not in database:
        load_svg(database, line_thickness, process_img_ids)

        vector_curve_splitting(database, line_thickness, process_img_ids)
        # gen_stroke_chain_images(database, line_thickness, process_img_ids)
        # curve_translation_with_optical_flow_main(database, line_thickness, optical_flow_method, use_distance_transform, process_img_ids)

        curve_grouping(database, line_thickness, optical_flow_method, use_distance_transform, should_merge_single_curve, process_img_ids)
    else:
        curve_grouping_vis(database, line_thickness, optical_flow_method, use_distance_transform, process_img_ids)

    ## For component transform task
    component_translation_with_optical_flow_main(database, line_thickness, optical_flow_method, use_distance_transform, process_img_ids)
    gen_component_images(database, line_thickness, process_img_ids)

    ## For endpoint matching task
    gen_stroke_images(database, line_thickness, process_img_ids)

    ## For control point matching task
    gen_stroke_images_ctrlpoint(database, line_thickness, process_img_ids)