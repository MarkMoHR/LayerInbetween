import os
import numpy as np
import shutil
import jsonlines
from PIL import Image

from utils.svg_write import write_svg_chain
from configs.example_configs import test_data_base, test_img_id, gen_time


def write_svg(database, process_img_id):
    svg_save_base = os.path.join(database, '[0inv]', 'svg')
    os.makedirs(svg_save_base, exist_ok=True)

    img_file_path = os.path.join(database, 'raster_black', str(process_img_id) + '_ref.png')
    img = Image.open(img_file_path).convert('RGB')
    img_width, img_height = img.width, img.height

    data_base_extra = "outputs/stroke_correspondence_results"
    if gen_time > 0:
        data_base_extra += '-[Gen%d]' % gen_time

    params_path = os.path.join(data_base_extra, 'params', 'tar_pred-' + str(process_img_id) + '.jsonl')
    with open(params_path, "r+") as f:
        for item in jsonlines.Reader(f):
            stroke_data_b = item['stroke_params']
            # stroke_data_b: a list of layers
            #   => layer: a list of stroke chains
            #     => stroke chain: a list of strokes
            #       => stroke: (4, 2)

    stroke_chains_flatten = []
    for c_i, component in enumerate(stroke_data_b):
        for curve in component:  # (N, 4, 2) for bezier
            curve_flatten = [curve[0][0]]
            for i in range(len(curve)):
                curve_flatten += curve[i][1:]

            assert (len(curve_flatten) - 1) // 3 == len(curve)
            curve_flatten = np.stack(curve_flatten, axis=0)  # (N_point, 2)
            stroke_chains_flatten.append(curve_flatten)

    svg_save_path = os.path.join(svg_save_base, str(process_img_id) + '_tar0.svg')
    write_svg_chain(stroke_chains_flatten, img_height, img_width, color='#ff0044', svg_save_path=svg_save_path)


def switch_ref_tar_imgs(database, process_img_id):
    ori_img_base = os.path.join(database, 'raster_black')
    inv_img_base = os.path.join(database, '[0inv]', 'raster_black')
    os.makedirs(inv_img_base, exist_ok=True)

    ori_ref_path = os.path.join(ori_img_base, str(process_img_id) + '_ref.png')
    ori_tar_path = os.path.join(ori_img_base, str(process_img_id) + '_tar.png')

    inv_ref_path = os.path.join(inv_img_base, str(process_img_id) + '_ref.png')
    inv_tar_path = os.path.join(inv_img_base, str(process_img_id) + '_tar.png')

    shutil.copy(ori_ref_path, inv_tar_path)
    shutil.copy(ori_tar_path, inv_ref_path)


if __name__ == '__main__':
    database = test_data_base
    process_img_id = test_img_id

    write_svg(database, process_img_id)
    switch_ref_tar_imgs(database, process_img_id)
