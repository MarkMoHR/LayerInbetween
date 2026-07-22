import argparse
import os
import numpy as np
from PIL import Image
import copy

from configs.example_configs import test_img_id, gen_time


def vis_gif(img_id, inbetween_result_base):
    save_base = os.path.join(inbetween_result_base, str(img_id), 'gifs')
    os.makedirs(save_base, exist_ok=True)

    img_dir = os.path.join(inbetween_result_base, str(img_id), 'linearts_occ_resolve')
    all_files = os.listdir(img_dir)
    all_files.sort()
    all_files = [item for item in all_files if '.png' in item]

    frame_num = len(all_files)
    all_files = [str(frame_i) + '.png' for frame_i in range(frame_num)]
    all_files_single_pass = copy.deepcopy(all_files)
    all_files += all_files[::-1][1:]

    # generate cycling gif
    gif_frames = []
    for img_name in all_files:
        img_i = Image.open(os.path.join(img_dir, img_name))
        gif_frames.append(img_i)

    save_path = os.path.join(save_base, str(img_id) + '.gif')
    first_frame = gif_frames[0]
    first_frame.save(save_path, save_all=True, append_images=gif_frames, loop=0, duration=0.01)

    # generate single-pass gif
    gif_frames = []
    for img_name in all_files_single_pass:
        img_i = Image.open(os.path.join(img_dir, img_name))
        gif_frames.append(img_i)

    save_path = os.path.join(save_base, str(img_id) + '_single_pass.gif')
    first_frame = gif_frames[0]
    first_frame.save(save_path, save_all=True, append_images=gif_frames, loop=0, duration=0.01)


def vis_gif_multikeyframe(inbetween_result_base):
    image_frame_ids = ["41-0", "41-1", "41-2"]
    image_id = image_frame_ids[0].split("-")[0]

    save_base = os.path.join(inbetween_result_base, str(image_id), 'gifs')
    os.makedirs(save_base, exist_ok=True)

    all_files_global = []
    for ii, img_frame_id in enumerate(image_frame_ids):
        image_id_ = img_frame_id.split("-")[0]
        assert image_id_ == image_id

        img_dir = os.path.join(inbetween_result_base, str(img_frame_id), 'linearts_occ_resolve')
        all_files = os.listdir(img_dir)
        all_files = [item for item in all_files if '.png' in item]

        frame_idx_start = 0 if len(all_files_global) == 0 else 1
        frame_idx_end = len(all_files)
        all_files = [os.path.join(img_dir, str(frame_i) + '.png') for frame_i in range(frame_idx_start, frame_idx_end)]
        all_files_global += all_files

    all_files_global_single_pass = copy.deepcopy(all_files_global)
    all_files_global += all_files_global[::-1][1:]

    # generate cycling gif
    gif_frames = []
    for img_path in all_files_global:
        img_i = Image.open(img_path)
        gif_frames.append(img_i)

    save_path = os.path.join(save_base, str(image_id) + '.gif')
    first_frame = gif_frames[0]
    first_frame.save(save_path, save_all=True, append_images=gif_frames, loop=0, duration=0.01)

    # generate single-pass gif
    gif_frames = []
    for img_path in all_files_global_single_pass:
        img_i = Image.open(img_path)
        gif_frames.append(img_i)

    save_path = os.path.join(save_base, str(image_id) + '_single_pass.gif')
    first_frame = gif_frames[0]
    first_frame.save(save_path, save_all=True, append_images=gif_frames, loop=0, duration=0.01)


def gen_intensity_list(max_inten, min_inten, num):
    interval = (max_inten - min_inten) // (num - 1)
    intensity_list = [min_inten + i * interval for i in range(num)]
    intensity_list = intensity_list[::-1]
    return intensity_list


def make_inbetweening_img(data_base, image_sequence, save_base, shift=0):
    max_intensity = 200
    min_intensity = 0
    black_threshold = 128

    img_num = len(image_sequence)

    intensity_list = gen_intensity_list(max_intensity, min_intensity, img_num)
    # print('intensity_list', intensity_list)

    img = Image.open(os.path.join(data_base, image_sequence[0])).convert('RGB')
    height, width = img.height, img.width

    new_width = width + shift * (len(image_sequence) - 1)

    img_inbetween = np.ones(shape=(height, new_width)) * 255
    # print('img_inbetween', img_inbetween.shape)

    for i, img_name in enumerate(image_sequence):
        img_path = os.path.join(data_base, img_name)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)[:, :, 0]  # (H, W)
        img = np.pad(img, ((0, 0), (i * shift, (len(image_sequence) - i - 1) * shift)), 'constant', constant_values=255)

        intensity = intensity_list[i]

        img_inbetween[img <= black_threshold] = intensity

    img_inbetween = Image.fromarray(img_inbetween.astype(np.uint8), 'L')
    save_path = os.path.join(save_base, 'inbetweening.png')
    img_inbetween.save(save_path)


def vis_inbetweenn_img(img_id, inbetween_result_base):
    frame_base = os.path.join(inbetween_result_base, str(img_id), 'linearts_occ_resolve')

    # You can define the file for making the inbetweenning image
    image_sequence = ['0.png', '5.png', '11.png', '16.png']

    save_base = os.path.join(inbetween_result_base, str(img_id), 'inbetween_img')
    os.makedirs(save_base, exist_ok=True)

    make_inbetweening_img(frame_base, image_sequence, save_base, shift=0)


def split_gif():
    gif_path = 'dynamic.gif'
    img = Image.open(gif_path)
    for i in range(img.n_frames):
        img.seek(i)
        new = Image.new("RGBA", img.size)
        new.paste(img)
        new.save("%d.png" % (i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_id', default=test_img_id, type=int, help='image ID for evaluation')
    args = parser.parse_args()

    inbetween_result_base = "outputs/inbetweening_results"
    if gen_time > 0:
        inbetween_result_base += '-Gen%d' % gen_time

    vis_gif(args.image_id, inbetween_result_base)
    # vis_gif_multikeyframe(inbetween_result_base)

    vis_inbetweenn_img(args.image_id, inbetween_result_base)
