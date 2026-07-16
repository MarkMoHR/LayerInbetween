import os

from configs.example_configs import gpu_id, test_data_base, test_img_id
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

import argparse
import cv2
import matplotlib
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def inference_single_img_main(database, image_id):
    split_types = ['_ref', '_tar']

    filenames = []
    for split_type in split_types:
        image_path = os.path.join(database, 'raster_black', str(image_id) + split_type + ".png")
        filenames.append(image_path)

    save_base = os.path.join(database, args.outdir)
    save_base_vis = os.path.join(save_base, 'vis')
    save_base_vis_o = os.path.join(save_base_vis, 'overlap')
    save_base_params = os.path.join(save_base, 'params')
    os.makedirs(save_base_vis, exist_ok=True)
    os.makedirs(save_base_vis_o, exist_ok=True)
    os.makedirs(save_base_params, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for k, filename in enumerate(filenames):
        print(f'Progress {k + 1}/{len(filenames)}: {filename}')

        raw_image = cv2.imread(filename)

        depth = depth_anything.infer_image(raw_image, args.input_size)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)  # (H, W), [0, 255]

        cv2.imwrite(os.path.join(save_base_params, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        depth_overlap = np.copy(depth).astype(np.int32)
        depth_overlap -= 255 - raw_image.astype(np.int32)
        depth_overlap = np.clip(depth_overlap, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(save_base_vis, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        cv2.imwrite(os.path.join(save_base_vis_o, os.path.splitext(os.path.basename(filename))[0] + '.png'),
                    depth_overlap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--image_id', default=test_img_id, type=int, help='image ID for evaluation')
    parser.add_argument('--data_base', default=test_data_base, type=str, help="data base path")

    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--grayscale', dest='grayscale', type=int, default=0, help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), f'checkpoints/depth_anything_v2_{args.encoder}.pth'),
        map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    inference_single_img_main(args.data_base, args.image_id)
