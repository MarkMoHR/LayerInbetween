import os

from configs.example_configs import gpu_id, test_data_base, test_img_id, example_info_map, do_inv
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

from PIL import Image
import argparse
import numpy as np
import torch
from core.utils import flow_viz

from core.raft_gma import RAFTGMA
from core.raft import RAFT
from core.utils.utils import InputPadder
from core.common import warp_image_with_corr_mat, generate_correspondence_matrix, distance_transform
import cv2


@torch.no_grad()
def validate_anime(data_base, model, weight_name, iters=32, black_threshold=200):
    """ Peform validation using the Sintel (train) split """
    model.eval()

    save_dir = os.path.join(data_base, 'optical_flow')

    weight_name_sub = weight_name[weight_name.rfind('/') + 1: weight_name.rfind('.')]
    if args.use_distance_transform:
        weight_name_sub += '-[DT-' + str(args.distance_transform_factor) + ']'
    
    save_base_flow = os.path.join(save_dir, weight_name_sub, 'flow')
    save_base_warped = os.path.join(save_dir, weight_name_sub, 'warped')
    os.makedirs(save_base_flow, exist_ok=True)
    os.makedirs(save_base_warped, exist_ok=True)

    if not args.is_binary:
        image1_path = os.path.join(data_base, 'raster_black', "%s_ref.png" % args.image_id)
        image2_path = os.path.join(data_base, 'raster_black', "%s_tar.png" % args.image_id)
    else:
        image1_path = os.path.join(data_base, 'raster_black/binarized', "%s_ref.png" % args.image_id)
        image2_path = os.path.join(data_base, 'raster_black/binarized', "%s_tar.png" % args.image_id)

    image1 = Image.open(image1_path).convert('RGB')
    image1 = np.array(image1, dtype=np.float32)[:, :, 0]  # (H, W), [0-stroke, 255-BG]
    image1_t = torch.tensor(image1).float().cuda()
    image1_t = image1_t.unsqueeze(dim=0)  # (1, H, W), [0-stroke, 255-BG]
    if args.use_distance_transform:
        image1_t = distance_transform(image1_t / 255.0,
                                        factor=float(args.distance_transform_factor)) * 255.0  # (1, H, W), [0, 255]
    image1_t = image1_t.unsqueeze(dim=1).repeat(1, 3, 1, 1)  # (1, 3, H, W), [0, 255]

    image2 = Image.open(image2_path).convert('RGB')
    image2 = np.array(image2, dtype=np.float32)[:, :, 0]  # (H, W), [0-stroke, 255-BG]
    image2_t = torch.tensor(image2).float().cuda()
    image2_t = image2_t.unsqueeze(dim=0)  # (1, H, W), [0-stroke, 255-BG]
    if args.use_distance_transform:
        image2_t = distance_transform(image2_t / 255.0,
                                        factor=float(args.distance_transform_factor)) * 255.0  # (1, H, W), [0, 255]
    image2_t = image2_t.unsqueeze(dim=1).repeat(1, 3, 1, 1)  # (1, 3, H, W), [0, 255]

    padder = InputPadder(image1_t.shape, padding_factor=32)
    image1_t, image2_t = padder.pad(image1_t, image2_t)

    flow_low, flow_pr = model(image1_t, image2_t, iters=iters, test_mode=True)
    flow = padder.unpad(flow_pr[0]).cpu()  # (2, H, W)
    flow_np = flow.permute(1, 2, 0).data.numpy()

    flo_color = flow_viz.flow_to_image(flow_np, convert_to_bgr=True)
    flow_save_path = os.path.join(save_base_flow, 'flow-' + str(args.image_id) + '.png')
    cv2.imwrite(flow_save_path, flo_color)

    flow_mat_path = os.path.join(save_base_flow, 'flow-' + str(args.image_id) + '.npz')
    np.savez(flow_mat_path, flow_mat=flow_np)

    correspondence_mat, correspondence_mat_cling = generate_correspondence_matrix(image1, flow_np, image2, black_threshold=black_threshold)

    warped_image = warp_image_with_corr_mat(image1, correspondence_mat, image2, black_threshold=black_threshold)
    warped_image = warped_image.astype(np.uint8)
    warped_image_png = Image.fromarray(warped_image, 'RGB')
    warped_image_save_path = os.path.join(save_base_warped, 'warped-' + str(args.image_id) + '.png')
    warped_image_png.save(warped_image_save_path, 'PNG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_base', default=test_data_base, type=str, help="data base path")
    parser.add_argument('--image_id', default=test_img_id, type=int, help='image ID for evaluation')
    parser.add_argument('--model_name', default=example_info_map[str(test_img_id)]['optical_flow_method'], type=str, 
                        choices=['gma', 'raft'], help="restore checkpoint")
    
    parser.add_argument('--num_heads', default=1, type=int,
                    help='number of heads in attention and aggregation')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--position_only', default=False, action='store_true',
                    help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                    help='use position and content-wise attention')
    parser.add_argument('--small', action='store_true', help='use small model')

    parser.add_argument('--distance_transform_factor', default=10, type=float, help='factor for distance transform')
    parser.add_argument('--is_binary', default=False, type=bool, help='use binary image')
    args = parser.parse_args()

    args.use_distance_transform = example_info_map[str(test_img_id)]['use_distance_transform']

    if args.model_name == "gma":
        args.model = os.path.join(os.path.dirname(__file__), 'checkpoints', '20000_gma-animerun-v2-ft.pth')
        model = torch.nn.DataParallel(RAFTGMA(args))
    elif args.model_name == "raft":
        args.model = os.path.join(os.path.dirname(__file__), 'checkpoints', '20000_raft-animerun-v2-ft_again.pth')
        model = torch.nn.DataParallel(RAFT(args))
    else:
        raise ValueError("Unknown model name: {}".format(args.model_name))

    data_base_input = args.data_base if not do_inv else os.path.join(args.data_base, '[0inv]')

    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.eval()

    print('Testing {}'.format(args.model))
    with torch.no_grad():
        validate_anime(data_base_input, model.module, args.model)
