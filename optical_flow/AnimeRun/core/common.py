import numpy as np

import torch
import distmap
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


def warp_image_direct(source_img_, flow, target_img=None, black_threshold=200):
    # source_img: (H, W), [0-stroke, 255-BG]
    # flow: (H, W, 2)
    # target_img: (H, W), [0-stroke, 255-BG]
    source_img = 255 - source_img_  # (H, W), [0-BG, 255-stroke]
    warped_img = np.zeros_like(source_img).astype(np.float32)
    for i in range(source_img.shape[0]):
        for j in range(source_img.shape[1]):
            ori_pixel_value = source_img[i, j]
            if ori_pixel_value != 0:
                move_vector = flow[i, j]  # (2)
                i_new = int(min(max(0, i + move_vector[1]), source_img.shape[0] - 1))
                j_new = int(min(max(0, j + move_vector[0]), source_img.shape[1] - 1))
                warped_img[i_new, j_new] += ori_pixel_value
    warped_img = 255 - np.clip(warped_img, 0, 255)  # (H, W), [0-stroke, 255-BG]
    warped_img = np.tile(np.expand_dims(warped_img, axis=-1), (1, 1, 3))
    if target_img is not None:
        target_trans_mask = (warped_img < black_threshold).any(-1)
        warped_img_bg = np.expand_dims(target_img, axis=-1).astype(np.float32)
        warped_img_bg = np.concatenate(
            [np.ones_like(warped_img_bg) * 255,
             warped_img_bg,
             np.ones_like(warped_img_bg) * 255], axis=-1)
        warped_img_bg = 255 - (255 - warped_img_bg) * 0.7
        warped_img_bg[target_trans_mask] = warped_img[target_trans_mask]
    else:
        warped_img_bg = warped_img

    return warped_img_bg


def warp_image_with_corr_mat(source_img_, correspondence_mat, target_img, black_threshold):
    # source_img: (H, W), [0-stroke, 255-BG]
    # flow: (H, W, 2)
    # target_img: (H, W), [0-stroke, 255-BG]

    source_mask = source_img_ < black_threshold
    source_img = 255 - source_img_  # (H, W), [0-BG, 255-stroke]

    # warp image
    warped_img = np.zeros_like(source_img).astype(np.float32)
    for i in range(source_img.shape[0]):
        for j in range(source_img.shape[1]):
            ori_pixel_value = source_img[i, j]
            if source_mask[i, j]:
                i_new, j_new = correspondence_mat[i, j]
                assert i_new != -1 and j_new != -1
                warped_img[i_new, j_new] += ori_pixel_value

    warped_img = 255 - np.clip(warped_img, 0, 255)  # (H, W), [0-stroke, 255-BG]
    warped_img = np.tile(np.expand_dims(warped_img, axis=-1), (1, 1, 3))
    if target_img is not None:
        target_trans_mask = (warped_img < black_threshold).any(-1)
        warped_img_bg = np.expand_dims(target_img, axis=-1).astype(np.float32)
        warped_img_bg = np.concatenate(
            [np.ones_like(warped_img_bg) * 255,
             warped_img_bg,
             np.ones_like(warped_img_bg) * 255], axis=-1)
        warped_img_bg = 255 - (255 - warped_img_bg) * 0.7
        warped_img_bg[target_trans_mask] = warped_img[target_trans_mask]
    else:
        warped_img_bg = warped_img

    return warped_img_bg


def generate_correspondence_matrix(source_img_, flow, target_img_, black_threshold):
    # source_img: (H, W), [0-stroke, 255-BG]
    # flow: (H, W, 2)
    # target_img_: (H, W), [0-stroke, 255-BG]
    source_mask = source_img_ < black_threshold
    source_img = 255 - source_img_  # (H, W), [0-BG, 255-stroke]

    target_img = target_img_.astype(np.float32)
    target_img[target_img < black_threshold] = 0
    target_img = target_img / 255.0  # (H, W), [0.0-stroke, 1.0-BG])
    _, dt_inds = distance_transform_edt(target_img, return_indices=True)  # dt_inds: (2, H, W), (y, x)

    correspondence_mat = np.zeros_like(flow).astype(np.int32)  # (H, W, 2)
    correspondence_mat.fill(-1)
    correspondence_mat_cling = np.zeros_like(flow).astype(np.int32)  # (H, W, 2)
    correspondence_mat_cling.fill(-1)

    for i in range(source_img.shape[0]):
        for j in range(source_img.shape[1]):
            if source_mask[i, j]:
                move_vector = flow[i, j]  # (2), (x, y)
                i_new = int(min(max(0, i + move_vector[1]), source_img.shape[0] - 1))
                j_new = int(min(max(0, j + move_vector[0]), source_img.shape[1] - 1))
                i_new_cling, j_new_cling = dt_inds[:, i_new, j_new]  # (y, x)
                correspondence_mat[i, j] = np.array([i_new, j_new])
                correspondence_mat_cling[i, j] = np.array([i_new_cling, j_new_cling])
    return correspondence_mat, correspondence_mat_cling


def distance_transform(input_imgs, factor=10.0):
    # input_imgs: (N, H, W), [0-stroke, 1-BG]
    input_imgs_dt = torch.floor(input_imgs)  # (N, H, W), [0-stroke, 1-BG], avoid bad binarization on thin strokes
    input_imgs_dt = distmap.euclidean_distance_transform(input_imgs_dt, ndim=2)  # (N, H, W)

    input_imgs_dt_norm = 1 - torch.exp(-input_imgs_dt / factor)  # (N, H, W)
    max_value = torch.max(input_imgs_dt_norm.reshape(input_imgs_dt_norm.size(0), -1), dim=-1)[0]
    max_value = max_value.unsqueeze(dim=-1).unsqueeze(dim=-1)
    min_value = torch.min(input_imgs_dt_norm.reshape(input_imgs_dt_norm.size(0), -1), dim=-1)[0]
    min_value = min_value.unsqueeze(dim=-1).unsqueeze(dim=-1)
    input_imgs_dt_norm = (input_imgs_dt_norm - min_value) / (max_value - min_value)  # (N, H, W), [0, 1]
    input_imgs_dt_norm[torch.isnan(input_imgs_dt_norm)] = 1.0  # avoid nan in empty canvas. (N, H, W), [0, 1]

    return input_imgs_dt_norm


def visualize_correspondence_map(img_1, img_2, corr_mat, mask_threshold):
    # source_img: (H, W), [0-stroke, 255-BG]
    # target_img: (H, W), [0-stroke, 255-BG]
    # corr_mat: (H, W, 2)
    cmap = plt.cm.get_cmap('jet')
    img1_cmap = np.ones((img_1.shape[0], img_1.shape[1], 3)).astype(np.float32)
    img2_cmap = np.ones((img_2.shape[0], img_2.shape[1], 3)).astype(np.float32)

    mask_1 = img_1 < mask_threshold
    color_sum = np.sum(mask_1)
    color_i = 0
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            if mask_1[i, j]:
                # p_color = cmap((i * img_1.shape[0] + j) / (img_1.shape[0] * img_1.shape[1]))[0:3]
                p_color = cmap(color_i / color_sum)[0:3]
                color_i += 1

                i_new, j_new = corr_mat[i, j]
                assert i_new != -1 and j_new != -1
                img1_cmap[i, j] = np.asarray(p_color)
                img2_cmap[i_new, j_new] = np.asarray(p_color)

    img1_cmap = np.clip(np.round(img1_cmap * 255.0), 0.0, 255.0).astype(np.uint8)
    img2_cmap = np.clip(np.round(img2_cmap * 255.0), 0.0, 255.0).astype(np.uint8)
    return img1_cmap, img2_cmap

