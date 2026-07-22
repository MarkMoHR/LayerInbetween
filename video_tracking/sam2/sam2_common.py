import numpy as np
import matplotlib.pyplot as plt
import math
import colorsys
import cv2


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def cal_mask_bbox(mask):
    """
    Args:
        mask: (H, W),
    """
    mask_pos = np.where(mask)
    if len(mask_pos[0]) == 0:
        return None, 0
    else:
        y_min, y_max = np.min(mask_pos[0]), np.max(mask_pos[0])
        x_min, x_max = np.min(mask_pos[1]), np.max(mask_pos[1])
        bbox = [x_min, y_min, x_max, y_max]
        area = (x_max - x_min + 1) * (y_max - y_min + 1)
        return bbox, area


def cal_bbox_inclusion_area(bbox_large, bbox_small):
    """
    Args:
        bbox_large: xyxy
        bbox_small: xyxy
    """
    assert bbox_large is not None
    assert bbox_small is not None
    x0_l, y0_l, x1_l, y1_l = bbox_large
    x0_s, y0_s, x1_s, y1_s = bbox_small

    min_right = min(x1_l, x1_s)
    max_left = max(x0_l, x0_s)
    width = max(0, min_right - max_left)

    min_up = min(y1_l, y1_s)
    max_down = max(y0_l, y0_s)
    height = max(0, min_up - max_down)

    inclusion_area = width * height
    return inclusion_area


def generate_colors2(N, divide=11, order=[0, 4, 8, 1, 6, 10, 2, 5, 9, 3, 7], replace_interval=[0.12, 0.3], shift=0):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    N_new = int(math.ceil(N / float(divide)) * divide)

    brightness = 1.0
    hsv = [(i / N_new, 1, brightness) for i in range(N_new)]
    colors_bright = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

    brightness = 0.9
    hsv = [(i / N_new, 1, brightness) for i in range(N_new)]
    colors_dark = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

    colors = np.array(colors_bright)
    colors_dark = np.array(colors_dark)
    replace_start = int(replace_interval[0] * N)
    replace_end = int(replace_interval[1] * N)
    colors[replace_start:replace_end] = colors_dark[replace_start:replace_end]

    colors = colors.reshape((divide, -1, 3))
    assert len(order) == divide
    sort_index = np.argsort(order)
    colors_new = [colors[i] for i in sort_index]
    colors = np.stack(colors_new, axis=0)
    colors = np.transpose(colors, (1, 0, 2)).reshape((-1, 3))
    if shift > 0:
        colors = np.concatenate([colors[shift:], colors[0:shift]], axis=0)

    return colors


def mask_dilate(mask, dilate_size):
    """
    Args:
        mask: (H, W), (True, False)
    """
    mask_img = np.zeros_like(mask, dtype=np.uint8)
    mask_img[mask] = 255

    if dilate_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
        mask_img_d = cv2.dilate(mask_img, kernel)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (-dilate_size, -dilate_size))
        mask_img_d = cv2.erode(mask_img, kernel)

    return mask_img_d > 0
