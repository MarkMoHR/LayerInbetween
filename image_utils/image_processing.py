import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import cairocffi as cairo

from dataset_utils.common import generate_colors2


def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = np.array(image, dtype=np.float32)  # (H, W, 3), [0.0-strokes, 255.0-BG]
    image = image[:, :, 0] / 255.0  # (H, W), [0.0-strokes, 1.0-BG]
    return image


def save_image(img_np, save_root, save_name):
    img_np = (np.array(img_np) * 255.0).astype(np.uint8)
    save_path = os.path.join(save_root, save_name)
    img_png = Image.fromarray(img_np, 'L')
    img_png.save(save_path, 'PNG')
    return img_np


def save_image_overlap(img_np, img_bg_np, save_root=None, save_name=None):
    img_fg = np.array(img_np, dtype=np.float32) * 255.0
    img_fg = np.tile(np.expand_dims(img_fg, axis=-1), (1, 1, 3))
    fg_mask = (img_fg < 200).any(-1)
    img_bg = np.expand_dims(img_bg_np, axis=-1).astype(np.float32)
    img_bg = np.concatenate(
        [np.ones_like(img_bg) * 255,
         img_bg,
         np.ones_like(img_bg) * 255], axis=-1)
    img_bg = 255 - (255 - img_bg) * 0.7
    img_bg[fg_mask] = img_fg[fg_mask]
    img_bg = img_bg.astype(np.uint8)
    if save_root is not None:
        save_path = os.path.join(save_root, save_name)
        img_bg_png = Image.fromarray(img_bg, 'RGB')
        img_bg_png.save(save_path, 'PNG')


def draw_heatmap(coord, raster_size, save_root, save_name, background=None, sigma=2):
    '''
    coord: (2), [-1.0, 1.0]
    background: [0, 255], np.uint8
    '''
    coord_abs = (coord + 1.0) / 2.0  # (2), [0.0, 1.0]
    c_x = int(round(coord_abs[0] * raster_size))
    c_y = int(round(coord_abs[1] * raster_size))

    # c_x = min(max(0, c_x), raster_size - 1)
    # c_y = min(max(0, c_y), raster_size - 1)

    heatmap = np.zeros((raster_size, raster_size))
    if c_x < 0 or c_x >= raster_size or c_y < 0 or c_y >= raster_size:
        heatmap_smooth = 1.0 - heatmap  # (raster_size, raster_size), [0.0-stroke, 1.0-BG]
    else:
        heatmap[c_y, c_x] = 1
        heatmap_smooth = gaussian_filter(heatmap, sigma=sigma, mode="constant")  # [constant, nearest, | reflect, mirror, wrap]
        heatmap_smooth /= np.max(heatmap_smooth)  # [0.0-BG, 1.0-FG]
        heatmap_smooth = 1.0 - heatmap_smooth  # (raster_size, raster_size), [0.0-stroke, 1.0-BG]

    if background is not None:
        save_image_overlap(heatmap_smooth, background, save_root, save_name)
    else:
        heatmap_smooth_png = (heatmap_smooth * 255.0).astype(np.uint8)
        save_path = os.path.join(save_root, save_name)
        heatmap_smooth_png = Image.fromarray(heatmap_smooth_png, 'RGB')
        heatmap_smooth_png.save(save_path, 'PNG')


def draw_dot(img, dot_pos, color, radius=4):
    # img_dot = np.expand_dims(img, axis=-1)
    # img_dot = np.tile(img_dot, (1, 1, 3))
    img_dot = np.copy(img)
    img_size = img.shape[0]
    # dot_pos = (dot_pos_norm + 1.0) / 2.0 * img_size

    dot_left = int(max(dot_pos[0] - radius, 0))
    dot_right = int(min(dot_pos[0] + radius, img_size - 1))
    dot_up = int(max(dot_pos[1] - radius, 0))
    dot_down = int(min(dot_pos[1] + radius, img_size - 1))
    img_dot[dot_up:dot_down, dot_left:dot_right] = color
    return img_dot


def draw_dot_full(img, dots, save_root, save_name, radius=4, drawn_states=None):
    '''
    img: (H, W), [0-stroke, 255-BG]
    dots: (N, 2), in full size
    '''
    n_dot = len(dots)
    colors = generate_colors2(n_dot)  # list of (3), in [0., 1.]
    img_rst = np.expand_dims(img, axis=-1)
    img_rst = np.tile(img_rst, (1, 1, 3))
    for i in range(n_dot):
        if drawn_states is not None and not drawn_states[i]:
            continue
        img_rst = draw_dot(img_rst, dots[i], np.array(colors[i], dtype=np.float32) * 255,
                           radius=radius)
    img_rst_png = Image.fromarray(img_rst.astype(np.uint8), 'RGB')
    img_rst_png.save(os.path.join(save_root, save_name), 'PNG')
    return img_rst


def draw_stroke(stroke_points, raster_size, save_root=None, save_name=None, background=None,
                line_thickness=3, bg_color=(1, 1, 1), fg_color=(0, 0, 0)):
    '''
    stroke_points: (4, 2), [-1.0, 1.0]
    background: [0, 255], np.uint8
    '''
    stroke_points_abs = (stroke_points + 1.0) / 2.0 * raster_size  # (4, 2), in raster_size

    # cairo settings
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, raster_size, raster_size)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_thickness)

    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    # draw strokes, this is the most cpu-intensive part
    ctx.set_source_rgb(*fg_color)
    p0, p1, p2, p3 = stroke_points_abs.tolist()
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    ctx.move_to(x0, y0)
    ctx.curve_to(x1, y1, x2, y2, x3, y3)
    ctx.stroke()

    surface_data = surface.get_data()
    stroke_image = np.copy(np.asarray(surface_data))[::4].reshape(raster_size, raster_size)  # (raster_size, raster_size), [0-stroke, 255-BG]
    stroke_image = np.array(stroke_image, dtype=np.float32) / 255.0  # (raster_size, raster_size), [0.0-stroke, 1.0-BG]

    if background is not None:
        save_image_overlap(stroke_image, background, save_root, save_name)
    else:
        if save_root is not None:
            stroke_image_png = (stroke_image * 255.0).astype(np.uint8)
            save_path = os.path.join(save_root, save_name)
            stroke_image_png = Image.fromarray(stroke_image_png, 'RGB')
            stroke_image_png.save(save_path, 'PNG')
    return stroke_image


def draw_segment(stroke_endpoints, raster_size, save_root=None, save_name=None, background=None,
                 line_thickness=4, bg_color=(1, 1, 1), fg_color=(0, 0, 0)):
    '''
    stroke_endpoints: (4), [-1.0, 1.0]
    background: [0, 255], np.uint8
    '''
    stroke_endpoints_abs = (stroke_endpoints + 1.0) / 2.0 * raster_size  # (4), in raster_size

    # cairo settings
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, raster_size, raster_size)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_thickness)

    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    # draw strokes, this is the most cpu-intensive part
    ctx.set_source_rgb(*fg_color)
    x0, y0 = stroke_endpoints_abs[0], stroke_endpoints_abs[1]
    x1, y1 = stroke_endpoints_abs[2], stroke_endpoints_abs[3]
    ctx.move_to(x0, y0)
    ctx.line_to(x1, y1)
    ctx.stroke()

    surface_data = surface.get_data()
    stroke_image = np.copy(np.asarray(surface_data))[::4].reshape(raster_size, raster_size)  # (raster_size, raster_size), [0-stroke, 255-BG]
    stroke_image = np.array(stroke_image, dtype=np.float32) / 255.0  # (raster_size, raster_size), [0.0-stroke, 1.0-BG]

    if save_root is not None:
        if background is not None:
            save_image_overlap(stroke_image, background, save_root, save_name)
        else:
            stroke_image_png = (stroke_image * 255.0).astype(np.uint8)
            save_path = os.path.join(save_root, save_name)
            stroke_image_png = Image.fromarray(stroke_image_png, 'RGB')
            stroke_image_png.save(save_path, 'PNG')
    return stroke_image


def draw_sketch_stroke(stroke_data, outpath, bg_sketch=None, side=-1, line_diameter=3,
                       bg_color=(1, 1, 1), drawn_states=None):
    '''
    stroke_data: (N, 8)
    bg_sketch: (H, W), [0-stroke, 255-BG]
    '''
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    max_seq_number = len(stroke_data)
    colors = generate_colors2(max_seq_number)  # list of (3), in [0., 1.]

    for si in range(max_seq_number):
        if drawn_states is not None and not drawn_states[si]:
            continue
        stroke_i = stroke_data[si]
        p0, p1, p2, p3 = stroke_i[0:2], stroke_i[2:4], stroke_i[4:6], stroke_i[6:8]
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        ctx.set_source_rgb(colors[si][0], colors[si][1], colors[si][2])
        ctx.move_to(x0, y0)
        ctx.curve_to(x1, y1, x2, y2, x3, y3)
        ctx.stroke()
    surface_data = surface.get_data()
    raster_image_rgb = np.copy(np.asarray(surface_data)).reshape(side, side, 4)[:, :, :3]

    if bg_sketch is not None:
        # add baclground
        raster_image_rgb_bg = np.copy(bg_sketch).astype(np.float32)  # (H, W), [0-stroke, 255-BG]
        raster_image_rgb_bg = np.tile(np.expand_dims(raster_image_rgb_bg, axis=-1), (1, 1, 3))
        raster_image_rgb_bg = 255 - (255 - raster_image_rgb_bg) * 0.2
        rgb_mask = (raster_image_rgb != 255).any(-1)
        raster_image_rgb_bg[rgb_mask] = raster_image_rgb[rgb_mask]
        raster_image_rgb_bg = raster_image_rgb_bg.astype(np.uint8)
        raster_image_rgb_bg = Image.fromarray(raster_image_rgb_bg, 'RGB')
        raster_image_rgb_bg.save(outpath, 'PNG')
    else:
        raster_image_rgb = Image.fromarray(raster_image_rgb.astype(np.uint8), 'RGB')
        raster_image_rgb.save(outpath, 'PNG')


def cal_stroke_pixel_iou(stroke_points_gt, stroke_points_pred, image_size, occluded_masks=None, binary_threshold=1.0):
    '''
    stroke_points_gt / stroke_points_pred: (N, 8), in full size
    occluded_masks: (N, H, W, 1), [0-occluded, 1-visible]
    '''
    stroke_pixel_iou = []
    for s_i in range(len(stroke_points_gt)):
        stroke_gt = np.array(stroke_points_gt[s_i], dtype=np.float32).reshape((4, 2))  # (4, 2), in full size
        stroke_pred = np.array(stroke_points_pred[s_i], dtype=np.float32).reshape((4, 2))  # (4, 2), in full size
        stroke_gt = stroke_gt / float(image_size) * 2.0 - 1.0  # (4, 2), [-1.0, 1.0]
        stroke_pred = stroke_pred / float(image_size) * 2.0 - 1.0  # (4, 2), [-1.0, 1.0]

        if occluded_masks is not None:
            occluded_mask = occluded_masks[s_i, :, :, 0]  # (H, W), [0-occluded, 1-visible]
        else:
            occluded_mask = np.ones(shape=(image_size, image_size), dtype=np.float32)

        stroke_gt_img = draw_stroke(stroke_gt, image_size)  # (H, W), [0.0-stroke, 1.0-BG]
        stroke_gt_img = 1.0 - (1.0 - stroke_gt_img) * occluded_mask  # (H, W), [0.0-stroke, 1.0-BG]
        stroke_gt_mask = stroke_gt_img < binary_threshold  # (H, W)

        stroke_pred_img = draw_stroke(stroke_pred, image_size)  # (H, W), [0.0-stroke, 1.0-BG]
        stroke_pred_img = 1.0 - (1.0 - stroke_pred_img) * occluded_mask  # (H, W), [0.0-stroke, 1.0-BG]
        stroke_pred_mask = stroke_pred_img < binary_threshold  # (H, W)

        stroke_intersection = np.logical_and(stroke_gt_mask, stroke_pred_mask)
        stroke_union = np.logical_or(stroke_gt_mask, stroke_pred_mask)

        stroke_intersection_pixel_num = np.sum(stroke_intersection)
        stroke_union_pixel_num = np.sum(stroke_union)
        if stroke_union_pixel_num != 0.0:
            stroke_iou = stroke_intersection_pixel_num / stroke_union_pixel_num
        else:
            stroke_iou = 1.0
        stroke_pixel_iou.append(stroke_iou)
    return np.stack(stroke_pixel_iou)


def cal_stroke_pixel_iou_image(image_occ_gt, image_occ_pred, binary_threshold=1.0):
    '''
    image_occ_gt / image_occ_pred: (N, H, W), [0.0-stroke, 1.0-BG]
    '''
    stroke_pixel_iou = []
    for s_i in range(len(image_occ_gt)):
        stroke_gt_img = image_occ_gt[s_i]  # (H, W), [0.0-stroke, 1.0-BG]
        stroke_gt_mask = stroke_gt_img < binary_threshold  # (H, W)

        stroke_pred_img = image_occ_pred[s_i]  # (H, W), [0.0-stroke, 1.0-BG]
        stroke_pred_mask = stroke_pred_img < binary_threshold  # (H, W)

        stroke_intersection = np.logical_and(stroke_gt_mask, stroke_pred_mask)
        stroke_union = np.logical_or(stroke_gt_mask, stroke_pred_mask)

        stroke_intersection_pixel_num = np.sum(stroke_intersection)
        stroke_union_pixel_num = np.sum(stroke_union)
        if stroke_union_pixel_num != 0.0:
            stroke_iou = stroke_intersection_pixel_num / stroke_union_pixel_num
        else:
            stroke_iou = 1.0
        stroke_pixel_iou.append(stroke_iou)
    return np.stack(stroke_pixel_iou)


def cal_single_stroke_size(stroke_points):
    # stroke_points: (4, 2)
    p_start = np.array(stroke_points[0], dtype=np.float32)  # (2)
    p_end = np.array(stroke_points[-1], dtype=np.float32)  # (2)
    stroke_size_dist = np.abs(p_start - p_end)  # (2), full size
    return stroke_size_dist


def disturb_endpoint(stroke_points, is_start, occluded_map, image_size, try_times=10, offset_distance_threshold=0.33):
    '''
    stroke_points: (4, 2), in full size
    occluded_map: (H, W), [0-occluded, 1-visible]
    '''
    stroke_sizes = cal_single_stroke_size(stroke_points)  # (2), [x, y]
    candidate_point = stroke_points[0] if is_start else stroke_points[-1]

    try_num = 0
    while True:
        offset = np.random.random(size=(2)) * 2.0 - 1.0  # (2), [-1.0, 1.0)
        offset *= stroke_sizes * offset_distance_threshold
        disturb_point = candidate_point + offset  # (2)
        disturb_point = np.clip(disturb_point, 0.0, float(image_size - 1))
        px, py = int(disturb_point[0]), int(disturb_point[1])
        if occluded_map[py, px] == 0:
            if is_start:
                stroke_points_dist = np.concatenate([np.expand_dims(disturb_point, axis=0),
                                                     stroke_points[1:]], axis=0)  # (4, 2), in full size
            else:
                stroke_points_dist = np.concatenate([stroke_points[0:3],
                                                     np.expand_dims(disturb_point, axis=0)], axis=0)  # (4, 2), in full size
            return stroke_points_dist

        try_num += 1
        if try_num >= try_times:
            break

    return stroke_points


