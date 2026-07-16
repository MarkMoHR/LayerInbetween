import torch
import torch.nn.functional as F

import numpy as np
import math
from image_utils.image_processing import draw_segment

import pydiffvg

pydiffvg.set_use_gpu(torch.cuda.is_available())
# print('Setting pydiffvg.set_use_gpu:', torch.cuda.is_available())


def get_coordconv(raster_size):
    xx_ones = torch.ones(raster_size, dtype=torch.int32)  # e.g. (raster_size)
    xx_ones = xx_ones.unsqueeze(dim=-1)  # e.g. (raster_size, 1)
    xx_range = torch.arange(raster_size, dtype=torch.int32)  # e.g. (raster_size)
    xx_range = xx_range.unsqueeze(0)  # e.g. (1, raster_size)

    xx_channel = torch.matmul(xx_ones, xx_range)  # e.g. (raster_size, raster_size)
    xx_channel = xx_channel.unsqueeze(0)  # e.g. (1, raster_size, raster_size)

    yy_ones = torch.ones(raster_size, dtype=torch.int32)  # e.g. (raster_size)
    yy_ones = yy_ones.unsqueeze(0)  # e.g. (1, raster_size)
    yy_range = torch.arange(raster_size, dtype=torch.int32)  # (raster_size)
    yy_range = yy_range.unsqueeze(-1)  # e.g. (raster_size, 1)

    yy_channel = torch.matmul(yy_range, yy_ones)  # e.g. (raster_size, raster_size)
    yy_channel = yy_channel.unsqueeze(0)  # e.g. (1, raster_size, raster_size)

    xx_channel = xx_channel.float() / (raster_size - 1)
    yy_channel = yy_channel.float() / (raster_size - 1)
    # xx_channel = xx_channel * 2 - 1  # [-1, 1]
    # yy_channel = yy_channel * 2 - 1

    # xx_channel = xx_channel.cuda()
    # yy_channel = yy_channel.cuda()

    ret = torch.cat([
        xx_channel,
        yy_channel,
    ], dim=0)  # (2, raster_size, raster_size)
    ret = ret.detach()

    return ret


def add_coords(input_tensor, coordconv_input):
    batch_size = input_tensor.size()[0]  # get N size
    coords = torch.unsqueeze(coordconv_input, dim=0).repeat(batch_size, 1, 1, 1)  # (N, 2, raster_size, raster_size)
    coords = coords.to(input_tensor.device)
    result = torch.cat([input_tensor, coords], dim=1)  # (N, C+2, raster_size, raster_size)
    return result


def normalize_image_m1to1(in_img_0to1):
    norm_img_m1to1 = torch.mul(in_img_0to1, 2.0)
    norm_img_m1to1 = torch.sub(norm_img_m1to1, 1.0)
    return norm_img_m1to1


def rendering_curve_image(curve_params_batch, stroke_thickness, image_size):
    """
    :param curve_params_batch: (N, 4, 2)
    """
    batch_size, _, _ = curve_params_batch.shape
    curve_image_batch = []

    for batch_i in range(batch_size):
        shapes = []
        shape_groups = []

        curve_params = curve_params_batch[batch_i]  # (4, 2)

        num_control_points = torch.tensor([2])
        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=curve_params,
                             is_closed=False,
                             stroke_width=torch.tensor(stroke_thickness))
        shapes.append(path)

        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]),
                                         fill_color=None,
                                         stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
        shape_groups.append(path_group)

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            image_size, image_size, shapes, shape_groups)

        background = torch.ones(image_size, image_size, 4)

        render = pydiffvg.RenderFunction.apply
        img = render(image_size,  # width
                     image_size,  # height
                     2,  # num_samples_x
                     2,  # num_samples_y
                     0,  # seed
                     background,  # background_image
                     *scene_args)  # (H, W, 4), [0.0-stroke, 1.0-BG]
        curve_img = img[:, :, 0]  # (H, W), [0.0-stroke, 1.0-BG]
        curve_image_batch.append(curve_img)

    curve_image_batch = torch.stack(curve_image_batch, dim=0)  # (N, H, W), [0.0-stroke, 1.0-BG]
    return curve_image_batch


def rendering_line_image(line_params_batch, stroke_thickness, image_size):
    """
    :param line_params_batch: (N, 2, 2)
    """
    batch_size, _, _ = line_params_batch.shape
    line_image_batch = []

    # no need to calculate gradient
    with torch.no_grad():
        for batch_i in range(batch_size):
            shapes = []
            shape_groups = []

            line_params = line_params_batch[batch_i]  # (2, 2)

            path = pydiffvg.Polygon(points=line_params,
                                    is_closed=False,
                                    stroke_width=torch.tensor(stroke_thickness))
            shapes.append(path)

            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]),
                                             fill_color=None,
                                             stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
            shape_groups.append(path_group)

            scene_args = pydiffvg.RenderFunction.serialize_scene(
                image_size, image_size, shapes, shape_groups)

            background = torch.ones(image_size, image_size, 4)

            render = pydiffvg.RenderFunction.apply
            img = render(image_size,  # width
                         image_size,  # height
                         2,  # num_samples_x
                         2,  # num_samples_y
                         0,  # seed
                         background,  # background_image
                         *scene_args)  # (H, W, 4), [0.0-stroke, 1.0-BG]
            line_img = img[:, :, 0]  # (H, W), [0.0-stroke, 1.0-BG]
            line_image_batch.append(line_img)

    line_image_batch = torch.stack(line_image_batch, dim=0)  # (N, H, W), [0.0-stroke, 1.0-BG]
    return line_image_batch


def gen_segment_imgs_on_the_fly(end_ctrl_offset, raster_size):
    '''
    end_ctrl_offset: (N, 1, 8), [-1.0, 1.0], relative to target stroke window
    '''
    endpoint0_offsets, _, _, endpoint3_offsets = torch.split(end_ctrl_offset, 2, dim=-1)  # each of (N, 1, 2), [-1.0, 1.0]
    stroke_num = end_ctrl_offset.size()[0]
    endpoints_offsets = torch.cat([endpoint0_offsets, endpoint3_offsets], dim=-1)  # (N, 1, 4), [-1.0, 1.0]
    endpoints_offsets_np = endpoints_offsets.squeeze(dim=1).cpu().data.numpy()  # (N, 4), [-1.0, 1.0]

    segment_imgs = []
    for s_i in range(stroke_num):
        endpoints_offset_np = endpoints_offsets_np[s_i]  # (4), [-1.0, 1.0]
        segment_image = draw_segment(endpoints_offset_np, raster_size=raster_size)  # (raster_size, raster_size), [0.0-stroke, 1.0-BG]
        segment_imgs.append(segment_image)
    segment_imgs = np.expand_dims(np.stack(segment_imgs, axis=0), axis=-1)  # (N, raster_size, raster_size, 1), [0.0-strokes, 1.0-BG]

    segment_imgs = torch.tensor(segment_imgs).float()
    segment_imgs = segment_imgs.cuda()
    return segment_imgs


def spatial_transform_rot_shear_point(points_pos, rotation_angle, shear_x_angle=None, shear_y_angle=None):
    """
    :param points_pos: (N, 2), [-1.0, 1.0]
    :param rotation_angle: (N, 1), in degree
    :param shear_x_angle / shear_y_angle: (N, 1), in degree
    """
    ones_tensor = torch.ones_like(rotation_angle)  # (N, 1)
    zeros_tensor = torch.zeros_like(rotation_angle)  # (N, 1)

    rotation_angle_norm = rotation_angle / 180.0 * math.pi
    rotate_matrix = torch.cat([torch.cos(rotation_angle_norm), torch.sin(rotation_angle_norm), zeros_tensor,
                               -torch.sin(rotation_angle_norm), torch.cos(rotation_angle_norm), zeros_tensor,
                               zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
    rotate_matrix = torch.reshape(rotate_matrix, (-1, 3, 3))  # (N, 3, 3)

    points_pos_trans = torch.cat([points_pos, ones_tensor], dim=-1).unsqueeze(dim=1)  # (N, 1, 3)
    points_pos_trans = torch.matmul(points_pos_trans, rotate_matrix)  # (N, 1, 3)

    if shear_x_angle is not None:
        assert shear_y_angle is not None
        shear_x_angle_norm = -shear_x_angle / 180.0 * math.pi
        shear_y_angle_norm = -shear_y_angle / 180.0 * math.pi
        shear_x_matrix = torch.cat([ones_tensor, torch.tan(shear_x_angle_norm), zeros_tensor,
                                    zeros_tensor, ones_tensor, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        shear_y_matrix = torch.cat([ones_tensor, zeros_tensor, zeros_tensor,
                                    torch.tan(shear_y_angle_norm), ones_tensor, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        shear_x_matrix = torch.reshape(shear_x_matrix, (-1, 3, 3))  # (N, 3, 3)
        shear_y_matrix = torch.reshape(shear_y_matrix, (-1, 3, 3))  # (N, 3, 3)

        points_pos_trans = torch.permute(points_pos_trans, (0, 2, 1))  # (N, 3, 1)
        points_pos_trans = torch.matmul(shear_x_matrix, points_pos_trans)  # (N, 3, 1)
        points_pos_trans = torch.matmul(shear_y_matrix, points_pos_trans)  # (N, 3, 1)
        points_pos_trans = torch.permute(points_pos_trans, (0, 2, 1))  # (N, 1, 3)

    points_pos_trans = points_pos_trans.squeeze(dim=1)  # (N, 3)
    points_pos_trans = points_pos_trans[:, 0:2]  # (N, 2), might larger than 1.0
    return points_pos_trans


def spatial_transform_point_with_additional(point_offset, based_window_size, based_cursor,
                                            trans_cursor, trans_win_size, trans_rot_angle, trans_shear_x, trans_shear_y,
                                            addi_offset, addi_scaling):
    point_global = point_offset * (based_window_size / 2.0) + based_cursor  # (N, 1, 2), in full size
    point_offset_global = point_global - trans_cursor  # (N, 1, 2)
    point_offset_inv = point_offset_global / (trans_win_size / 2.0)  # (N, 1, 2), [-1.0+, 1.0+]
    point_offset_inv = spatial_transform_rot_shear_point(point_offset_inv.squeeze(dim=1), trans_rot_angle,
                                                         shear_x_angle=trans_shear_x, shear_y_angle=trans_shear_y)  # (N, 2), [-1.0+, 1.0+]
    point_offset_trans = (point_offset_inv.unsqueeze(dim=1) - addi_offset) / addi_scaling  # (N, 1, 2), [-1.0, 1.0]
    return point_offset_trans


def spatial_transform_stroke_with_additional(all_points_offset, based_window_size, based_cursor,
                                             trans_cursor, trans_win_size, trans_rot_angle, trans_shear_x, trans_shear_y,
                                             addi_offset, addi_scaling):
    '''
    all_points_offset: (N, 1, 8), [-1.0, 1.0]
    '''
    all_points_offset_split = torch.split(all_points_offset, 2, dim=-1)  # list (4) of  (N, 1, 2), float32, [-1.0, 1.0]
    assert len(all_points_offset_split) == 4
    all_points_offset_split_trans = []
    for point_offset in all_points_offset_split:
        point_offset_trans = spatial_transform_point_with_additional(
            point_offset, based_window_size, based_cursor,
            trans_cursor, trans_win_size, trans_rot_angle, trans_shear_x, trans_shear_y,
            addi_offset, addi_scaling
        )
        all_points_offset_split_trans.append(point_offset_trans)
    all_points_offset_trans = torch.cat(all_points_offset_split_trans, dim=-1)  # (N, 1, 8), [-1.0, 1.0]
    return all_points_offset_trans


def spatial_transform_reverse_rot_shear_point(points_pos, rotation_angle, shear_x_angle=None, shear_y_angle=None):
    """
    :param points_pos: (N, 2), [-1.0, 1.0]
    :param rotation_angle: (N, 1), in degree
    :param shear_x_angle / shear_y_angle: (N, 1), in degree
    """
    ones_tensor = torch.ones_like(rotation_angle)  # (N, 1)
    zeros_tensor = torch.zeros_like(rotation_angle)  # (N, 1)

    rotation_angle_norm_re = -rotation_angle / 180.0 * math.pi
    rotate_matrix = torch.cat([torch.cos(rotation_angle_norm_re), torch.sin(rotation_angle_norm_re), zeros_tensor,
                               -torch.sin(rotation_angle_norm_re), torch.cos(rotation_angle_norm_re), zeros_tensor,
                               zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
    rotate_matrix = torch.reshape(rotate_matrix, (-1, 3, 3))  # (N, 3, 3)

    points_pos_re = torch.cat([points_pos, ones_tensor], dim=-1).unsqueeze(dim=1)  # (N, 1, 3)

    if shear_x_angle is not None:
        assert shear_y_angle is not None
        shear_x_angle_norm_re = shear_x_angle / 180.0 * math.pi
        shear_y_angle_norm_re = shear_y_angle / 180.0 * math.pi
        shear_x_matrix = torch.cat([ones_tensor, torch.tan(shear_x_angle_norm_re), zeros_tensor,
                                    zeros_tensor, ones_tensor, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        shear_y_matrix = torch.cat([ones_tensor, zeros_tensor, zeros_tensor,
                                    torch.tan(shear_y_angle_norm_re), ones_tensor, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        shear_x_matrix = torch.reshape(shear_x_matrix, (-1, 3, 3))  # (N, 3, 3)
        shear_y_matrix = torch.reshape(shear_y_matrix, (-1, 3, 3))  # (N, 3, 3)

        points_pos_re = torch.permute(points_pos_re, (0, 2, 1))  # (N, 3, 1)
        points_pos_re = torch.matmul(shear_y_matrix, points_pos_re)  # (N, 3, 1)
        points_pos_re = torch.matmul(shear_x_matrix, points_pos_re)  # (N, 3, 1)
        points_pos_re = torch.permute(points_pos_re, (0, 2, 1))  # (N, 1, 3)

    points_pos_re = torch.matmul(points_pos_re, rotate_matrix).squeeze(dim=1)  # (N, 1, 3) => (N, 3)
    points_pos_re = points_pos_re[:, 0:2]  # (N, 2), might larger than 1.0
    return points_pos_re


def spatial_transform_reverse_point_with_additional(point_offset_trans, based_window_size, based_cursor,
                                                    trans_cursor, trans_win_size,
                                                    trans_rot_angle, trans_shear_x, trans_shear_y,
                                                    addi_offset, addi_scaling):
    # point_offset_trans: (N, 1, 2), [-1.0, 1.0]
    point_offset_inv = point_offset_trans * addi_scaling + addi_offset  # (N, 1, 2), [-1.0, 1.0]
    point_offset_inv = spatial_transform_reverse_rot_shear_point(point_offset_inv.squeeze(dim=1), trans_rot_angle,
                                                                 shear_x_angle=trans_shear_x,
                                                                 shear_y_angle=trans_shear_y)  # (N, 2), [-1.0+, 1.0+]

    point_offset_global = point_offset_inv.unsqueeze(dim=1) * (trans_win_size / 2.0)  # (N, 1, 2), in image size
    point_global = trans_cursor + point_offset_global  # (N, 1, 2), in image size
    point_offset = (point_global - based_cursor) / (based_window_size / 2.0)  # (N, 1, 2), [-1.0, 1.0]
    return point_offset


def spatial_transform_reverse_stroke_with_additional(all_points_offset_trans, based_window_size, based_cursor,
                                                     trans_cursor, trans_win_size,
                                                     trans_rot_angle, trans_shear_x, trans_shear_y,
                                                     addi_offset, addi_scaling):
    '''
    all_points_offset_trans: (N, 1, 8), [-1.0, 1.0]
    '''
    all_points_offset_trans_split = torch.split(all_points_offset_trans, 2, dim=-1)  # list (4) of  (N, 1, 2), float32, [-1.0, 1.0]
    assert len(all_points_offset_trans_split) == 4
    all_points_offset_split = []
    for point_offset_trans in all_points_offset_trans_split:
        point_offset = spatial_transform_reverse_point_with_additional(
            point_offset_trans, based_window_size, based_cursor,
            trans_cursor, trans_win_size, trans_rot_angle, trans_shear_x, trans_shear_y,
            addi_offset, addi_scaling
        )
        all_points_offset_split.append(point_offset)
    all_points_offset = torch.cat(all_points_offset_split, dim=-1)  # (N, 1, 8), [-1.0, 1.0]
    return all_points_offset


def image_cropping_stn(cursor_position, input_img, image_size, raster_size, window_sizes_in, rotation_angle=None,
                       shear_x_angle=None, shear_y_angle=None,
                       additional_transform=False, addi_offset=None, addi_scale=None):
    """
    :param cursor_position: (N, 1, 2), float type, in size [0.0, 1.0)
    :param input_img: [0.0-stroke, 1.0-BG]
    :param window_sizes_in: (N, 1, 2), float type, in full size
    :param rotation_angle: (N, 1), in degree
    :param shear_x_angle / shear_y_angle: (N, 1), in degree
    :param addi_offset: (N, 1, 2), [-1, 1]
    :param addi_scale: (N, 1, 2), [0, 1+]
    """
    center_pos = cursor_position.squeeze(dim=1)  # (N, 2), float type, in size [0.0, 1.0)
    window_size = window_sizes_in.squeeze(dim=1)  # (N, 2), float type, in full size
    img = input_img.permute(0, 3, 1, 2)

    # if not window_size_cropping_grad:
    #     window_size = window_size.detach()

    center_pos_norm = center_pos * 2.0 - 1.0  # (N, 2), [-1.0, 1.0]
    center_pos_x, center_pos_y = torch.split(center_pos_norm, 1, dim=-1)  # (N, 1), [-1.0, 1.0]
    window_size_norm = window_size / float(image_size)  # (N, 2), [0.0, 1.0]
    window_size_x, window_size_y = torch.split(window_size_norm, 1, dim=-1)  # (N, 1), [0.0, 1.0]

    batch_size = img.size(0)
    channel = img.size(1)

    ones_tensor = torch.ones_like(center_pos_x)  # (N, 1)
    zeros_tensor = torch.zeros_like(center_pos_x)  # (N, 1)

    # shifting
    translate_matrix = torch.cat([ones_tensor, zeros_tensor, center_pos_x,
                                  zeros_tensor, ones_tensor, center_pos_y,
                                  zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
    translate_matrix = torch.reshape(translate_matrix, (-1, 3, 3))  # (N, 3, 3)

    if additional_transform and addi_offset is not None:
        center_pos_x2, center_pos_y2 = torch.split(addi_offset.squeeze(dim=1), 1, dim=-1)  # (N, 1), [-1.0, 1.0]
        translate_matrix2 = torch.cat([ones_tensor, zeros_tensor, center_pos_x2,
                                      zeros_tensor, ones_tensor, center_pos_y2,
                                      zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        translate_matrix2 = torch.reshape(translate_matrix2, (-1, 3, 3))  # (N, 3, 3)

    # scaling
    scaling_matrix = torch.cat([window_size_x, zeros_tensor, zeros_tensor,
                                zeros_tensor, window_size_y, zeros_tensor,
                                zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
    scaling_matrix = torch.reshape(scaling_matrix, (-1, 3, 3))  # (N, 3, 3)

    if additional_transform and addi_scale is not None:
        window_size_x2, window_size_y2 = torch.split(addi_scale.squeeze(dim=1), 1, dim=-1)  # (N, 1), [0.0, 1.0+]
        scaling_matrix2 = torch.cat([window_size_x2, zeros_tensor, zeros_tensor,
                                    zeros_tensor, window_size_y2, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        scaling_matrix2 = torch.reshape(scaling_matrix2, (-1, 3, 3))  # (N, 3, 3)

    # rotation
    if rotation_angle is not None:
        rotation_angle_norm = rotation_angle / 180.0 * math.pi
        # if not window_size_cropping_grad:
        #     rotation_angle_norm = rotation_angle_norm.detach()
        rotate_matrix = torch.cat([torch.cos(rotation_angle_norm), torch.sin(rotation_angle_norm), zeros_tensor,
                                   -torch.sin(rotation_angle_norm), torch.cos(rotation_angle_norm), zeros_tensor,
                                   zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        rotate_matrix = torch.reshape(rotate_matrix, (-1, 3, 3))  # (N, 3, 3)
        matrix = torch.matmul(torch.matmul(translate_matrix, scaling_matrix), rotate_matrix)
    else:
        matrix = torch.matmul(translate_matrix, scaling_matrix)

    # shearing
    if shear_x_angle is not None:
        assert shear_y_angle is not None
        shear_x_angle_norm = shear_x_angle / 180.0 * math.pi
        shear_y_angle_norm = shear_y_angle / 180.0 * math.pi
        shear_x_matrix = torch.cat([ones_tensor, torch.tan(shear_x_angle_norm), zeros_tensor,
                                    zeros_tensor, ones_tensor, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        shear_y_matrix = torch.cat([ones_tensor, zeros_tensor, zeros_tensor,
                                    torch.tan(shear_y_angle_norm), ones_tensor, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        shear_x_matrix = torch.reshape(shear_x_matrix, (-1, 3, 3))  # (N, 3, 3)
        shear_y_matrix = torch.reshape(shear_y_matrix, (-1, 3, 3))  # (N, 3, 3)
        matrix = torch.matmul(torch.matmul(matrix, shear_x_matrix), shear_y_matrix)

    if additional_transform:
        matrix = torch.matmul(torch.matmul(matrix, translate_matrix2), scaling_matrix2)

    matrix = matrix[:, 0:2, :]

    affine_grid_points = F.affine_grid(matrix, size=[batch_size, channel, raster_size, raster_size],
                                       align_corners=False)
    rois = F.grid_sample(1.0 - img, affine_grid_points, align_corners=False)
    rois = 1.0 - rois  # (N, C, raster_size, raster_size), [0.0-stroke, 1.0-BG]
    rois = rois.permute(0, 2, 3, 1)  # (N, raster_size, raster_size, C), [0.0-stroke, 1.0-BG]
    return rois


def image_cropping_stn_multi(cursor_position, input_img, image_size, raster_size, window_sizes_in, rotation_angle=None,
                             shear_x_angle=None, shear_y_angle=None,
                             additional_transform=False, addi_offset=None, addi_scale=None,
                             additional_transform3=False,
                             addi_offset3=None, addi_scale3=None, addi_rotate3=None, addi_shear_x3=None, addi_shear_y3=None,
                             additional_transform4=False, addi_scale4=None,
                             additional_transform5=False, addi_offset5=None, addi_scale5=None):
    """
    :param cursor_position: (N, 1, 2), float type, in size [0.0, 1.0)
    :param input_img: [0.0-stroke, 1.0-BG]
    :param window_sizes_in: (N, 1, 2), float type, in full size
    :param rotation_angle: (N, 1), in degree
    :param shear_x_angle / shear_y_angle: (N, 1), in degree
    :param addi_offset: (N, 1, 2), [-1, 1]
    :param addi_scale: (N, 1, 2), [0, 1+]
    :param addi_offset3: (N, 1, 2), [-1, 1]
    :param addi_scale3: (N, 1, 2), [0, 1+]
    :param addi_rotate3: (N, 1), in degree
    :param addi_shear_x3 / addi_shear_y3: (N, 1), in degree
    :param addi_scale4: (N, 1, 2), [0, 1+]
    """
    center_pos = cursor_position.squeeze(dim=1)  # (N, 2), float type, in size [0.0, 1.0)
    window_size = window_sizes_in.squeeze(dim=1)  # (N, 2), float type, in full size
    img = input_img.permute(0, 3, 1, 2)

    # if not window_size_cropping_grad:
    #     window_size = window_size.detach()

    center_pos_norm = center_pos * 2.0 - 1.0  # (N, 2), [-1.0, 1.0]
    center_pos_x, center_pos_y = torch.split(center_pos_norm, 1, dim=-1)  # (N, 1), [-1.0, 1.0]
    window_size_norm = window_size / float(image_size)  # (N, 2), [0.0, 1.0]
    window_size_x, window_size_y = torch.split(window_size_norm, 1, dim=-1)  # (N, 1), [0.0, 1.0]

    batch_size = img.size(0)
    channel = img.size(1)

    ones_tensor = torch.ones_like(center_pos_x)  # (N, 1)
    zeros_tensor = torch.zeros_like(center_pos_x)  # (N, 1)

    # shifting
    translate_matrix = torch.cat([ones_tensor, zeros_tensor, center_pos_x,
                                  zeros_tensor, ones_tensor, center_pos_y,
                                  zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
    translate_matrix = torch.reshape(translate_matrix, (-1, 3, 3))  # (N, 3, 3)

    if additional_transform and addi_offset is not None:
        center_pos_x2, center_pos_y2 = torch.split(addi_offset.squeeze(dim=1), 1, dim=-1)  # (N, 1), [-1.0, 1.0]
        translate_matrix2 = torch.cat([ones_tensor, zeros_tensor, center_pos_x2,
                                      zeros_tensor, ones_tensor, center_pos_y2,
                                      zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        translate_matrix2 = torch.reshape(translate_matrix2, (-1, 3, 3))  # (N, 3, 3)

    if additional_transform3 and addi_offset3 is not None:
        center_pos_x3, center_pos_y3 = torch.split(addi_offset3.squeeze(dim=1), 1, dim=-1)  # (N, 1), [-1.0, 1.0]
        translate_matrix3 = torch.cat([ones_tensor, zeros_tensor, center_pos_x3,
                                      zeros_tensor, ones_tensor, center_pos_y3,
                                      zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        translate_matrix3 = torch.reshape(translate_matrix3, (-1, 3, 3))  # (N, 3, 3)

    if additional_transform5 and addi_offset5 is not None:
        center_pos_x5, center_pos_y5 = torch.split(addi_offset5.squeeze(dim=1), 1, dim=-1)  # (N, 1), [-1.0, 1.0]
        translate_matrix5 = torch.cat([ones_tensor, zeros_tensor, center_pos_x5,
                                      zeros_tensor, ones_tensor, center_pos_y5,
                                      zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        translate_matrix5 = torch.reshape(translate_matrix5, (-1, 3, 3))  # (N, 3, 3)

    # scaling
    scaling_matrix = torch.cat([window_size_x, zeros_tensor, zeros_tensor,
                                zeros_tensor, window_size_y, zeros_tensor,
                                zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
    scaling_matrix = torch.reshape(scaling_matrix, (-1, 3, 3))  # (N, 3, 3)

    if additional_transform and addi_scale is not None:
        window_size_x2, window_size_y2 = torch.split(addi_scale.squeeze(dim=1), 1, dim=-1)  # (N, 1), [0.0, 1.0+]
        scaling_matrix2 = torch.cat([window_size_x2, zeros_tensor, zeros_tensor,
                                    zeros_tensor, window_size_y2, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        scaling_matrix2 = torch.reshape(scaling_matrix2, (-1, 3, 3))  # (N, 3, 3)

    if additional_transform3 and addi_scale3 is not None:
        window_size_x3, window_size_y3 = torch.split(addi_scale3.squeeze(dim=1), 1, dim=-1)  # (N, 1), [0.0, 1.0+]
        scaling_matrix3 = torch.cat([window_size_x3, zeros_tensor, zeros_tensor,
                                    zeros_tensor, window_size_y3, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        scaling_matrix3 = torch.reshape(scaling_matrix3, (-1, 3, 3))  # (N, 3, 3)

    if additional_transform4 and addi_scale4 is not None:
        window_size_x4, window_size_y4 = torch.split(addi_scale4.squeeze(dim=1), 1, dim=-1)  # (N, 1), [0.0, 1.0+]
        scaling_matrix4 = torch.cat([window_size_x4, zeros_tensor, zeros_tensor,
                                    zeros_tensor, window_size_y4, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        scaling_matrix4 = torch.reshape(scaling_matrix4, (-1, 3, 3))  # (N, 3, 3)

    if additional_transform5 and addi_scale5 is not None:
        window_size_x5, window_size_y5 = torch.split(addi_scale5.squeeze(dim=1), 1, dim=-1)  # (N, 1), [0.0, 1.0+]
        scaling_matrix5 = torch.cat([window_size_x5, zeros_tensor, zeros_tensor,
                                    zeros_tensor, window_size_y5, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        scaling_matrix5 = torch.reshape(scaling_matrix5, (-1, 3, 3))  # (N, 3, 3)

    # rotation
    if rotation_angle is not None:
        rotation_angle_norm = rotation_angle / 180.0 * math.pi
        rotate_matrix = torch.cat([torch.cos(rotation_angle_norm), torch.sin(rotation_angle_norm), zeros_tensor,
                                   -torch.sin(rotation_angle_norm), torch.cos(rotation_angle_norm), zeros_tensor,
                                   zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        rotate_matrix = torch.reshape(rotate_matrix, (-1, 3, 3))  # (N, 3, 3)
        matrix = torch.matmul(torch.matmul(translate_matrix, scaling_matrix), rotate_matrix)
    else:
        matrix = torch.matmul(translate_matrix, scaling_matrix)

    if additional_transform3 and addi_rotate3 is not None:
        rotation_angle_norm3 = addi_rotate3 / 180.0 * math.pi
        rotate_matrix3 = torch.cat([torch.cos(rotation_angle_norm3), torch.sin(rotation_angle_norm3), zeros_tensor,
                                   -torch.sin(rotation_angle_norm3), torch.cos(rotation_angle_norm3), zeros_tensor,
                                   zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        rotate_matrix3 = torch.reshape(rotate_matrix3, (-1, 3, 3))  # (N, 3, 3)

    # shearing
    if shear_x_angle is not None:
        assert shear_y_angle is not None
        shear_x_angle_norm = shear_x_angle / 180.0 * math.pi
        shear_y_angle_norm = shear_y_angle / 180.0 * math.pi
        shear_x_matrix = torch.cat([ones_tensor, torch.tan(shear_x_angle_norm), zeros_tensor,
                                    zeros_tensor, ones_tensor, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        shear_y_matrix = torch.cat([ones_tensor, zeros_tensor, zeros_tensor,
                                    torch.tan(shear_y_angle_norm), ones_tensor, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        shear_x_matrix = torch.reshape(shear_x_matrix, (-1, 3, 3))  # (N, 3, 3)
        shear_y_matrix = torch.reshape(shear_y_matrix, (-1, 3, 3))  # (N, 3, 3)
        matrix = torch.matmul(torch.matmul(matrix, shear_x_matrix), shear_y_matrix)

    if additional_transform3 and addi_shear_x3 is not None:
        assert addi_shear_y3 is not None
        shear_x_angle_norm3 = addi_shear_x3 / 180.0 * math.pi
        shear_y_angle_norm3 = addi_shear_y3 / 180.0 * math.pi
        shear_x_matrix3 = torch.cat([ones_tensor, torch.tan(shear_x_angle_norm3), zeros_tensor,
                                    zeros_tensor, ones_tensor, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        shear_y_matrix3 = torch.cat([ones_tensor, zeros_tensor, zeros_tensor,
                                    torch.tan(shear_y_angle_norm3), ones_tensor, zeros_tensor,
                                    zeros_tensor, zeros_tensor, ones_tensor], dim=-1)  # (N, 9)
        shear_x_matrix3 = torch.reshape(shear_x_matrix3, (-1, 3, 3))  # (N, 3, 3)
        shear_y_matrix3 = torch.reshape(shear_y_matrix3, (-1, 3, 3))  # (N, 3, 3)

    if additional_transform:
        matrix = torch.matmul(torch.matmul(matrix, translate_matrix2), scaling_matrix2)

    if additional_transform3:
        matrix = torch.matmul(torch.matmul(torch.matmul(torch.matmul(
            torch.matmul(matrix, translate_matrix3), scaling_matrix3), rotate_matrix3), shear_x_matrix3), shear_y_matrix3)

    if additional_transform4:
        matrix = torch.matmul(matrix, scaling_matrix4)

    if additional_transform5:
        matrix = torch.matmul(torch.matmul(matrix, translate_matrix5), scaling_matrix5)

    matrix = matrix[:, 0:2, :]

    affine_grid_points = F.affine_grid(matrix, size=[batch_size, channel, raster_size, raster_size],
                                       align_corners=False)
    rois = F.grid_sample(1.0 - img, affine_grid_points, align_corners=False)
    rois = 1.0 - rois  # (N, C, raster_size, raster_size), [0.0-stroke, 1.0-BG]
    rois = rois.permute(0, 2, 3, 1)  # (N, raster_size, raster_size, C), [0.0-stroke, 1.0-BG]
    return rois


def load_weights(pre_train_path, network):
    pretrained_dict = torch.load(pre_train_path)
    model_dict = network.state_dict()
    # print('pretrained_dict')
    # print(pretrained_dict.keys())
    # print('model_dict')
    # print(model_dict.keys())

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    print('Loaded', pre_train_path)


def print_model_variables(param_list, model_name):
    print('-' * 100)
    print('#', model_name)
    count_t_vars = 0
    for name, param in param_list:
        num_param = np.prod(list(param.size()))
        count_t_vars += num_param
        print('%s | shape: %s | num_param: %i |' % (name, str(param.size()), num_param), 'requires_grad:', param.requires_grad)
    print('Total trainable variables of %s: %i.' % (model_name, count_t_vars))
    # print('-' * 100)
    return count_t_vars
