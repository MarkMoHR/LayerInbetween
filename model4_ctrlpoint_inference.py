import tensorflow as tf
from logger import Logger
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time
import os
from PIL import Image
import jsonlines

from hparam import HParams
from image_utils.image_processing import save_image, save_image_overlap
from image_utils.image_processing import draw_sketch_stroke, cal_stroke_pixel_iou
from image_utils.model_processing import get_coordconv, add_coords, normalize_image_m1to1, image_cropping_stn, image_cropping_stn_multi, \
    load_weights, spatial_transform_stroke_with_additional, spatial_transform_reverse_stroke_with_additional, gen_segment_imgs_on_the_fly
from network.vanilla import CNN_Encoder, MLP_Decoder, CNN_SepEncoder
from configs.example_configs import test_img_id

tf.get_logger().setLevel('INFO')


def get_default_hparams():
    """Return default HParams for sketch-rnn."""
    hparams = HParams(
        workspace='FAD3-CP46-sep-dist',
        transform_model_name='FAD3-T12-2.0x-51-min=64',  # 'FAD-2.0x-51' / 'FAD2-2.0x-51-v3' / 'FAD3-2.0x-51-min=64' / 'FAD3-T12-2.0x-51-min=64'
        transform_local_model_name='FAD3-T13-2.0x-51',  # FAD3-T13-2.0x-51 / FAD3-T13-1.5x-51
        endpoint_model_name='FAD3-EPO35-1.5x-min=64',

        ############ For inference only ############
        do_dataset_filtering=False,  # set to True for training
        ############################################

        dataset_base='/home/Datasets/CreativeSketch/proc_data3/',

        multi_gpu=False,

        num_steps=30000,
        save_every=10000,  # Number of steps per checkpoint creation.
        log_img_every=500,  # Number of steps per log image creation.

        batch_size=25,

        # image_size=512,

        enc_model_ctrlpoint='separated',  # ['combined', 'separated']
        dec_model_ctrlpoint='mlp',  # ['rnn', 'mlp']
        z_size=256,  # Size of latent vector z.
        ctrlpoint_module_zero_init='last',  # ['none', 'last', 'all']
        add_coordconv=True,

        vector_loss_type='L1',  # ['MSE', 'L1']

        grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.

        learning_rate=1e-4,  # Learning rate.
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        decay_power=0.9,
        min_learning_rate=1e-6,  # Minimum learning rate.

        snapshot_root='outputs/ctrlpoint/snapshot',
        log_img_root='outputs/ctrlpoint/log_img',
        log_root='outputs/ctrlpoint/log',
        inference_root='outputs/ctrlpoint/inference-FAD3',
        inference_full_root='outputs/ctrlpoint/inference_FULL-FAD3_split_pred',
        inference_full_real_root='outputs/stroke_correspondence_results',
    )
    return hparams


class Controlpoint_Model(nn.Module):
    def __init__(self, hps):
        super(Controlpoint_Model, self).__init__()
        self.hps = hps

        ctrlpoint_out_size = 4  # offset
        cnn_out_size = self.hps.z_size

        # ctrlpoint encoder
        if self.hps.enc_model_ctrlpoint == 'combined':
            cnn_in_size = 4
            if self.hps.add_coordconv:
                cnn_in_size += 2
            self.encoder_ctrlpoint = CNN_Encoder(cnn_in_size, cnn_out_size, input_size=self.hps.raster_size)
        elif self.hps.enc_model_ctrlpoint == 'separated':
            cnn_in_size_ref = 2
            cnn_in_size_tar = 2
            if self.hps.add_coordconv:
                cnn_in_size_ref += 2
                cnn_in_size_tar += 2
            self.encoder_ctrlpoint = CNN_SepEncoder(cnn_in_size_ref, cnn_in_size_tar, cnn_out_size, input_size=self.hps.raster_size)
        else:
            raise Exception('Unknown enc_model_ctrlpoint:', self.hps.enc_model_ctrlpoint)

        dec_in_size = self.hps.z_size
        if self.hps.dec_model_ctrlpoint == 'mlp':
            self.decoder_ctrlpoint = MLP_Decoder(dec_in_size, ctrlpoint_out_size, zero_init=self.hps.ctrlpoint_module_zero_init)
        else:
            raise Exception('Unknown dec_model_ctrlpoint:', self.hps.dec_model_ctrlpoint)

        if self.hps.add_coordconv:
            self.coordconv_input = get_coordconv(self.hps.raster_size)  # (2, raster_size, raster_size)

    def forward(self, reference_images, reference_strokes, reference_strokes_ctrl, target_images,
                centerpoints_pos_ref, centerpoints_pos_tar, base_window_size, base_window_size_single, image_size,
                end_ctrl_offset_ref,
                end_ctrl_offset_tar, end_offset_tar_pred,
                component_centerpoints, component_win_sizes,
                target_transform_cursors, target_transform_win_sizes, target_transform_angles,
                target_transform_shear_x_angles, target_transform_shear_y_angles,
                target_transform1_translate, target_transform1_scaling,
                target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y,
                fixing_states
                ):
        """
        :param reference_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_strokes: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_strokes_ctrl: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param centerpoints_pos_ref: (N, 1, 2), float32, in [0.0, 1.0]
        :param centerpoints_pos_tar: (N, 1, 2), float32, in [0.0, 1.0]
        :param base_window_size: (N, 1), float32, in [0.0, 1.0]
        :param base_window_size_single: (N, 1), float32, in [0.0, 1.0]
        :param end_ctrl_offset_ref: (N, 1, 8), float32, [-1.0, 1.0]
        :param end_ctrl_offset_tar: (N, 1, 8), float32, [-1.0, 1.0]
        :param end_offset_tar_pred: (N, 1, 4), float32, [-1.0, 1.0]
        :param component_centerpoints: (N, 1, 2), in [0.0, 1.0], relative to image size
        :param component_win_sizes: (N, 1, 2), in image size
        :param target_transform_cursors: (N, 1, 2), in [0.0, 1.0], relative to image size
        :param target_transform_win_sizes: (N, 1, 2), in image size
        :param target_transform_angles: (N, 1), in [-180.0, 180.0]
        :param target_transform_shear_x_angles: (N, 1), in [-90.0, 90.0]
        :param target_transform_shear_y_angles: (N, 1), in [-90.0, 90.0]
        :param target_transform1_translate: (N, 1, 2), [-1.0, 1.0], relative to target trans0 window
        :param target_transform1_scaling: (N, 1, 2), [0.2, 2.0], relative to target trans0 window
        :param target_transform1_rotate: (N, 1), [-180.0, 180.0]
        :param target_transform1_shear_x: (N, 1), [-90.0, 90.0]
        :param target_transform1_shear_y: (N, 1), [-90.0, 90.0]
        :param fixing_states: (N, 1), [0-nonfix, 1-fix]
        :return:
        """
        self.image_size = image_size

        patch_reference, patch_stroke_reference, \
            patch_reference2, patch_stroke_reference2, \
            patch_target_ori, patch_target_trans1, patch_target_trans2, patch_segment_target_trans2, \
            end_ctrl_offset_tar_trans2, ctrlpoints_offset_trans2_pred, ctrlpoints_offset_pred = \
            self.get_points_and_raster_image(reference_images, reference_strokes, reference_strokes_ctrl, target_images,
                                             centerpoints_pos_ref, centerpoints_pos_tar,
                                             base_window_size, base_window_size_single,
                                             end_ctrl_offset_ref,
                                             end_ctrl_offset_tar, end_offset_tar_pred,
                                             component_centerpoints, component_win_sizes,
                                             target_transform_cursors, target_transform_win_sizes, target_transform_angles,
                                             target_transform_shear_x_angles, target_transform_shear_y_angles,
                                             target_transform1_translate, target_transform1_scaling,
                                             target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y,
                                             fixing_states
                                             )
        # patch_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_stroke_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_reference2: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_stroke_reference2: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_target_ori: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_target_trans1: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_target_trans2: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_segment_target_trans2: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # end_ctrl_offset_tar_trans2: (N, 1, 8), [-1.0, 1.0], relative to target stroke window
        # ctrlpoints_offset_trans2_pred: (N, 4), [-1.0, 1.0], relative to target stroke window
        # ctrlpoints_offset_pred: (N, 4), [-1.0, 1.0], relative to ref window

        return patch_reference, patch_stroke_reference, \
            patch_reference2, patch_stroke_reference2, \
            patch_target_ori, patch_target_trans1, patch_target_trans2, patch_segment_target_trans2, \
            end_ctrl_offset_tar_trans2, ctrlpoints_offset_trans2_pred, ctrlpoints_offset_pred

    def get_points_and_raster_image(self, reference_images, reference_strokes, reference_strokes_ctrl, target_images,
                                    centerpoints_pos_ref, centerpoints_pos_tar,
                                    base_window_size, base_window_size_single,
                                    end_ctrl_offset_ref,
                                    end_ctrl_offset_tar, end_offset_tar_pred,
                                    component_centerpoints, component_win_sizes,
                                    target_transform_cursors, target_transform_win_sizes, target_transform_angles,
                                    target_transform_shear_x_angles, target_transform_shear_y_angles,
                                    target_transform1_translate, target_transform1_scaling,
                                    target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y,
                                    fixing_states):
        """
        :param reference_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_strokes: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_strokes_ctrl: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param centerpoints_pos_ref: (N, 1, 2), float32, in [0.0, 1.0]
        :param centerpoints_pos_tar: (N, 1, 2), float32, in [0.0, 1.0]
        :param base_window_size: (N, 1), float32, in [0.0, 1.0]
        :param base_window_size_single: (N, 1), float32, in [0.0, 1.0]
        :param end_ctrl_offset_ref: (N, 1, 8), float32, [-1.0, 1.0]
        :param end_ctrl_offset_tar: (N, 1, 8), float32, [-1.0, 1.0], relative to ref window
        :param end_offset_tar_pred: (N, 1, 4), float32, [-1.0, 1.0], relative to ref window
        :param component_centerpoints: (N, 1, 2), in [0.0, 1.0], relative to image size
        :param component_win_sizes: (N, 1, 2), in image size
        :param target_transform_cursors: (N, 1, 2), in [0.0, 1.0], relative to image size
        :param target_transform_win_sizes: (N, 1, 2), in image size
        :param target_transform_angles: (N, 1), in [-180.0, 180.0]
        :param target_transform_shear_x_angles: (N, 1), in [-90.0, 90.0]
        :param target_transform_shear_y_angles: (N, 1), in [-90.0, 90.0]
        :param target_transform1_translate: (N, 1, 2), [-1.0, 1.0], relative to target trans0 window
        :param target_transform1_scaling: (N, 1, 2), [0.2, 2.0], relative to target trans0 window
        :param target_transform1_rotate: (N, 1), [-180.0, 180.0]
        :param target_transform1_shear_x: (N, 1), [-90.0, 90.0]
        :param target_transform1_shear_y: (N, 1), [-90.0, 90.0]
        :param fixing_states: (N, 1), [0-nonfix, 1-fix]
        :return:
        """
        # cursor position
        cursor_position_loop_ref = centerpoints_pos_ref  # (N, 1, 2), in size [0.0, 1.0]
        cursor_position_loop_tar = centerpoints_pos_tar  # (N, 1, 2), in size [0.0, 1.0]

        curr_window_size = base_window_size.unsqueeze(dim=-1)  # (N, 1, 1), in [0.0, 1.0]
        curr_window_size = torch.mul(curr_window_size, self.image_size)  # (N, 1, 1), in full size
        curr_window_size = torch.mul(curr_window_size, self.hps.window_size_scaling_ref)  # (N, 1, 1), in full size
        curr_window_size = torch.max(curr_window_size, torch.tensor(self.hps.window_size_min).float().cuda())
        curr_window_size = torch.min(curr_window_size, torch.tensor(self.image_size * 1.5).float().cuda())
        curr_window_size = torch.cat([curr_window_size, curr_window_size], dim=-1)  # (N, 1, 2), in full size

        ## reference_images: (N, H, W, 1), [0.0-stroke, 1.0-BG]
        crop_inputs_ref = torch.cat([reference_images, reference_strokes], dim=-1)  # (N, H, W, *)
        cropped_outputs = image_cropping_stn(cursor_position_loop_ref, crop_inputs_ref, self.image_size, self.hps.raster_size, curr_window_size)

        curr_patch_image_ref = cropped_outputs[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_stroke_ref = cropped_outputs[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]

        curr_patch_image_ref_out = torch.squeeze(curr_patch_image_ref, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_stroke_ref_out = torch.squeeze(curr_patch_stroke_ref, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        ## target_images: (N, H, W, 1), [0.0-stroke, 1.0-BG]
        crop_inputs_tar = target_images  # (N, H, W, *)

        # crop without transform
        cropped_outputs = image_cropping_stn(cursor_position_loop_tar, crop_inputs_tar, self.image_size, self.hps.raster_size, curr_window_size)
        curr_patch_image_tar_ori = cropped_outputs[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_image_tar_ori_out = torch.squeeze(curr_patch_image_tar_ori, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        # crop with transform1
        curr_window_size_local_trans = base_window_size.unsqueeze(dim=-1)  # (N, 1, 1), in [0.0, 1.0]
        curr_window_size_local_trans = torch.mul(curr_window_size_local_trans, self.image_size)  # (N, 1, 1), in full size
        curr_window_size_local_trans = torch.mul(curr_window_size_local_trans, self.hps.window_size_scaling_ref_comp_local)  # (N, 1, 1), in full size
        curr_window_size_local_trans = torch.max(curr_window_size_local_trans, torch.tensor(self.hps.window_size_min).float().cuda())
        curr_window_size_local_trans = torch.min(curr_window_size_local_trans, torch.tensor(self.image_size * 1.5).float().cuda())
        curr_window_size_local_trans = torch.cat([curr_window_size_local_trans, curr_window_size_local_trans], dim=-1)  # (N, 1, 2), in full size

        additional_offset = (cursor_position_loop_ref - component_centerpoints) * self.image_size / (component_win_sizes / 2.0)  # (N, 1, 2), [-1, 1]
        additional_scale_local_trans = curr_window_size_local_trans / component_win_sizes  # (N, 1, 2), [0, 1+]
        additional_scale_stroke = curr_window_size / curr_window_size_local_trans  # (N, 1, 2), [0, 1+]
        cropped_outputs = image_cropping_stn_multi(target_transform_cursors, crop_inputs_tar, self.image_size, self.hps.raster_size,
                                                   target_transform_win_sizes,
                                                   rotation_angle=target_transform_angles,
                                                   shear_x_angle=target_transform_shear_x_angles,
                                                   shear_y_angle=target_transform_shear_y_angles,
                                                   additional_transform=True,
                                                   addi_offset=additional_offset, addi_scale=additional_scale_local_trans,
                                                   additional_transform3=True,
                                                   addi_offset3=target_transform1_translate,
                                                   addi_scale3=target_transform1_scaling,
                                                   addi_rotate3=target_transform1_rotate,
                                                   addi_shear_x3=target_transform1_shear_x,
                                                   addi_shear_y3=target_transform1_shear_y,
                                                   additional_transform4=True,
                                                   addi_scale4=additional_scale_stroke
                                                   )
        curr_patch_image_tar_trans1_temp = cropped_outputs[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_image_tar_trans1_out = torch.squeeze(curr_patch_image_tar_trans1_temp, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

        # crop with transform2: based on target stroke
        target_transform_cursors0 = target_transform_cursors * float(self.image_size) + additional_offset * (component_win_sizes / 2.0)  # (N, 1, 2), in image size
        target_transform_win_sizes0 = target_transform_win_sizes * additional_scale_local_trans  # (N, 1, 2), in image size
        target_transform_cursors1 = target_transform_cursors0 + (target_transform1_translate * target_transform_win_sizes0 / 2.0)  # (N, 1, 2), in image size
        target_transform_win_sizes1 = target_transform_win_sizes0 * target_transform1_scaling  # (N, 1, 2), in image size
        target_transform_cursors1_addi = target_transform_cursors1  # (N, 1, 2), in image size
        target_transform_win_sizes1_addi = target_transform_win_sizes1 * additional_scale_stroke  # (N, 1, 2), in image size

        ## Convert target endpoints and control points (offset) in normal coordinate to transformed coordinate
        end_ctrl_offset_tar_input = torch.cat([end_offset_tar_pred[:, :, 0:2],
                                               end_ctrl_offset_tar[:, :, 2:6],
                                               end_offset_tar_pred[:, :, 2:4]], dim=-1)  # (N, 1, 8), float32, [-1.0, 1.0], relative to ref window
        # ori -> trans0
        end_ctrl_offset_tar_trans0 = spatial_transform_stroke_with_additional(
            end_ctrl_offset_tar_input, curr_window_size, cursor_position_loop_tar * float(self.image_size),
            target_transform_cursors * float(self.image_size), target_transform_win_sizes,
            target_transform_angles, target_transform_shear_x_angles, target_transform_shear_y_angles,
            additional_offset, additional_scale_local_trans
        )  # (N, 1, 8), [-1.0, 1.0]

        # trans0 -> trans1
        end_ctrl_offset_tar_trans1 = spatial_transform_stroke_with_additional(
            end_ctrl_offset_tar_trans0, target_transform_win_sizes0, target_transform_cursors0,
            target_transform_cursors1, target_transform_win_sizes1,
            target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y,
            torch.zeros_like(additional_offset), additional_scale_stroke
        )  # (N, 1, 8), [-1.0, 1.0], relative to reference stroke window
        endpoint0_tar_offset_trans1, _, _, endpoint3_tar_offset_trans1 = torch.split(
            end_ctrl_offset_tar_trans1, 2, dim=-1)  # each of (N, 1, 2), [-1.0, 1.0], relative to reference stroke window

        # trans1 -> trans2
        additional_offset_to_target = endpoint0_tar_offset_trans1  # (N, 1, 2), [-1.0, 1.0]
        additional_scale_to_target = torch.max(torch.abs(endpoint0_tar_offset_trans1 - endpoint3_tar_offset_trans1), dim=-1, keepdim=True)[0] * self.hps.window_size_scaling_ref  # (N, 1, 1)
        additional_scale_to_target = torch.cat([additional_scale_to_target, additional_scale_to_target], dim=-1)  # (N, 1, 2)

        target_transform_cursors2 = target_transform_cursors1_addi + (additional_offset_to_target * target_transform_win_sizes1_addi / 2.0)  # (N, 1, 2), in image size
        target_transform_win_sizes2 = target_transform_win_sizes1_addi * additional_scale_to_target  # (N, 1, 2), in image size

        end_ctrl_offset_tar_trans2 = spatial_transform_stroke_with_additional(
            end_ctrl_offset_tar_trans1, target_transform_win_sizes1_addi, target_transform_cursors1_addi,
            target_transform_cursors2, target_transform_win_sizes2,
            torch.zeros_like(target_transform1_rotate), torch.zeros_like(target_transform1_shear_x), torch.zeros_like(target_transform1_shear_y),
            torch.zeros_like(additional_offset), torch.ones_like(additional_scale_stroke)
        )  # (N, 1, 8), [-1.0, 1.0], relative to target stroke window

        # crop target based on target stroke window
        cropped_outputs = image_cropping_stn_multi(target_transform_cursors, crop_inputs_tar, self.image_size,
                                                   self.hps.raster_size,
                                                   target_transform_win_sizes,
                                                   rotation_angle=target_transform_angles,
                                                   shear_x_angle=target_transform_shear_x_angles,
                                                   shear_y_angle=target_transform_shear_y_angles,
                                                   additional_transform=True,
                                                   addi_offset=additional_offset,
                                                   addi_scale=additional_scale_local_trans,
                                                   additional_transform3=True,
                                                   addi_offset3=target_transform1_translate,
                                                   addi_scale3=target_transform1_scaling,
                                                   addi_rotate3=target_transform1_rotate,
                                                   addi_shear_x3=target_transform1_shear_x,
                                                   addi_shear_y3=target_transform1_shear_y,
                                                   additional_transform4=True,
                                                   addi_scale4=additional_scale_stroke,
                                                   additional_transform5=True,
                                                   addi_offset5=additional_offset_to_target,
                                                   addi_scale5=additional_scale_to_target,
                                                   )
        curr_patch_image_tar_trans2 = cropped_outputs[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_image_tar_trans2_out = torch.squeeze(curr_patch_image_tar_trans2, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        # generate segment images on-the-fly
        curr_patch_segment_tar_trans2 = gen_segment_imgs_on_the_fly(end_ctrl_offset_tar_trans2, self.hps.raster_size)  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_segment_tar_trans2_out = torch.squeeze(curr_patch_segment_tar_trans2, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        curr_patch_image_tar_trans2 = normalize_image_m1to1(curr_patch_image_tar_trans2)
        curr_patch_segment_tar_trans2 = normalize_image_m1to1(curr_patch_segment_tar_trans2)
        # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

        # crop reference images based on the additional_scale_to_target
        curr_window_size_single = base_window_size_single.unsqueeze(dim=-1)  # (N, 1, 1), in [0.0, 1.0]
        curr_window_size_single = torch.mul(curr_window_size_single, self.image_size)  # (N, 1, 1), in full size
        curr_window_size_single = torch.mul(curr_window_size_single, self.hps.window_size_scaling_ref)  # (N, 1, 1), in full size
        curr_window_size_single = torch.cat([curr_window_size_single, curr_window_size_single], dim=-1)  # (N, 1, 2), in full size

        crop_inputs_ref = torch.cat([reference_images, reference_strokes_ctrl], dim=-1)  # (N, H, W, *)
        cropped_outputs = image_cropping_stn(cursor_position_loop_ref, crop_inputs_ref, self.image_size, self.hps.raster_size,
                                             curr_window_size_single)

        curr_patch_image_ref2 = cropped_outputs[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_stroke_ref2 = cropped_outputs[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]

        curr_patch_image_ref_out2 = torch.squeeze(curr_patch_image_ref2, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_stroke_ref_out2 = torch.squeeze(curr_patch_stroke_ref2, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        curr_patch_image_ref2 = normalize_image_m1to1(curr_patch_image_ref2)
        curr_patch_stroke_ref2 = normalize_image_m1to1(curr_patch_stroke_ref2)
        # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

        encoded_z = self.build_encoder_ctrlpoint(curr_patch_image_ref2, curr_patch_stroke_ref2,
                                                 curr_patch_image_tar_trans2, curr_patch_segment_tar_trans2)  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        ctrlpoints_offset_trans2_pred = self.build_decoder_ctrlpoint(encoded_z)
        # ctrlpoints_offset_trans2_pred: (N, 4), [-1.0, 1.0], relative to target stroke window

        ## Fixing endpoints of fully occluded curves
        ctrlpoints_offset_ref = end_ctrl_offset_ref[:, 0, 2:6]  # (N, 4), [-1.0, 1.0]
        ctrlpoints_offset_trans2_pred = ctrlpoints_offset_trans2_pred * (1.0 - fixing_states) + ctrlpoints_offset_ref * fixing_states

        end_ctrl_offset_tar_trans2_pred = torch.cat([end_ctrl_offset_tar_trans2[:, :, 0:2],
                                                     ctrlpoints_offset_trans2_pred.unsqueeze(dim=1),
                                                     end_ctrl_offset_tar_trans2[:, :, 6:8]], dim=-1)
        # (N, 1, 8), [-1.0, 1.0], relative to target stroke window

        ## inverse transform 2 -> 1
        end_ctrl_offset_tar_trans1_pred = spatial_transform_reverse_stroke_with_additional(
            end_ctrl_offset_tar_trans2_pred, target_transform_win_sizes1_addi, target_transform_cursors1_addi,
            target_transform_cursors2, target_transform_win_sizes2,
            torch.zeros_like(target_transform1_rotate), torch.zeros_like(target_transform1_shear_x),
            torch.zeros_like(target_transform1_shear_y),
            torch.zeros_like(additional_offset), torch.ones_like(additional_scale_stroke)
        )  # (N, 1, 8), [-1.0, 1.0], relative to reference stroke window

        ## inverse transform 1 -> 0
        end_ctrl_offset_tar_trans0_pred = spatial_transform_reverse_stroke_with_additional(
            end_ctrl_offset_tar_trans1_pred, target_transform_win_sizes0, target_transform_cursors0,
            target_transform_cursors1, target_transform_win_sizes1,
            target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y,
            torch.zeros_like(additional_offset), additional_scale_stroke
        )  # (N, 1, 8), [-1.0, 1.0]

        ## inverse transform 0 -> ori
        end_ctrl_offset_tar_pred = spatial_transform_reverse_stroke_with_additional(
            end_ctrl_offset_tar_trans0_pred, curr_window_size, cursor_position_loop_tar * float(self.image_size),
            target_transform_cursors * float(self.image_size), target_transform_win_sizes,
            target_transform_angles, target_transform_shear_x_angles, target_transform_shear_y_angles,
            additional_offset, additional_scale_local_trans
        )  # (N, 1, 8), [-1.0, 1.0], relative to ref window

        ctrlpoints_offset_pred = end_ctrl_offset_tar_pred.squeeze(dim=1)[:, 2:6]  # (N, 4), [-1.0, 1.0], relative to ref window

        return curr_patch_image_ref_out, curr_patch_stroke_ref_out, \
               curr_patch_image_ref_out2, curr_patch_stroke_ref_out2, \
               curr_patch_image_tar_ori_out, curr_patch_image_tar_trans1_out, \
               curr_patch_image_tar_trans2_out, curr_patch_segment_tar_trans2_out, \
               end_ctrl_offset_tar_trans2, ctrlpoints_offset_trans2_pred, ctrlpoints_offset_pred

    def build_encoder_ctrlpoint(self, patch_image_ref, patch_stroke_ref, patch_image_tar, patch_segment_tar):
        """
        :param patch_image_ref & patch_stroke_ref & patch_image_tar: (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        :return:
        """
        # transform to nchw
        patch_images_ref = patch_image_ref  # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        patch_images_ref = patch_images_ref.permute(0, 3, 1, 2)  # (N, 1, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
        patch_strokes_ref = patch_stroke_ref  # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        patch_strokes_ref = patch_strokes_ref.permute(0, 3, 1, 2)  # (N, 1, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
        patch_images_tar = patch_image_tar  # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        patch_images_tar = patch_images_tar.permute(0, 3, 1, 2)  # (N, 1, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
        patch_segments_tar = patch_segment_tar  # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        patch_segments_tar = patch_segments_tar.permute(0, 3, 1, 2)  # (N, 1, raster_size, raster_size), [-1.0-stroke, 1.0-BG]

        if self.hps.enc_model_ctrlpoint == 'combined':
            batch_input = torch.cat([patch_images_ref, patch_strokes_ref, patch_images_tar, patch_segments_tar], dim=1)  # (N, 2, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
            if self.hps.add_coordconv:
                batch_input = add_coords(batch_input, self.coordconv_input)  # (N, in_dim + 2, in_H, in_W)
            output = self.encoder_ctrlpoint(batch_input)  # (N, z_size)
        elif self.hps.enc_model_ctrlpoint == 'separated':
            batch_input_ref = torch.cat([patch_images_ref, patch_strokes_ref], dim=1)  # (N, 2, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
            batch_input_tar = torch.cat([patch_images_tar, patch_segments_tar], dim=1)  # (N, 2, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
            if self.hps.add_coordconv:
                batch_input_ref = add_coords(batch_input_ref, self.coordconv_input)  # (N, in_dim + 2, in_H, in_W)
                batch_input_tar = add_coords(batch_input_tar, self.coordconv_input)  # (N, in_dim + 2, in_H, in_W)
            output = self.encoder_ctrlpoint(batch_input_ref, batch_input_tar)  # (N, z_size)
        else:
            raise Exception('Unknown enc_model_ctrlpoint:', self.hps.enc_model_ctrlpoint)

        return output

    def build_decoder_ctrlpoint(self, dec_input):
        """
        :param dec_input: (N, in_dim)
        :return:
        """
        output = self.decoder_ctrlpoint(dec_input)
        output = torch.tanh(output)  # (N, 4), [-1.0, 1.0]
        return output


class FullModel(object):
    def __init__(self, hps, train_set, valid_set,
                 log_dir, snapshot_dir, log_img_dir):
        self.hps = hps
        self.train_set = train_set
        self.valid_set = valid_set
        self.log_dir = log_dir
        self.snapshot_dir = snapshot_dir
        self.log_img_dir = log_img_dir

        self.controlpoint_model = Controlpoint_Model(hps)

        params_list = []

        if self.hps.multi_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.controlpoint_model = nn.DataParallel(self.controlpoint_model)
            params_list.append({'params': self.controlpoint_model.module.parameters()})
        else:
            params_list.append({'params': self.controlpoint_model.parameters()})
        self.optimizer = optim.Adam(params_list, lr=hps.learning_rate)

        self.start_step = 0
        self.use_cuda = torch.cuda.is_available()

    def evaluate(self, load_trained_weights=False, occluded_only=False, test_max_batch_size=180,
                 endpoint_distance_threshold=7, stroke_pixels_iou_threshold=0.5, stroke_pixels_iou_threshold_occ=0.5):
        print('-' * 100)
        print('Evaluation begins ...')

        if load_trained_weights:
            print('-' * 100)
            trained_controlpoint_model_path = os.path.join(self.snapshot_dir, "sketch_ctrlpoint_" + str(self.hps.num_steps) + ".pkl")
            if self.hps.multi_gpu:
                load_weights(trained_controlpoint_model_path, self.controlpoint_model.module)
            else:
                load_weights(trained_controlpoint_model_path, self.controlpoint_model)
            print('-' * 100)

            if self.use_cuda:
                self.controlpoint_model = self.controlpoint_model.cuda()

        self.controlpoint_model.eval()

        self.valid_set.batch_size = 1
        batch_num = self.valid_set.example_num // self.valid_set.batch_size
        print('batch_num:', batch_num)

        stroke_incorrect_state_set = []
        stroke_iou_set = []
        stroke_epe_set = []
        total_stroke_num = 0

        with (torch.no_grad()):
            for batch_i in range(batch_num):
                print('# batch_i', batch_i)
                batch_data = self.valid_set.get_batch(self.use_cuda, batch_idx=batch_i, all_example=True, occluded_only=occluded_only)
                if batch_data is None:
                    continue

                reference_images, reference_strokes, reference_strokes_ctrl, target_images, \
                    reference_centerpoints, target_centerpoints, \
                    reference_end_ctrl_offset, reference_end_ctrl_offset_single, target_end_ctrl_offset_gt, target_end_offset_pred, target_occluded_masks, \
                    base_window_size, base_window_size_single, _, _, _, \
                    component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
                    target_transform_shear_x_angles, target_transform_shear_y_angles,\
                    target_transform1_translate, target_transform1_scaling, \
                    target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y, fixing_states = batch_data
                # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_strokes_ctrl: (N, H, W, 1), [0-stroke, 1-BG]
                # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
                # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
                # reference_end_ctrl_offset / target_end_ctrl_offset_gt: (N, 1, 8), in [-1.0, 1.0]
                # reference_end_ctrl_offset_single: (N, 1, 8), in [-1.0, 1.0]
                # target_end_offset_pred: (N, 1, 4), in [-1.0, 1.0]
                # target_occluded_masks: (N, H, W, 1), [0-occluded, 1-visible]
                # base_window_size / base_window_size_single: (N, 1), in [0.0, 1.0]

                # component_centerpoints: (N, 1, 2), in [0.0, 1.0], relative to image size
                # component_win_sizes: (N, 1, 2), in image size
                # target_transform_cursors: (N, 1, 2), in [0.0, 1.0], relative to image size
                # target_transform_win_sizes: (N, 1, 2), in image size
                # target_transform_angles: (N, 1), in [-180.0, 180.0]
                # target_transform_shear_x_angles: (N, 1), in [-90.0, 90.0]
                # target_transform_shear_y_angles: (N, 1), in [-90.0, 90.0]

                # target_transform1_translate:(N, 1, 2), [-1.0, 1.0], relative to target trans0 window
                # target_transform1_scaling: (N, 1, 2), [0.2, 2.0], relative to target trans0 window
                # target_transform1_rotate: (N, 1), [-180.0, 180.0]
                # target_transform1_shear_x / target_transform1_shear_y: (N, 1), [-90.0, 90.0]

                # fixing_states: (N, 1), [0-nonfix, 1-fix]

                image_size = reference_images.size()[1]

                stroke_num = reference_images.size()[0]
                total_stroke_num += stroke_num

                chunks = stroke_num // test_max_batch_size + 1

                reference_images_chunks = torch.chunk(reference_images, chunks, dim=0)
                reference_strokes_chunks = torch.chunk(reference_strokes, chunks, dim=0)
                reference_strokes_ctrl_chunks = torch.chunk(reference_strokes_ctrl, chunks, dim=0)
                target_images_chunks = torch.chunk(target_images, chunks, dim=0)
                reference_centerpoints_chunks = torch.chunk(reference_centerpoints, chunks, dim=0)
                reference_end_ctrl_offset_single_chunks = torch.chunk(reference_end_ctrl_offset_single, chunks, dim=0)
                target_centerpoints_chunks = torch.chunk(target_centerpoints, chunks, dim=0)
                target_end_ctrl_offset_gt_chunks = torch.chunk(target_end_ctrl_offset_gt, chunks, dim=0)
                target_end_offset_pred_chunks = torch.chunk(target_end_offset_pred, chunks, dim=0)
                base_window_size_chunks = torch.chunk(base_window_size, chunks, dim=0)
                base_window_size_single_chunks = torch.chunk(base_window_size_single, chunks, dim=0)
                component_centerpoints_chunks = torch.chunk(component_centerpoints, chunks, dim=0)
                component_win_sizes_chunks = torch.chunk(component_win_sizes, chunks, dim=0)
                target_transform_cursors_chunks = torch.chunk(target_transform_cursors, chunks, dim=0)
                target_transform_win_sizes_chunks = torch.chunk(target_transform_win_sizes, chunks, dim=0)
                target_transform_angles_chunks = torch.chunk(target_transform_angles, chunks, dim=0)
                target_transform_shear_x_angles_chunks = torch.chunk(target_transform_shear_x_angles, chunks, dim=0)
                target_transform_shear_y_angles_chunks = torch.chunk(target_transform_shear_y_angles, chunks, dim=0)
                target_transform1_translate_chunks = torch.chunk(target_transform1_translate, chunks, dim=0)
                target_transform1_scaling_chunks = torch.chunk(target_transform1_scaling, chunks, dim=0)
                target_transform1_rotate_chunks = torch.chunk(target_transform1_rotate, chunks, dim=0)
                target_transform1_shear_x_chunks = torch.chunk(target_transform1_shear_x, chunks, dim=0)
                target_transform1_shear_y_chunks = torch.chunk(target_transform1_shear_y, chunks, dim=0)
                fixing_states_chunks = torch.chunk(fixing_states, chunks, dim=0)

                ctrlpoints_offset_pred_np = []

                for chunk_i in range(chunks):
                    _, _, _, _, _, _, _, _, _, _, ctrlpoints_offset_pred_ch = \
                        self.controlpoint_model(reference_images=reference_images_chunks[chunk_i],
                                                reference_strokes=reference_strokes_chunks[chunk_i],
                                                reference_strokes_ctrl=reference_strokes_ctrl_chunks[chunk_i],
                                                target_images=target_images_chunks[chunk_i],
                                                centerpoints_pos_ref=reference_centerpoints_chunks[chunk_i],
                                                centerpoints_pos_tar=target_centerpoints_chunks[chunk_i],
                                                end_ctrl_offset_ref=reference_end_ctrl_offset_single_chunks[chunk_i],
                                                end_ctrl_offset_tar=target_end_ctrl_offset_gt_chunks[chunk_i],
                                                end_offset_tar_pred=target_end_offset_pred_chunks[chunk_i],
                                                base_window_size=base_window_size_chunks[chunk_i],
                                                base_window_size_single=base_window_size_single_chunks[chunk_i],
                                                image_size=image_size,
                                                component_centerpoints=component_centerpoints_chunks[chunk_i],
                                                component_win_sizes=component_win_sizes_chunks[chunk_i],
                                                target_transform_cursors=target_transform_cursors_chunks[chunk_i],
                                                target_transform_win_sizes=target_transform_win_sizes_chunks[chunk_i],
                                                target_transform_angles=target_transform_angles_chunks[chunk_i],
                                                target_transform_shear_x_angles=target_transform_shear_x_angles_chunks[chunk_i],
                                                target_transform_shear_y_angles=target_transform_shear_y_angles_chunks[chunk_i],
                                                target_transform1_translate=target_transform1_translate_chunks[chunk_i],
                                                target_transform1_scaling=target_transform1_scaling_chunks[chunk_i],
                                                target_transform1_rotate=target_transform1_rotate_chunks[chunk_i],
                                                target_transform1_shear_x=target_transform1_shear_x_chunks[chunk_i],
                                                target_transform1_shear_y=target_transform1_shear_y_chunks[chunk_i],
                                                fixing_states=fixing_states_chunks[chunk_i],
                                                )
                    # ctrlpoints_offset_pred: (N, 4), [-1.0, 1.0], relative to ref window

                    ctrlpoints_offset_pred_np_ch = ctrlpoints_offset_pred_ch.cpu().data.numpy()  # (N, 4), [-1.0, 1.0]
                    ctrlpoints_offset_pred_np.append(ctrlpoints_offset_pred_np_ch)

                ctrlpoints_offset_pred_np = np.concatenate(ctrlpoints_offset_pred_np, axis=0)  # (N, 4), [-1.0, 1.0]
                endpoints_offset_pred_np = target_end_offset_pred.squeeze(dim=1).cpu().data.numpy()  # (N, 4) [-1.0, 1.0]

                target_end_ctrl_offset_gt_np = target_end_ctrl_offset_gt.squeeze(dim=1).cpu().data.numpy()  # (N, 8), in [-1.0, 1.0]
                endpoints_offset_gt_np = np.concatenate([target_end_ctrl_offset_gt_np[:, 0:2],
                                                         target_end_ctrl_offset_gt_np[:, 6:8]], axis=-1)  # (N, 4), [-1.0, 1.0]
                ctrlpoints_offset_gt_np = target_end_ctrl_offset_gt_np[:, 2:6]  # (N, 4), [-1.0, 1.0]

                ## convert to global coordinate
                base_window_size = torch.squeeze(base_window_size, dim=-1)  # (N), in [0.0, 1.0]
                base_window_size_np = base_window_size.cpu().data.numpy()  # (N), in [0.0, 1.0]
                base_window_size_np_scaled = base_window_size_np * image_size * self.hps.window_size_scaling_ref  # (N), in full size
                base_window_size_np_scaled = np.clip(base_window_size_np_scaled, self.hps.window_size_min, image_size * 1.5)
                base_window_size_np_scaled = np.expand_dims(base_window_size_np_scaled, axis=-1)
                base_window_size_np_scaled = np.tile(base_window_size_np_scaled, (1, 4))  # (N, 4)

                centerpoints_pos_tar = torch.squeeze(target_centerpoints, dim=1)  # (N, 2), in [0.0, 1.0], relative to full size
                centerpoints_pos_tar_np = centerpoints_pos_tar.cpu().data.numpy()  # (N, 2), in [0.0, 1.0], relative to full size
                centerpoints_pos_tar_np = np.concatenate([centerpoints_pos_tar_np, centerpoints_pos_tar_np], axis=-1)  # (N, 4)

                ctrlpoints_pred_rel = ctrlpoints_offset_pred_np  # (N, 4), [-1.0, 1.0] relative to window
                ctrlpoints_pred_offset_global = ctrlpoints_pred_rel * (base_window_size_np_scaled / 2.0)
                ctrlpoints_pred_global = ctrlpoints_pred_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

                ctrlpoints_gt_rel = ctrlpoints_offset_gt_np  # (N, 4), [-1.0, 1.0] relative to window
                ctrlpoints_gt_offset_global = ctrlpoints_gt_rel * (base_window_size_np_scaled / 2.0)
                ctrlpoints_gt_global = ctrlpoints_gt_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

                endpoints_pred_rel = endpoints_offset_pred_np  # (N, 4), [-1.0, 1.0] relative to window
                endpoints_pred_offset_global = endpoints_pred_rel * (base_window_size_np_scaled / 2.0)
                endpoints_pred_global = endpoints_pred_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

                endpoints_gt_rel = endpoints_offset_gt_np  # (N, 4), [-1.0, 1.0] relative to window
                endpoints_gt_offset_global = endpoints_gt_rel * (base_window_size_np_scaled / 2.0)
                endpoints_gt_global = endpoints_gt_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

                stroke_points_gt = np.concatenate([endpoints_gt_global[:, 0:2], ctrlpoints_gt_global, endpoints_gt_global[:, 2:4]], axis=1)  # (N, 8), in full size
                stroke_points_pred = np.concatenate([endpoints_pred_global[:, 0:2], ctrlpoints_pred_global, endpoints_pred_global[:, 2:4]], axis=1)  # (N, 8), in full size

                if not occluded_only:
                    endpoint0_error_global = np.sqrt(np.sum(np.power(endpoints_pred_global[:, 0:2] - endpoints_gt_global[:, 0:2], 2), axis=-1))  # (N)
                    endpoint1_error_global = np.sqrt(np.sum(np.power(endpoints_pred_global[:, 2:4] - endpoints_gt_global[:, 2:4], 2), axis=-1))  # (N)
                    endpoint_error_global = np.stack([endpoint0_error_global, endpoint1_error_global], axis=-1)
                    endpoint_error_global_mean = np.mean(endpoint_error_global, axis=-1)  # (N)
                    endpoint_error_global = np.max(endpoint_error_global, axis=-1)  # (N)

                    rendered_stroke_pixel_ious = cal_stroke_pixel_iou(stroke_points_gt, stroke_points_pred, image_size)  # (N)

                    # First, endpoints should be accurate
                    incorrect_endpoint_states = endpoint_error_global > endpoint_distance_threshold
                    # Then,
                    incorrect_pixel_states = rendered_stroke_pixel_ious < stroke_pixels_iou_threshold

                    incorrect_stroke_states = np.logical_or(incorrect_endpoint_states, incorrect_pixel_states)
                else:
                    target_occluded_masks_np = target_occluded_masks.cpu().data.numpy()  # (N, H, W, 1), [0-occluded, 1-visible]
                    rendered_stroke_pixel_ious = cal_stroke_pixel_iou(stroke_points_gt, stroke_points_pred, image_size,
                                                                      occluded_masks=target_occluded_masks_np)  # (N)
                    incorrect_stroke_states = rendered_stroke_pixel_ious < stroke_pixels_iou_threshold_occ
                    endpoint_error_global_mean = np.zeros_like(rendered_stroke_pixel_ious)

                stroke_epe_set += endpoint_error_global_mean.tolist()
                stroke_iou_set += rendered_stroke_pixel_ious.tolist()
                stroke_incorrect_state_set += incorrect_stroke_states.tolist()

        print('total_stroke_num', total_stroke_num)
        tf.get_logger().info('total_stroke_num: ' + str(total_stroke_num))
        assert total_stroke_num == len(stroke_incorrect_state_set) == len(stroke_iou_set), (total_stroke_num, len(stroke_incorrect_state_set), len(stroke_iou_set))
        accurate_rate = 1.0 - np.mean(stroke_incorrect_state_set)
        iou_avg = np.mean(stroke_iou_set)
        epe_avg = np.mean(stroke_epe_set)
        print('Accuracy_rate (ACC): ', accurate_rate * 100.0, '%')
        print('Average IoU: ', iou_avg * 100.0, '%')
        print('Average EPE: ', epe_avg)
        print('snapshot_dir:', self.snapshot_dir)
        tf.get_logger().info('Accuracy_rate (ACC): ' + str(accurate_rate * 100.0) + '%')
        tf.get_logger().info('Average IoU: ' + str(iou_avg * 100.0) + '%')
        tf.get_logger().info('Average EPE: ' + str(epe_avg))
        tf.get_logger().info('snapshot_dir: ' + self.snapshot_dir)

    def inference_full(self, save_root, show_data='selected', occluded_only=False, test_max_batch_size=180,
                       endpoint_distance_threshold=7, stroke_pixels_iou_threshold=0.5, stroke_pixels_iou_threshold_occ=0.5):
        # print('-' * 100)
        print('Inference begins ...')

        trained_controlpoint_model_path = os.path.join(self.snapshot_dir, "sketch_ctrlpoint_" + str(self.hps.num_steps) + ".pkl")
        if self.hps.multi_gpu:
            load_weights(trained_controlpoint_model_path, self.controlpoint_model.module)
        else:
            load_weights(trained_controlpoint_model_path, self.controlpoint_model)

        if self.use_cuda:
            self.controlpoint_model = self.controlpoint_model.cuda()

        self.controlpoint_model.eval()

        if show_data == 'all':
            show_num = self.valid_set.example_num
            batch_idx_offsets = [0]
            
            vis_gt = True
            vis_incorrect = True
            save_parameters = False
        elif show_data == 'selected':
            show_num = 100
            batch_idx_offsets = [0, 737]
            vis_gt = True
            vis_incorrect = True
            save_parameters = False
        else:
            raise Exception('Unknown show_data:', show_data)

        batch_num = self.valid_set.example_num // self.valid_set.batch_size
        print('batch_num:', batch_num)

        if not occluded_only:
            save_base = os.path.join(save_root, 'non_occluded')
        else:
            save_base = os.path.join(save_root, 'occluded')
        os.makedirs(save_base, exist_ok=True)

        with torch.no_grad():
            for batch_idx_offset in batch_idx_offsets:
                show_i = 0
                for batch_i in range(batch_num):
                    print('# batch_i', batch_i)

                    batch_data = self.valid_set.get_batch(self.use_cuda, batch_idx=batch_i, all_example=True, batch_idx_offset=batch_idx_offset, occluded_only=occluded_only)
                    if batch_data is None:
                        continue

                    reference_images, reference_strokes, reference_strokes_ctrl, target_images, \
                        reference_centerpoints, target_centerpoints, \
                        reference_end_ctrl_offset, reference_end_ctrl_offset_single, target_end_ctrl_offset_gt, target_end_offset_pred, target_occluded_masks, \
                        base_window_size, base_window_size_single, image_ids, stroke_ids, num_components, \
                        component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
                        target_transform_shear_x_angles, target_transform_shear_y_angles,\
                        target_transform1_translate, target_transform1_scaling, \
                        target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y, fixing_states = batch_data
                    # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                    # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
                    # reference_strokes_ctrl: (N, H, W, 1), [0-stroke, 1-BG]
                    # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                    # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
                    # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
                    # reference_end_ctrl_offset / target_end_ctrl_offset_gt: (N, 1, 8), in [-1.0, 1.0]
                    # reference_end_ctrl_offset_single: (N, 1, 8), in [-1.0, 1.0]
                    # target_end_offset_pred: (N, 1, 4), in [-1.0, 1.0]
                    # target_occluded_masks: (N, H, W, 1), [0-occluded, 1-visible]
                    # base_window_size / base_window_size_single: (N, 1), in [0.0, 1.0]

                    # component_centerpoints: (N, 1, 2), in [0.0, 1.0], relative to image size
                    # component_win_sizes: (N, 1, 2), in image size
                    # target_transform_cursors: (N, 1, 2), in [0.0, 1.0], relative to image size
                    # target_transform_win_sizes: (N, 1, 2), in image size
                    # target_transform_angles: (N, 1), in [-180.0, 180.0]
                    # target_transform_shear_x_angles: (N, 1), in [-90.0, 90.0]
                    # target_transform_shear_y_angles: (N, 1), in [-90.0, 90.0]

                    # target_transform1_translate:(N, 1, 2), [-1.0, 1.0], relative to target trans0 window
                    # target_transform1_scaling: (N, 1, 2), [0.2, 2.0], relative to target trans0 window
                    # target_transform1_rotate: (N, 1), [-180.0, 180.0]
                    # target_transform1_shear_x / target_transform1_shear_y: (N, 1), [-90.0, 90.0]

                    # fixing_states: (N, 1), [0-nonfix, 1-fix]

                    image_size = reference_images.size()[1]
                    if target_end_ctrl_offset_gt is None:
                        assert target_occluded_masks is None
                        target_end_ctrl_offset_gt = reference_end_ctrl_offset

                    assert len(image_ids) == 1
                    img_index = image_ids[0]
                    print(' >> img_index', img_index)

                    stroke_num = reference_images.size()[0]
                    chunks = stroke_num // test_max_batch_size + 1

                    reference_images_chunks = torch.chunk(reference_images, chunks, dim=0)
                    reference_strokes_chunks = torch.chunk(reference_strokes, chunks, dim=0)
                    reference_strokes_ctrl_chunks = torch.chunk(reference_strokes_ctrl, chunks, dim=0)
                    target_images_chunks = torch.chunk(target_images, chunks, dim=0)
                    reference_centerpoints_chunks = torch.chunk(reference_centerpoints, chunks, dim=0)
                    reference_end_ctrl_offset_single_chunks = torch.chunk(reference_end_ctrl_offset_single, chunks, dim=0)
                    target_centerpoints_chunks = torch.chunk(target_centerpoints, chunks, dim=0)
                    target_end_ctrl_offset_gt_chunks = torch.chunk(target_end_ctrl_offset_gt, chunks, dim=0)
                    target_end_offset_pred_chunks = torch.chunk(target_end_offset_pred, chunks, dim=0)
                    base_window_size_chunks = torch.chunk(base_window_size, chunks, dim=0)
                    base_window_size_single_chunks = torch.chunk(base_window_size_single, chunks, dim=0)
                    component_centerpoints_chunks = torch.chunk(component_centerpoints, chunks, dim=0)
                    component_win_sizes_chunks = torch.chunk(component_win_sizes, chunks, dim=0)
                    target_transform_cursors_chunks = torch.chunk(target_transform_cursors, chunks, dim=0)
                    target_transform_win_sizes_chunks = torch.chunk(target_transform_win_sizes, chunks, dim=0)
                    target_transform_angles_chunks = torch.chunk(target_transform_angles, chunks, dim=0)
                    target_transform_shear_x_angles_chunks = torch.chunk(target_transform_shear_x_angles, chunks, dim=0)
                    target_transform_shear_y_angles_chunks = torch.chunk(target_transform_shear_y_angles, chunks, dim=0)
                    target_transform1_translate_chunks = torch.chunk(target_transform1_translate, chunks, dim=0)
                    target_transform1_scaling_chunks = torch.chunk(target_transform1_scaling, chunks, dim=0)
                    target_transform1_rotate_chunks = torch.chunk(target_transform1_rotate, chunks, dim=0)
                    target_transform1_shear_x_chunks = torch.chunk(target_transform1_shear_x, chunks, dim=0)
                    target_transform1_shear_y_chunks = torch.chunk(target_transform1_shear_y, chunks, dim=0)
                    fixing_states_chunks = torch.chunk(fixing_states, chunks, dim=0)

                    ctrlpoints_offset_pred_np = []

                    for chunk_i in range(chunks):
                        _, _, _, _, _, _, _, _, _, _, ctrlpoints_offset_pred_ch = \
                            self.controlpoint_model(reference_images=reference_images_chunks[chunk_i],
                                                    reference_strokes=reference_strokes_chunks[chunk_i],
                                                    reference_strokes_ctrl=reference_strokes_ctrl_chunks[chunk_i],
                                                    target_images=target_images_chunks[chunk_i],
                                                    centerpoints_pos_ref=reference_centerpoints_chunks[chunk_i],
                                                    centerpoints_pos_tar=target_centerpoints_chunks[chunk_i],
                                                    end_ctrl_offset_ref=reference_end_ctrl_offset_single_chunks[chunk_i],
                                                    end_ctrl_offset_tar=target_end_ctrl_offset_gt_chunks[chunk_i],
                                                    end_offset_tar_pred=target_end_offset_pred_chunks[chunk_i],
                                                    base_window_size=base_window_size_chunks[chunk_i],
                                                    base_window_size_single=base_window_size_single_chunks[chunk_i],
                                                    image_size=image_size,
                                                    component_centerpoints=component_centerpoints_chunks[chunk_i],
                                                    component_win_sizes=component_win_sizes_chunks[chunk_i],
                                                    target_transform_cursors=target_transform_cursors_chunks[chunk_i],
                                                    target_transform_win_sizes=target_transform_win_sizes_chunks[chunk_i],
                                                    target_transform_angles=target_transform_angles_chunks[chunk_i],
                                                    target_transform_shear_x_angles=target_transform_shear_x_angles_chunks[chunk_i],
                                                    target_transform_shear_y_angles=target_transform_shear_y_angles_chunks[chunk_i],
                                                    target_transform1_translate=target_transform1_translate_chunks[chunk_i],
                                                    target_transform1_scaling=target_transform1_scaling_chunks[chunk_i],
                                                    target_transform1_rotate=target_transform1_rotate_chunks[chunk_i],
                                                    target_transform1_shear_x=target_transform1_shear_x_chunks[chunk_i],
                                                    target_transform1_shear_y=target_transform1_shear_y_chunks[chunk_i],
                                                    fixing_states=fixing_states_chunks[chunk_i],
                                                    )
                        # target_end_ctrl_offset_gt_trans2: (N, 1, 8), [-1.0, 1.0], relative to target stroke window
                        # ctrlpoints_offset_trans2_pred: (N, 4), [-1.0, 1.0], relative to target stroke window
                        # ctrlpoints_offset_pred: (N, 4), [-1.0, 1.0], relative to ref window

                        ctrlpoints_offset_pred_np_ch = ctrlpoints_offset_pred_ch.cpu().data.numpy()  # (N, 4), [-1.0, 1.0]
                        ctrlpoints_offset_pred_np.append(ctrlpoints_offset_pred_np_ch)

                    ctrlpoints_offset_pred_np = np.concatenate(ctrlpoints_offset_pred_np, axis=0)  # (N, 4), [-1.0, 1.0]
                    endpoints_offset_pred_np = target_end_offset_pred.squeeze(dim=1).cpu().data.numpy()  # (N, 4) [-1.0, 1.0]

                    target_end_ctrl_offset_gt_np = target_end_ctrl_offset_gt.squeeze(dim=1).cpu().data.numpy()  # (N, 8), in [-1.0, 1.0]
                    endpoints_offset_gt_np = np.concatenate([target_end_ctrl_offset_gt_np[:, 0:2],
                                                             target_end_ctrl_offset_gt_np[:, 6:8]], axis=-1)  # (N, 4), [-1.0, 1.0]
                    ctrlpoints_offset_gt_np = target_end_ctrl_offset_gt_np[:, 2:6]  # (N, 4), [-1.0, 1.0]

                    reference_end_ctrl_offset_np = reference_end_ctrl_offset.squeeze(dim=1).cpu().data.numpy()  # (N, 8), in [-1.0, 1.0]
                    endpoints_offset_ref_np = np.concatenate([reference_end_ctrl_offset_np[:, 0:2],
                                                             reference_end_ctrl_offset_np[:, 6:8]], axis=-1)  # (N, 4), [-1.0, 1.0]
                    ctrlpoints_offset_ref_np = reference_end_ctrl_offset_np[:, 2:6]  # (N, 4), [-1.0, 1.0]

                    ## convert to global coordinate
                    base_window_size = torch.squeeze(base_window_size, dim=-1)  # (N), in [0.0, 1.0]
                    base_window_size_np = base_window_size.cpu().data.numpy()  # (N), in [0.0, 1.0]
                    base_window_size_np_scaled = base_window_size_np * image_size * self.hps.window_size_scaling_ref  # (N), in full size
                    base_window_size_np_scaled = np.clip(base_window_size_np_scaled, self.hps.window_size_min, image_size * 1.5)
                    base_window_size_np_scaled = np.expand_dims(base_window_size_np_scaled, axis=-1)
                    base_window_size_np_scaled = np.tile(base_window_size_np_scaled, (1, 4))  # (N, 4)

                    centerpoints_pos_tar = torch.squeeze(target_centerpoints, dim=1)  # (N, 2), in [0.0, 1.0], relative to full size
                    centerpoints_pos_tar_np = centerpoints_pos_tar.cpu().data.numpy()  # (N, 2), in [0.0, 1.0], relative to full size
                    centerpoints_pos_tar_np = np.concatenate([centerpoints_pos_tar_np, centerpoints_pos_tar_np], axis=-1)  # (N, 4)

                    ctrlpoints_pred_rel = ctrlpoints_offset_pred_np  # (N, 4), [-1.0, 1.0] relative to window
                    ctrlpoints_pred_offset_global = ctrlpoints_pred_rel * (base_window_size_np_scaled / 2.0)
                    ctrlpoints_pred_global = ctrlpoints_pred_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

                    ctrlpoints_gt_rel = ctrlpoints_offset_gt_np  # (N, 4), [-1.0, 1.0] relative to window
                    ctrlpoints_gt_offset_global = ctrlpoints_gt_rel * (base_window_size_np_scaled / 2.0)
                    ctrlpoints_gt_global = ctrlpoints_gt_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

                    ctrlpoints_ref_rel = ctrlpoints_offset_ref_np  # (N, 4), [-1.0, 1.0] relative to window
                    ctrlpoints_ref_offset_global = ctrlpoints_ref_rel * (base_window_size_np_scaled / 2.0)
                    ctrlpoints_ref_global = ctrlpoints_ref_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

                    endpoints_pred_rel = endpoints_offset_pred_np  # (N, 4), [-1.0, 1.0] relative to window
                    endpoints_pred_offset_global = endpoints_pred_rel * (base_window_size_np_scaled / 2.0)
                    endpoints_pred_global = endpoints_pred_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

                    endpoints_gt_rel = endpoints_offset_gt_np  # (N, 4), [-1.0, 1.0] relative to window
                    endpoints_gt_offset_global = endpoints_gt_rel * (base_window_size_np_scaled / 2.0)
                    endpoints_gt_global = endpoints_gt_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

                    endpoints_ref_rel = endpoints_offset_ref_np  # (N, 4), [-1.0, 1.0] relative to window
                    endpoints_ref_offset_global = endpoints_ref_rel * (base_window_size_np_scaled / 2.0)
                    endpoints_ref_global = endpoints_ref_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

                    stroke_points_ref = np.concatenate([endpoints_ref_global[:, 0:2], ctrlpoints_ref_global, endpoints_ref_global[:, 2:4]], axis=1)  # (N, 8), in full size
                    stroke_points_gt = np.concatenate([endpoints_gt_global[:, 0:2], ctrlpoints_gt_global, endpoints_gt_global[:, 2:4]], axis=1)  # (N, 8), in full size
                    stroke_points_pred = np.concatenate([endpoints_pred_global[:, 0:2], ctrlpoints_pred_global, endpoints_pred_global[:, 2:4]], axis=1)  # (N, 8), in full size

                    reference_images_np = reference_images.cpu().data.numpy()  # (N, H, W, 1), [0-stroke, 1-BG]
                    assert len(stroke_ids) == reference_images_np.shape[0]
                    reference_image_np = reference_images_np[0, :, :, 0] * 255.0  # (H, W), [0-stroke, 255-BG]
                    target_images_np = target_images.cpu().data.numpy()  # (N, H, W, 1), [0-stroke, 1-BG]
                    target_image_np = target_images_np[0, :, :, 0] * 255.0  # (H, W), [0-stroke, 255-BG]

                    img_vis_save_path_ref = os.path.join(save_base, 'ref-' + str(img_index) + '.png')
                    img_vis_save_path_tar_pred = os.path.join(save_base, 'tar_pred-' + str(img_index) + '.png')
                    img_vis_save_path_tar_gt = os.path.join(save_base, 'tar_gt-' + str(img_index) + '.png')

                    draw_sketch_stroke(stroke_points_ref, img_vis_save_path_ref, reference_image_np, image_size)
                    draw_sketch_stroke(stroke_points_pred, img_vis_save_path_tar_pred, target_image_np, image_size)
                    if vis_gt:
                        draw_sketch_stroke(stroke_points_gt, img_vis_save_path_tar_gt, target_image_np, image_size)

                    ## save predicted vector parameters
                    if save_parameters:
                        params_save_dir = os.path.join(save_base, 'params')
                        os.makedirs(params_save_dir, exist_ok=True)
                        stroke_pred_data_save_path = os.path.join(params_save_dir, 'tar_pred-' + str(img_index) + '.jsonl')

                        num_component = num_components[0]
                        target_pred_stroke_data = [[] for _ in range(num_component)]  # component list => curve list => stroke list (N', 4, 2)
                        for s_i, stroke_id in enumerate(stroke_ids):
                            comp_curve_point = stroke_id.split('_')
                            c_i, curve_i, stroke_index = comp_curve_point
                            c_i, curve_i, stroke_index = int(c_i), int(curve_i), int(stroke_index)

                            stroke_points_pred_i = np.reshape(stroke_points_pred[s_i], (4, 2))  # (4, 2), in full size
                            # if len(target_pred_stroke_data) <= c_i:
                            #     target_pred_stroke_data += [[] for _ in range(c_i + 1 - len(target_pred_stroke_data))]

                            if len(target_pred_stroke_data[c_i]) <= curve_i:
                                target_pred_stroke_data[c_i].append([])

                            target_pred_stroke_data[c_i][curve_i].append(stroke_points_pred_i.tolist())

                        stroke_params_data = {}
                        stroke_params_data['stroke_params'] = target_pred_stroke_data
                        with jsonlines.open(stroke_pred_data_save_path, mode='w') as json_writer:
                            json_writer.write(stroke_params_data)

                    if vis_incorrect:
                        # draw incorrect strokes
                        if not occluded_only:
                            endpoint0_error_global = np.sqrt(np.sum(np.power(endpoints_pred_global[:, 0:2] - endpoints_gt_global[:, 0:2], 2), axis=-1))  # (N)
                            endpoint1_error_global = np.sqrt(np.sum(np.power(endpoints_pred_global[:, 2:4] - endpoints_gt_global[:, 2:4], 2), axis=-1))  # (N)
                            endpoint_error_global = np.stack([endpoint0_error_global, endpoint1_error_global], axis=-1)
                            endpoint_error_global = np.max(endpoint_error_global, axis=-1)  # (N)

                            rendered_stroke_pixel_ious = cal_stroke_pixel_iou(stroke_points_gt, stroke_points_pred, image_size)  # (N)

                            # for s_i, stroke_id in enumerate(stroke_ids):
                            #     print(stroke_id, ':', endpoint0_error_global[s_i], endpoint1_error_global[s_i], '|', rendered_stroke_pixel_ious[s_i],
                            #           '| point=', endpoint_error_global[s_i] > endpoint_distance_threshold,
                            #           '| stroke=', rendered_stroke_pixel_ious[s_i] < stroke_pixels_iou_threshold)

                            # First, endpoints should be accurate
                            incorrect_endpoint_states = endpoint_error_global > endpoint_distance_threshold
                            # Then,
                            incorrect_pixel_states = rendered_stroke_pixel_ious < stroke_pixels_iou_threshold

                            incorrect_stroke_states = np.logical_or(incorrect_endpoint_states, incorrect_pixel_states)
                        else:
                            target_occluded_masks_np = target_occluded_masks.cpu().data.numpy()  # (N, H, W, 1), [0-occluded, 1-visible]
                            rendered_stroke_pixel_ious = cal_stroke_pixel_iou(stroke_points_gt, stroke_points_pred, image_size,
                                                                              occluded_masks=target_occluded_masks_np)  # (N)
                            incorrect_stroke_states = rendered_stroke_pixel_ious < stroke_pixels_iou_threshold_occ

                            # for s_i, stroke_id in enumerate(stroke_ids):
                            #     print(stroke_id, ':', rendered_stroke_pixel_ious[s_i],
                            #           '| stroke=', rendered_stroke_pixel_ious[s_i] < stroke_pixels_iou_threshold_occ)

                        save_dir = os.path.join(save_base, 'incorrect')
                        os.makedirs(save_dir, exist_ok=True)

                        draw_sketch_stroke(stroke_points_ref, os.path.join(save_dir, 'ref-' + str(img_index) + '.png'),
                                           reference_image_np, image_size, drawn_states=incorrect_stroke_states)
                        draw_sketch_stroke(stroke_points_gt, os.path.join(save_dir, 'tar_gt-' + str(img_index) + '.png'),
                                           target_image_np, image_size, drawn_states=incorrect_stroke_states)
                        draw_sketch_stroke(stroke_points_pred, os.path.join(save_dir, 'tar_pred-' + str(img_index) + '.png'),
                                           target_image_np, image_size, drawn_states=incorrect_stroke_states)

                    show_i += 1
                    if show_i >= show_num:
                        break

    def inference_full_real(self, save_root, test_max_batch_size=180, do_inv=False):
        # print('-' * 100)
        print('Inference of [Control Point Matching (4/4)] begins ...')

        trained_controlpoint_model_path = os.path.join(self.snapshot_dir, "sketch_ctrlpoint_" + str(self.hps.num_steps) + ".pkl")
        if self.hps.multi_gpu:
            load_weights(trained_controlpoint_model_path, self.controlpoint_model.module)
        else:
            load_weights(trained_controlpoint_model_path, self.controlpoint_model)

        if self.use_cuda:
            self.controlpoint_model = self.controlpoint_model.cuda()

        self.controlpoint_model.eval()

        is_debug = False

        with torch.no_grad():
            batch_data = self.valid_set.get_batch(self.use_cuda, test_img_id=test_img_id)

            reference_images, reference_strokes, reference_strokes_ctrl, target_images, target_images_ori, \
                reference_centerpoints, target_centerpoints, \
                reference_end_ctrl_offset, reference_end_ctrl_offset_single, target_end_offset_pred, \
                base_window_size, base_window_size_single, image_ids, stroke_ids, num_components, \
                component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
                target_transform_shear_x_angles, target_transform_shear_y_angles,\
                target_transform1_translate, target_transform1_scaling, \
                target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y, fixing_states = batch_data
            # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
            # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
            # reference_strokes_ctrl: (N, H, W, 1), [0-stroke, 1-BG]
            # target_images: (N, H, W, 1), [0-stroke, 1-BG]
            # target_images_ori: (N, H, W, 1), [0-stroke, 1-BG]
            # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
            # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
            # reference_end_ctrl_offset: (N, 1, 8), in [-1.0, 1.0]
            # reference_end_ctrl_offset_single: (N, 1, 8), in [-1.0, 1.0]
            # target_end_offset_pred: (N, 1, 4), in [-1.0, 1.0]
            # base_window_size / base_window_size_single: (N, 1), in [0.0, 1.0]

            # component_centerpoints: (N, 1, 2), in [0.0, 1.0], relative to image size
            # component_win_sizes: (N, 1, 2), in image size
            # target_transform_cursors: (N, 1, 2), in [0.0, 1.0], relative to image size
            # target_transform_win_sizes: (N, 1, 2), in image size
            # target_transform_angles: (N, 1), in [-180.0, 180.0]
            # target_transform_shear_x_angles: (N, 1), in [-90.0, 90.0]
            # target_transform_shear_y_angles: (N, 1), in [-90.0, 90.0]

            # target_transform1_translate:(N, 1, 2), [-1.0, 1.0], relative to target trans0 window
            # target_transform1_scaling: (N, 1, 2), [0.2, 2.0], relative to target trans0 window
            # target_transform1_rotate: (N, 1), [-180.0, 180.0]
            # target_transform1_shear_x / target_transform1_shear_y: (N, 1), [-90.0, 90.0]

            # fixing_states: (N, 1), [0-nonfix, 1-fix]

            image_size = reference_images.size()[1]

            assert len(image_ids) == 1
            img_index = image_ids[0]
            # print(' >> img_index', img_index)

            stroke_num = reference_images.size()[0]
            chunks = stroke_num // test_max_batch_size + 1

            reference_images_chunks = torch.chunk(reference_images, chunks, dim=0)
            reference_strokes_chunks = torch.chunk(reference_strokes, chunks, dim=0)
            reference_strokes_ctrl_chunks = torch.chunk(reference_strokes_ctrl, chunks, dim=0)
            target_images_chunks = torch.chunk(target_images, chunks, dim=0)
            reference_centerpoints_chunks = torch.chunk(reference_centerpoints, chunks, dim=0)
            reference_end_ctrl_offset_single_chunks = torch.chunk(reference_end_ctrl_offset_single, chunks, dim=0)
            target_centerpoints_chunks = torch.chunk(target_centerpoints, chunks, dim=0)
            target_end_offset_pred_chunks = torch.chunk(target_end_offset_pred, chunks, dim=0)
            base_window_size_chunks = torch.chunk(base_window_size, chunks, dim=0)
            base_window_size_single_chunks = torch.chunk(base_window_size_single, chunks, dim=0)
            component_centerpoints_chunks = torch.chunk(component_centerpoints, chunks, dim=0)
            component_win_sizes_chunks = torch.chunk(component_win_sizes, chunks, dim=0)
            target_transform_cursors_chunks = torch.chunk(target_transform_cursors, chunks, dim=0)
            target_transform_win_sizes_chunks = torch.chunk(target_transform_win_sizes, chunks, dim=0)
            target_transform_angles_chunks = torch.chunk(target_transform_angles, chunks, dim=0)
            target_transform_shear_x_angles_chunks = torch.chunk(target_transform_shear_x_angles, chunks, dim=0)
            target_transform_shear_y_angles_chunks = torch.chunk(target_transform_shear_y_angles, chunks, dim=0)
            target_transform1_translate_chunks = torch.chunk(target_transform1_translate, chunks, dim=0)
            target_transform1_scaling_chunks = torch.chunk(target_transform1_scaling, chunks, dim=0)
            target_transform1_rotate_chunks = torch.chunk(target_transform1_rotate, chunks, dim=0)
            target_transform1_shear_x_chunks = torch.chunk(target_transform1_shear_x, chunks, dim=0)
            target_transform1_shear_y_chunks = torch.chunk(target_transform1_shear_y, chunks, dim=0)
            fixing_states_chunks = torch.chunk(fixing_states, chunks, dim=0)

            patch_reference_np = []
            patch_stroke_reference_np = []
            patch_reference2_np = []
            patch_stroke_reference2_np = []
            patch_target_ori_np = []
            patch_target_trans1_np = []
            patch_target_trans2_np = []
            patch_segment_target_trans2_np = []
            ctrlpoints_offset_pred_np = []

            for chunk_i in range(chunks):
                patch_reference_ch, patch_stroke_reference_ch, \
                    patch_reference2_ch, patch_stroke_reference2_ch, \
                    patch_target_ori_ch, patch_target_trans1_ch, patch_target_trans2_ch, patch_segment_target_trans2_ch, \
                    _, _, ctrlpoints_offset_pred_ch = \
                    self.controlpoint_model(reference_images=reference_images_chunks[chunk_i],
                                            reference_strokes=reference_strokes_chunks[chunk_i],
                                            reference_strokes_ctrl=reference_strokes_ctrl_chunks[chunk_i],
                                            target_images=target_images_chunks[chunk_i],
                                            centerpoints_pos_ref=reference_centerpoints_chunks[chunk_i],
                                            centerpoints_pos_tar=target_centerpoints_chunks[chunk_i],
                                            end_ctrl_offset_ref=reference_end_ctrl_offset_single_chunks[chunk_i],
                                            end_ctrl_offset_tar=torch.cat([target_end_offset_pred_chunks[chunk_i], target_end_offset_pred_chunks[chunk_i]], dim=-1),
                                            end_offset_tar_pred=target_end_offset_pred_chunks[chunk_i],
                                            base_window_size=base_window_size_chunks[chunk_i],
                                            base_window_size_single=base_window_size_single_chunks[chunk_i],
                                            image_size=image_size,
                                            component_centerpoints=component_centerpoints_chunks[chunk_i],
                                            component_win_sizes=component_win_sizes_chunks[chunk_i],
                                            target_transform_cursors=target_transform_cursors_chunks[chunk_i],
                                            target_transform_win_sizes=target_transform_win_sizes_chunks[chunk_i],
                                            target_transform_angles=target_transform_angles_chunks[chunk_i],
                                            target_transform_shear_x_angles=target_transform_shear_x_angles_chunks[chunk_i],
                                            target_transform_shear_y_angles=target_transform_shear_y_angles_chunks[chunk_i],
                                            target_transform1_translate=target_transform1_translate_chunks[chunk_i],
                                            target_transform1_scaling=target_transform1_scaling_chunks[chunk_i],
                                            target_transform1_rotate=target_transform1_rotate_chunks[chunk_i],
                                            target_transform1_shear_x=target_transform1_shear_x_chunks[chunk_i],
                                            target_transform1_shear_y=target_transform1_shear_y_chunks[chunk_i],
                                            fixing_states=fixing_states_chunks[chunk_i],
                                            )
                # ctrlpoints_offset_trans2_pred: (N, 4), [-1.0, 1.0], relative to target stroke window
                # ctrlpoints_offset_pred: (N, 4), [-1.0, 1.0], relative to ref window

                patch_reference_np_ch = patch_reference_ch.cpu().data.numpy()
                patch_stroke_reference_np_ch = patch_stroke_reference_ch.cpu().data.numpy()
                patch_reference2_np_ch = patch_reference2_ch.cpu().data.numpy()
                patch_stroke_reference2_np_ch = patch_stroke_reference2_ch.cpu().data.numpy()
                patch_target_ori_np_ch = patch_target_ori_ch.cpu().data.numpy()
                patch_target_trans1_np_ch = patch_target_trans1_ch.cpu().data.numpy()
                patch_target_trans2_np_ch = patch_target_trans2_ch.cpu().data.numpy()
                patch_segment_target_trans2_np_ch = patch_segment_target_trans2_ch.cpu().data.numpy()

                patch_reference_np.append(patch_reference_np_ch)
                patch_stroke_reference_np.append(patch_stroke_reference_np_ch)
                patch_reference2_np.append(patch_reference2_np_ch)
                patch_stroke_reference2_np.append(patch_stroke_reference2_np_ch)
                patch_target_ori_np.append(patch_target_ori_np_ch)
                patch_target_trans1_np.append(patch_target_trans1_np_ch)
                patch_target_trans2_np.append(patch_target_trans2_np_ch)
                patch_segment_target_trans2_np.append(patch_segment_target_trans2_np_ch)

                ctrlpoints_offset_pred_np_ch = ctrlpoints_offset_pred_ch.cpu().data.numpy()  # (N, 4), [-1.0, 1.0]
                ctrlpoints_offset_pred_np.append(ctrlpoints_offset_pred_np_ch)

            patch_reference_np = np.concatenate(patch_reference_np, axis=0)
            patch_stroke_reference_np = np.concatenate(patch_stroke_reference_np, axis=0)
            patch_reference2_np = np.concatenate(patch_reference2_np, axis=0)
            patch_stroke_reference2_np = np.concatenate(patch_stroke_reference2_np, axis=0)
            patch_target_ori_np = np.concatenate(patch_target_ori_np, axis=0)
            patch_target_trans1_np = np.concatenate(patch_target_trans1_np, axis=0)
            patch_target_trans2_np = np.concatenate(patch_target_trans2_np, axis=0)
            patch_segment_target_trans2_np = np.concatenate(patch_segment_target_trans2_np, axis=0)

            if is_debug:
                for s_i in range(patch_reference_np.shape[0]):
                    stroke_id = stroke_ids[s_i]

                    debug_root = os.path.join(save_root, 'debug')
                    os.makedirs(debug_root, exist_ok=True)

                    patch_reference_np_i = save_image(patch_reference_np[s_i], debug_root,
                                                        'ref-' + str(img_index) + '-' + stroke_id + '.png')
                    save_image(patch_stroke_reference_np[s_i], debug_root,
                                'ref_stroke-' + str(img_index) + '-' + stroke_id + '.png')
                    save_image(patch_target_ori_np[s_i], debug_root,
                                'tar_ori-' + str(img_index) + '-' + stroke_id + '.png')

                    patch_reference2_np_i = save_image(patch_reference2_np[s_i], debug_root,
                                                        'ref2-' + str(img_index) + '-' + stroke_id + '.png')
                    save_image(patch_stroke_reference2_np[s_i], debug_root,
                                'ref_stroke2-' + str(img_index) + '-' + stroke_id + '.png')
                    patch_target_np_i = save_image(patch_target_trans2_np[s_i], debug_root,
                                                    'tar_trans2-' + str(img_index) + '-' + stroke_id + '.png')

                    save_image_overlap(patch_target_trans1_np[s_i], patch_reference_np_i, debug_root,
                                        'tar_trans1_vis-' + str(img_index) + '-' + stroke_id + '.png')
                    save_image_overlap(patch_target_trans2_np[s_i], patch_reference2_np_i, debug_root,
                                        'tar_trans2_vis-' + str(img_index) + '-' + stroke_id + '.png')
                    save_image_overlap(patch_segment_target_trans2_np[s_i], patch_target_np_i, debug_root,
                                        'tar_segment_vis-' + str(img_index) + '-' + stroke_id + '.png')

            ctrlpoints_offset_pred_np = np.concatenate(ctrlpoints_offset_pred_np, axis=0)  # (N, 4), [-1.0, 1.0]
            endpoints_offset_pred_np = target_end_offset_pred.squeeze(dim=1).cpu().data.numpy()  # (N, 4) [-1.0, 1.0]

            reference_end_ctrl_offset_np = reference_end_ctrl_offset.squeeze(dim=1).cpu().data.numpy()  # (N, 8), in [-1.0, 1.0]
            endpoints_offset_ref_np = np.concatenate([reference_end_ctrl_offset_np[:, 0:2],
                                                        reference_end_ctrl_offset_np[:, 6:8]], axis=-1)  # (N, 4), [-1.0, 1.0]
            ctrlpoints_offset_ref_np = reference_end_ctrl_offset_np[:, 2:6]  # (N, 4), [-1.0, 1.0]

            ## convert to global coordinate
            base_window_size = torch.squeeze(base_window_size, dim=-1)  # (N), in [0.0, 1.0]
            base_window_size_np = base_window_size.cpu().data.numpy()  # (N), in [0.0, 1.0]
            base_window_size_np_scaled = base_window_size_np * image_size * self.hps.window_size_scaling_ref  # (N), in full size
            base_window_size_np_scaled = np.clip(base_window_size_np_scaled, self.hps.window_size_min, image_size * 1.5)
            base_window_size_np_scaled = np.expand_dims(base_window_size_np_scaled, axis=-1)
            base_window_size_np_scaled = np.tile(base_window_size_np_scaled, (1, 4))  # (N, 4)

            centerpoints_pos_tar = torch.squeeze(target_centerpoints, dim=1)  # (N, 2), in [0.0, 1.0], relative to full size
            centerpoints_pos_tar_np = centerpoints_pos_tar.cpu().data.numpy()  # (N, 2), in [0.0, 1.0], relative to full size
            centerpoints_pos_tar_np = np.concatenate([centerpoints_pos_tar_np, centerpoints_pos_tar_np], axis=-1)  # (N, 4)

            ctrlpoints_pred_rel = ctrlpoints_offset_pred_np  # (N, 4), [-1.0, 1.0] relative to window
            ctrlpoints_pred_offset_global = ctrlpoints_pred_rel * (base_window_size_np_scaled / 2.0)
            ctrlpoints_pred_global = ctrlpoints_pred_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

            ctrlpoints_ref_rel = ctrlpoints_offset_ref_np  # (N, 4), [-1.0, 1.0] relative to window
            ctrlpoints_ref_offset_global = ctrlpoints_ref_rel * (base_window_size_np_scaled / 2.0)
            ctrlpoints_ref_global = ctrlpoints_ref_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

            endpoints_pred_rel = endpoints_offset_pred_np  # (N, 4), [-1.0, 1.0] relative to window
            endpoints_pred_offset_global = endpoints_pred_rel * (base_window_size_np_scaled / 2.0)
            endpoints_pred_global = endpoints_pred_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

            endpoints_ref_rel = endpoints_offset_ref_np  # (N, 4), [-1.0, 1.0] relative to window
            endpoints_ref_offset_global = endpoints_ref_rel * (base_window_size_np_scaled / 2.0)
            endpoints_ref_global = endpoints_ref_offset_global + centerpoints_pos_tar_np * image_size  # (N, 4), in full size

            stroke_points_ref = np.concatenate([endpoints_ref_global[:, 0:2], ctrlpoints_ref_global, endpoints_ref_global[:, 2:4]], axis=1)  # (N, 8), in full size
            stroke_points_pred = np.concatenate([endpoints_pred_global[:, 0:2], ctrlpoints_pred_global, endpoints_pred_global[:, 2:4]], axis=1)  # (N, 8), in full size

            reference_images_np = reference_images.cpu().data.numpy()  # (N, H, W, 1), [0-stroke, 1-BG]
            reference_image_np = reference_images_np[0, :, :, 0] * 255.0  # (H, W), [0-stroke, 255-BG]
            target_images_np = target_images_ori.cpu().data.numpy()  # (N, H, W, 1), [0-stroke, 1-BG]
            target_image_np = target_images_np[0, :, :, 0] * 255.0  # (H, W), [0-stroke, 255-BG]

            ref_bg_save_dir = os.path.join(save_root, 'ref-bg')
            os.makedirs(ref_bg_save_dir, exist_ok=True)
            draw_sketch_stroke(stroke_points_ref, os.path.join(save_root, 'ref-' + str(img_index) + '.png'),
                                None, image_size)
            draw_sketch_stroke(stroke_points_ref, os.path.join(ref_bg_save_dir, 'ref-' + str(img_index) + '.png'),
                                reference_image_np, image_size)
            draw_sketch_stroke(stroke_points_pred, os.path.join(save_root, 'tar_pred-' + str(img_index) + '.png'),
                                target_image_np, image_size)

            ## save predicted vector parameters
            params_save_dir = os.path.join(save_root, 'params')
            os.makedirs(params_save_dir, exist_ok=True)
            stroke_pred_data_save_path = os.path.join(params_save_dir, 'tar_pred-' + str(img_index) + '.jsonl')

            num_component = num_components[0]
            target_pred_stroke_data = [[] for _ in range(num_component)]  # component list => curve list => stroke list (N', 4, 2)
            for s_i, stroke_id in enumerate(stroke_ids):
                comp_curve_point = stroke_id.split('_')
                c_i, curve_i, stroke_index = comp_curve_point
                c_i, curve_i, stroke_index = int(c_i), int(curve_i), int(stroke_index)

                stroke_points_pred_i = np.reshape(stroke_points_pred[s_i], (4, 2))  # (4, 2), in full size
                # if len(target_pred_stroke_data) <= c_i:
                #     target_pred_stroke_data += [[] for _ in range(c_i + 1 - len(target_pred_stroke_data))]

                if len(target_pred_stroke_data[c_i]) <= curve_i:
                    target_pred_stroke_data[c_i].append([])

                target_pred_stroke_data[c_i][curve_i].append(stroke_points_pred_i.tolist())

            stroke_params_data = {}
            stroke_params_data['stroke_params'] = target_pred_stroke_data
            with jsonlines.open(stroke_pred_data_save_path, mode='w') as json_writer:
                json_writer.write(stroke_params_data)

            print('Inference finished, results saved to', save_root)