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

from vgg_utils.VGG16 import VGG_Slim
from hparam import HParams
from image_utils.image_processing import save_image, save_image_overlap
from image_utils.model_processing import get_coordconv, add_coords, normalize_image_m1to1, image_cropping_stn, image_cropping_stn_multi, \
    load_weights, print_model_variables
from network.vanilla import CNN_Encoder, MLP_Decoder
from configs.example_configs import test_img_id

tf.get_logger().setLevel('INFO')


def get_default_hparams():
    """Return default HParams for sketch-rnn."""
    hparams = HParams(
        workspace='FAD3-T13-2.0x-51',
        transform_global_model_name='FAD3-T12-2.0x-51-min=64',
        dataset_base='/home/Datasets/CreativeSketch/proc_data3/',

        ############ For inference only ############
        do_dataset_filtering=False,  # set to True for training
        ############################################

        multi_gpu=False,

        num_steps=30000,
        save_every=10000,  # Number of steps per checkpoint creation.
        log_img_every=500,  # Number of steps per log image creation.

        batch_size=25,

        # image_size=512,

        transform_with_shear=True,

        enc_model_transform_local='combined',  # ['combined', 'separated']
        dec_model_transform_local='mlp',  # ['rnn', 'mlp']
        z_size=256,  # Size of latent vector z.
        transform_local_module_zero_init='last',  # ['none', 'last', 'all']
        add_coordconv=True,

        raster_loss_base_type='perceptual',  # [l1, mse, perceptual]
        perc_loss_layers=['ReLU5_1'],
        perc_loss_layer_eval=['ReLU3_3', 'ReLU5_1'],
        perc_loss_fuse_type='add',  # ['max', 'add', 'raw_add', 'weighted_sum']
        perceptual_model_path='vgg_utils/quickdraw-perceptual.pth',

        grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.

        learning_rate=1e-4,  # Learning rate.
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        decay_power=0.9,
        min_learning_rate=1e-6,  # Minimum learning rate.

        snapshot_root='outputs/transform/snapshot',
        log_img_root='outputs/transform/log_img',
        log_root='outputs/transform/log',
        inference_root='outputs/transform/inference-FAD3',
        inference_root_real='outputs/transform/inference-Real',
    )
    return hparams


class Transformation_Local_Model(nn.Module):
    def __init__(self, hps):
        super(Transformation_Local_Model, self).__init__()
        self.hps = hps

        transform_out_size = 2  # scaling
        transform_out_size += 2  # translation
        transform_out_size += 1  # rotation
        if self.hps.transform_with_shear:
            transform_out_size += 2
        cnn_out_size = self.hps.z_size

        # transform encoder
        if self.hps.enc_model_transform_local == 'combined':
            cnn_in_size = 3
            if self.hps.add_coordconv:
                cnn_in_size += 2
            self.encoder_transform_local = CNN_Encoder(cnn_in_size, cnn_out_size, input_size=self.hps.raster_size)
        else:
            raise Exception('Unknown enc_model_transform_local:', self.hps.enc_model_transform_local)

        dec_in_size = self.hps.z_size
        if self.hps.dec_model_transform_local == 'mlp':
            self.decoder_transform_local = MLP_Decoder(dec_in_size, transform_out_size, zero_init=self.hps.transform_local_module_zero_init)
        else:
            raise Exception('Unknown dec_model_transform_local:', self.hps.dec_model_transform_local)

        if self.hps.add_coordconv:
            self.coordconv_input = get_coordconv(self.hps.raster_size)  # (2, raster_size, raster_size)

    def forward(self, reference_images, reference_components, reference_strokes, reference_strokes_prev, reference_strokes_next,
                target_images, target_components, target_layer_masks,
                centerpoints_pos_ref, centerpoints_pos_tar, base_window_size, image_size,
                component_centerpoints, component_win_sizes,
                target_transform_cursors, target_transform_win_sizes, target_transform_angles,
                target_transform_shear_x_angles, target_transform_shear_y_angles):
        """
        :param reference_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_components: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_strokes: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_strokes_prev: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_strokes_next: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_components: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_layer_masks: (N, H, W, 1), float32, [0-FG, 1-BG]
        :param centerpoints_pos_ref: (N, 1, 2), float32, in [0.0, 1.0]
        :param centerpoints_pos_tar: (N, 1, 2), float32, in [0.0, 1.0]
        :param base_window_size: (N, 1), float32, in [0.0, 1.0]
        :param component_centerpoints: (N, 1, 2), in [0.0, 1.0], relative to image size
        :param component_win_sizes: (N, 1, 2), in image size
        :param target_transform_cursors: (N, 1, 2), in [0.0, 1.0], relative to image size
        :param target_transform_win_sizes: (N, 1, 2), in image size
        :param target_transform_angles: (N, 1), in [-180.0, 180.0]
        :param target_transform_shear_x_angles: (N, 1), in [-90.0, 90.0]
        :param target_transform_shear_y_angles: (N, 1), in [-90.0, 90.0]
        :return:
        """
        self.image_size = image_size

        patch_reference, patch_stroke_reference, patch_stroke_prev_reference, patch_stroke_next_reference, \
            patch_component_reference, \
            patch_target_trans0, patch_component_target_trans0, patch_layer_mask_target_trans0, \
            patch_target_trans1, patch_component_target_trans1, patch_layer_mask_target_trans1, \
            pred_translate_tar, pred_scaling_times_tar, pred_rotate_angle_tar, \
            pred_shear_x_angle_tar, pred_shear_y_angle_tar = \
            self.get_points_and_raster_image(reference_images, reference_components, reference_strokes,
                                             reference_strokes_prev, reference_strokes_next,
                                             target_images, target_components, target_layer_masks,
                                             centerpoints_pos_ref, centerpoints_pos_tar,
                                             base_window_size,
                                             component_centerpoints, component_win_sizes,
                                             target_transform_cursors, target_transform_win_sizes, target_transform_angles,
                                             target_transform_shear_x_angles, target_transform_shear_y_angles)
        # patch_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_stroke_reference / patch_stroke_prev_reference / patch_stroke_next_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_component_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_target_trans0: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_component_target_trans0: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_layer_mask_target_trans0: (N, raster_size, raster_size), [0.0-FG, 1.0-BG]
        # patch_target_trans1: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_component_target_trans1: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_layer_mask_target_trans1: (N, raster_size, raster_size), [0.0-FG, 1.0-BG]

        # pred_translate_tar:(N, 1, 2), [-1.0, 1.0], relative to target trans0 window
        # pred_scaling_times_tar: (N, 1, 2), [0.2, 2.0], relative to target trans0 window
        # pred_rotate_angle_tar: (N, 1), [-180.0, 180.0]
        # pred_shear_x_angle_tar / pred_shear_y_angle_tar: (N, 1), [-90.0, 90.0]

        return patch_reference, patch_stroke_reference, patch_stroke_prev_reference, patch_stroke_next_reference, \
            patch_component_reference, \
            patch_target_trans0, patch_component_target_trans0, patch_layer_mask_target_trans0, \
            patch_target_trans1, patch_component_target_trans1, patch_layer_mask_target_trans1, \
            pred_translate_tar, pred_scaling_times_tar, pred_rotate_angle_tar, \
            pred_shear_x_angle_tar, pred_shear_y_angle_tar

    def get_points_and_raster_image(self, reference_images, reference_components,
                                    reference_strokes, reference_strokes_prev, reference_strokes_next,
                                    target_images, target_components, target_layer_masks,
                                    centerpoints_pos_ref, centerpoints_pos_tar,
                                    base_window_size,
                                    component_centerpoints, component_win_sizes,
                                    target_transform_cursors, target_transform_win_sizes, target_transform_angles,
                                    target_transform_shear_x_angles, target_transform_shear_y_angles):
        """
        :param reference_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_components: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_strokes: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_strokes_prev: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_strokes_next: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_components: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_layer_masks: (N, H, W, 1), float32, [0-FG, 1-BG]
        :param centerpoints_pos_ref: (N, 1, 2), float32, in [0.0, 1.0]
        :param centerpoints_pos_tar: (N, 1, 2), float32, in [0.0, 1.0]
        :param base_window_size: (N, 1), float32, in [0.0, 1.0]
        :param component_centerpoints: (N, 1, 2), in [0.0, 1.0], relative to image size
        :param component_win_sizes: (N, 1, 2), in image size
        :param target_transform_cursors: (N, 1, 2), in [0.0, 1.0], relative to image size
        :param target_transform_win_sizes: (N, 1, 2), in image size
        :param target_transform_angles: (N, 1), in [-180.0, 180.0]
        :param target_transform_shear_x_angles: (N, 1), in [-90.0, 90.0]
        :param target_transform_shear_y_angles: (N, 1), in [-90.0, 90.0]
        :return:
        """
        # cursor position
        cursor_position_loop_ref = centerpoints_pos_ref  # (N, 1, 2), in size [0.0, 1.0]
        # cursor_position_loop_tar = centerpoints_pos_tar  # (N, 1, 2), in size [0.0, 1.0]

        curr_window_size = base_window_size.unsqueeze(dim=-1)  # (N, 1, 1), in [0.0, 1.0]
        curr_window_size = torch.mul(curr_window_size, self.image_size)  # (N, 1, 1), in full size
        curr_window_size = torch.mul(curr_window_size, self.hps.window_size_scaling_ref)  # (N, 1, 1), in full size
        curr_window_size = torch.max(curr_window_size, torch.tensor(self.hps.window_size_min).float().cuda())
        curr_window_size = torch.min(curr_window_size, torch.tensor(self.image_size * 1.5).float().cuda())
        curr_window_size = torch.cat([curr_window_size, curr_window_size], dim=-1)  # (N, 1, 2), in full size

        ## reference_images: (N, H, W, 1), [0.0-stroke, 1.0-BG]
        crop_inputs_ref = torch.cat([reference_images, reference_strokes, reference_components, reference_strokes_prev, reference_strokes_next], dim=-1)  # (N, H, W, *)
        cropped_outputs = image_cropping_stn(cursor_position_loop_ref, crop_inputs_ref, self.image_size, self.hps.raster_size, curr_window_size)

        curr_patch_image_ref = cropped_outputs[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_stroke_ref = cropped_outputs[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_component_ref = cropped_outputs[:, :, :, 2:3]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_stroke_prev_ref = cropped_outputs[:, :, :, 3:4]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_stroke_next_ref = cropped_outputs[:, :, :, 4:5]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]

        curr_patch_image_ref_out = torch.squeeze(curr_patch_image_ref, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_stroke_ref_out = torch.squeeze(curr_patch_stroke_ref, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_component_ref_out = torch.squeeze(curr_patch_component_ref, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_stroke_prev_ref_out = torch.squeeze(curr_patch_stroke_prev_ref, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_stroke_next_ref_out = torch.squeeze(curr_patch_stroke_next_ref, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        curr_patch_image_ref = normalize_image_m1to1(curr_patch_image_ref)
        curr_patch_component_ref = normalize_image_m1to1(curr_patch_component_ref)
        # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

        ## target_images: (N, H, W, 1), [0.0-stroke, 1.0-BG]
        crop_inputs_tar = torch.cat([target_images, target_components, target_layer_masks], dim=-1)  # (N, H, W, *)

        additional_offset = (cursor_position_loop_ref - component_centerpoints) * self.image_size / (component_win_sizes / 2.0)  # (N, 1, 2), [-1, 1]
        additional_scale = curr_window_size / component_win_sizes  # (N, 1, 2), [0, 1+]
        cropped_outputs = image_cropping_stn(target_transform_cursors, crop_inputs_tar, self.image_size, self.hps.raster_size, target_transform_win_sizes,
                                             rotation_angle=target_transform_angles,
                                             shear_x_angle=target_transform_shear_x_angles, shear_y_angle=target_transform_shear_y_angles,
                                             additional_transform=True, addi_offset=additional_offset, addi_scale=additional_scale)

        curr_patch_image_tar_temp = cropped_outputs[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_component_tar_temp = cropped_outputs[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_layer_mask_tar_temp = cropped_outputs[:, :, :, 2:3]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]

        curr_patch_image_tar_trans0_out = torch.squeeze(curr_patch_image_tar_temp, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_component_tar_trans0_out = torch.squeeze(curr_patch_component_tar_temp, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_layer_mask_tar_trans0_out = torch.squeeze(curr_patch_layer_mask_tar_temp, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        curr_patch_image_tar_trans0 = normalize_image_m1to1(curr_patch_image_tar_temp)
        # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

        encoded_z = self.build_encoder_transform_local(curr_patch_image_ref, curr_patch_component_ref, curr_patch_image_tar_trans0)  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        transform_local_output = self.build_decoder_transform_local(encoded_z)
        # transform_local_output: (N, 7)

        ## Transform locally
        transform_local_output_translation = transform_local_output[:, 0:2]  # (N, 2)
        transform_local_output_scaling = transform_local_output[:, 2:4]  # (N, 2)
        transform_local_output_rotate_angle = transform_local_output[:, 4:5]  # (N, 1)
        if self.hps.transform_with_shear:
            transform_local_output_shear_x_angle = transform_local_output[:, 5:6]  # (N, 1)
            transform_local_output_shear_y_angle = transform_local_output[:, 6:7]  # (N, 1)

        # Translation
        pred_window_translate_tar = torch.tanh(transform_local_output_translation)  # (N, 2), [-1.0, 1.0]
        pred_window_translate_tar = pred_window_translate_tar.unsqueeze(dim=1)  # (N, 1, 2), [-1.0, 1.0], relative to target trans0 window

        # Scaling
        pred_window_scaling_times_tar = torch.tanh(transform_local_output_scaling)  # (N, 2), [-1.0, 1.0]
        pred_window_scaling_times_tar = (pred_window_scaling_times_tar + 1.0) / 2.0 * self.hps.window_size_scaling_times_tar[1]  # (N, 2), [0.0, 2.0]
        pred_window_scaling_times_tar = torch.clamp(pred_window_scaling_times_tar,
                                                    self.hps.window_size_scaling_times_tar[0],
                                                    self.hps.window_size_scaling_times_tar[1])  # (N, 2), [0.2, 2.0], relative to target trans0 window
        pred_window_scaling_times_tar = pred_window_scaling_times_tar.unsqueeze(dim=1)  # (N, 1, 2), [0.2, 2.0], relative to target trans0 window

        # Rotation
        pred_window_rotate_angle_tar = torch.tanh(transform_local_output_rotate_angle)  # (N, 1), [-1.0, 1.0]
        pred_window_rotate_angle_tar = torch.mul(pred_window_rotate_angle_tar, 180.0)  # (N, 1), [-180.0, 180.0]

        if self.hps.transform_with_shear:
            pred_window_shear_x_angle_tar = torch.tanh(transform_local_output_shear_x_angle)  # (N, 1), [-1.0, 1.0]
            pred_window_shear_x_angle_tar = torch.mul(pred_window_shear_x_angle_tar, 90.0)  # (N, 1), [-90.0, 90.0]
            pred_window_shear_y_angle_tar = torch.tanh(transform_local_output_shear_y_angle)  # (N, 1), [-1.0, 1.0]
            pred_window_shear_y_angle_tar = torch.mul(pred_window_shear_y_angle_tar, 90.0)  # (N, 1), [-90.0, 90.0]
        else:
            pred_window_shear_x_angle_tar = None
            pred_window_shear_y_angle_tar = None

        ## crop the target again
        crop_inputs_tar = torch.cat([target_images, target_components, target_layer_masks], dim=-1)  # (N, H, W, *)
        cropped_outputs_tar = image_cropping_stn_multi(target_transform_cursors, crop_inputs_tar, self.image_size, self.hps.raster_size,
                                                       target_transform_win_sizes,
                                                       rotation_angle=target_transform_angles,
                                                       shear_x_angle=target_transform_shear_x_angles,
                                                       shear_y_angle=target_transform_shear_y_angles,
                                                       additional_transform=True,
                                                       addi_offset=additional_offset, addi_scale=additional_scale,
                                                       additional_transform3=True,
                                                       addi_offset3=pred_window_translate_tar,
                                                       addi_scale3=pred_window_scaling_times_tar,
                                                       addi_rotate3=pred_window_rotate_angle_tar,
                                                       addi_shear_x3=pred_window_shear_x_angle_tar,
                                                       addi_shear_y3=pred_window_shear_y_angle_tar)
        curr_patch_image_tar = cropped_outputs_tar[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_component_tar = cropped_outputs_tar[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_layer_mask_tar = cropped_outputs_tar[:, :, :, 2:3]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]

        curr_patch_image_tar_trans1_out = torch.squeeze(curr_patch_image_tar, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_component_tar_trans1_out = torch.squeeze(curr_patch_component_tar, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_layer_mask_tar_trans1_out = torch.squeeze(curr_patch_layer_mask_tar, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        return curr_patch_image_ref_out, curr_patch_stroke_ref_out, curr_patch_stroke_prev_ref_out, curr_patch_stroke_next_ref_out, \
               curr_patch_component_ref_out, \
               curr_patch_image_tar_trans0_out, curr_patch_component_tar_trans0_out, curr_patch_layer_mask_tar_trans0_out, \
               curr_patch_image_tar_trans1_out, curr_patch_component_tar_trans1_out, curr_patch_layer_mask_tar_trans1_out, \
               pred_window_translate_tar, pred_window_scaling_times_tar, pred_window_rotate_angle_tar, \
               pred_window_shear_x_angle_tar, pred_window_shear_y_angle_tar

    def build_encoder_transform_local(self, patch_image_ref, patch_component_ref, patch_image_tar):
        """
        :param patch_image_ref & patch_component_ref & patch_image_tar: (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        :return:
        """
        # transform to nchw
        patch_images_ref = patch_image_ref  # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        patch_images_ref = patch_images_ref.permute(0, 3, 1, 2)  # (N, 1, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
        patch_components_ref = patch_component_ref  # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        patch_components_ref = patch_components_ref.permute(0, 3, 1, 2)  # (N, 1, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
        patch_images_tar = patch_image_tar  # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]
        patch_images_tar = patch_images_tar.permute(0, 3, 1, 2)  # (N, 1, raster_size, raster_size), [-1.0-stroke, 1.0-BG]

        if self.hps.enc_model_transform_local == 'combined':
            batch_input = torch.cat([patch_images_ref, patch_components_ref, patch_images_tar], dim=1)  # (N, 3, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
            if self.hps.add_coordconv:
                batch_input = add_coords(batch_input, self.coordconv_input)  # (N, in_dim + 2, in_H, in_W)
            output = self.encoder_transform_local(batch_input)  # (N, z_size)
        else:
            raise Exception('Unknown enc_model_transform_local:', self.hps.enc_model_transform_local)

        return output

    def build_decoder_transform_local(self, dec_input):
        """
        :param dec_input: (N, in_dim)
        :return:
        """
        output = self.decoder_transform_local(dec_input)
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

        self.transform_local_model = Transformation_Local_Model(hps)

        self.perceptual_model = VGG_Slim()

        params_list = []

        if self.hps.multi_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.transform_local_model = nn.DataParallel(self.transform_local_model)
            self.perceptual_model = nn.DataParallel(self.perceptual_model)

            params_list.append({'params': self.transform_local_model.module.parameters()})
        else:
            params_list.append({'params': self.transform_local_model.parameters()})
        self.optimizer = optim.Adam(params_list, lr=hps.learning_rate)

        self.start_step = 0
        self.use_cuda = torch.cuda.is_available()

    def train(self):
        # load weight
        print('-' * 100)
        if self.hps.multi_gpu:
            load_weights(self.hps.perceptual_model_path, self.perceptual_model.module)
        else:
            load_weights(self.hps.perceptual_model_path, self.perceptual_model)
        print('-' * 100)

        print('## All variables:')
        if self.hps.multi_gpu:
            gen_num_param = print_model_variables(self.transform_local_model.module.named_parameters(), 'Transformation_Local_Model')
            vgg_num_param = print_model_variables(self.perceptual_model.module.named_parameters(), 'Perceptual model')
        else:
            gen_num_param = print_model_variables(self.transform_local_model.named_parameters(), 'Transformation_Local_Model')
            vgg_num_param = print_model_variables(self.perceptual_model.named_parameters(), 'Perceptual model')
        total_num_param = gen_num_param
        total_num_param += vgg_num_param
        print('Total variables %i.' % total_num_param)

        # print('## Trainable variables:')
        # for param_group in self.optimizer.param_groups:
        #     print(param_group["params"])

        # setup tensorboards
        train_summary_writer = Logger(self.log_dir)

        mean_perc_relu_losses = [0.0 for _ in range(len(self.hps.perc_loss_layers))]

        if self.use_cuda:
            self.transform_local_model = self.transform_local_model.cuda()
            self.perceptual_model = self.perceptual_model.cuda()

        start = time.time()

        self.perceptual_model.eval()

        for step in range(self.start_step, self.hps.num_steps):
            # print('## Step:', step)
            self.transform_local_model.train()

            curr_learning_rate = ((self.hps.learning_rate - self.hps.min_learning_rate) *
                                  (1 - step / self.hps.num_steps) ** self.hps.decay_power + self.hps.min_learning_rate)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = curr_learning_rate

            train_cost, raster_cost, perc_relu_costs_raw, perc_relu_costs_norm = \
                self.train_step(step, self.train_set, mean_perc_relu_losses)

            ## update mean_raster_loss
            for layer_i in range(len(self.hps.perc_loss_layers)):
                perc_relu_costs_raw_numpy = perc_relu_costs_raw.cpu().detach().numpy()
                perc_relu_cost_raw = perc_relu_costs_raw_numpy[layer_i]
                mean_perc_relu_loss = mean_perc_relu_losses[layer_i]
                mean_perc_relu_loss = (mean_perc_relu_loss * step + perc_relu_cost_raw) / float(step + 1)
                mean_perc_relu_losses[layer_i] = mean_perc_relu_loss

            if (step + 1) % 20 == 0:
                end = time.time()
                time_taken = end - start

                train_summary_writer.scalar_summary('Train_cost', train_cost.item(), step=step + 1)
                train_summary_writer.scalar_summary('Train_raster_cost', raster_cost.item(), step=step + 1)
                train_summary_writer.scalar_summary('Learning_Rate', curr_learning_rate, step=step + 1)
                train_summary_writer.scalar_summary('Time_Taken_Train', time_taken, step=step + 1)

                for loss_layer_i in range(len(self.hps.perc_loss_layers)):
                    loss_layer = self.hps.perc_loss_layers[loss_layer_i]
                    train_summary_writer.scalar_summary('Train_raster_cost_' + loss_layer,
                                                        perc_relu_costs_raw[loss_layer_i].item(), step=step + 1)
                    train_summary_writer.scalar_summary('Train_raster_cost_' + loss_layer + '_norm',
                                                        perc_relu_costs_norm[loss_layer_i].item(), step=step + 1)

                output_format = ('step: %d, lr: %.6f, cost: %.6f, ras: %.6f, '
                                 'time: %.1f')
                output_values = ((step + 1), curr_learning_rate, train_cost.item(), raster_cost.item(),
                                 time_taken)
                output_log = output_format % output_values
                print(output_log)
                tf.get_logger().info(output_log)
                start = time.time()

            if (step + 1) % self.hps.log_img_every == 0:
                self.transform_local_model.eval()
                self.save_log_images(self.valid_set, self.log_img_dir, (step + 1))

            if (step + 1) % self.hps.save_every == 0:
                self.save_model(step_num=step + 1, save_root=self.snapshot_dir)

        # save model for final step
        self.save_model(step_num=self.hps.num_steps, save_root=self.snapshot_dir)

    def train_step(self, step, data_set, perc_loss_mean_list):
        reference_images, reference_components, reference_strokes, target_images, target_components, \
            reference_centerpoints, target_centerpoints, base_window_size, _, _, \
            component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
            target_transform_shear_x_angles, target_transform_shear_y_angles = \
            data_set.get_batch(self.use_cuda)
        # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
        # reference_components: (N, H, W, 1), [0-stroke, 1-BG]
        # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
        # target_images: (N, H, W, 1), [0-stroke, 1-BG]
        # target_components: (N, H, W, 1), [0-stroke, 1-BG]
        # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
        # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
        # base_window_size: (N, 1), in [0.0, 1.0]

        # component_centerpoints: (N, 1, 2), in [0.0, 1.0], relative to image size
        # component_win_sizes: (N, 1, 2), in image size
        # target_transform_cursors: (N, 1, 2), in [0.0, 1.0], relative to image size
        # target_transform_win_sizes: (N, 1, 2), in image size
        # target_transform_angles: (N, 1), in [-180.0, 180.0]
        # target_transform_shear_x_angles: (N, 1), in [-90.0, 90.0]
        # target_transform_shear_y_angles: (N, 1), in [-90.0, 90.0]

        image_size = reference_images.size()[1]

        _, _, patch_component_reference, _, _, _, patch_component_target_trans1, _, _, _, _, _ = \
            self.transform_local_model(reference_images=reference_images, reference_components=reference_components, reference_strokes=reference_strokes,
                                target_images=target_images, target_components=target_components,
                                centerpoints_pos_ref=reference_centerpoints,
                                centerpoints_pos_tar=target_centerpoints,
                                base_window_size=base_window_size,
                                image_size=image_size,
                                component_centerpoints=component_centerpoints,
                                component_win_sizes=component_win_sizes,
                                target_transform_cursors=target_transform_cursors,
                                target_transform_win_sizes=target_transform_win_sizes,
                                target_transform_angles=target_transform_angles,
                                target_transform_shear_x_angles=target_transform_shear_x_angles,
                                target_transform_shear_y_angles=target_transform_shear_y_angles
                                )
        # patch_component_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_component_target_trans1: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        perc_map_pred = self.perceptual_model(patch_component_target_trans1)
        perc_map_gt = self.perceptual_model(patch_component_reference)

        raster_cost, perc_relu_losses_raw, perc_relu_losses_norm = \
            self.get_raster_loss(step, patch_component_target_trans1, patch_component_reference,
                                 loss_type=self.hps.raster_loss_base_type,
                                 return_map_pred=perc_map_pred, return_map_gt=perc_map_gt,
                                 raster_perc_loss_layer=self.hps.perc_loss_layers,
                                 perc_loss_mean_list=perc_loss_mean_list)
        cost = raster_cost

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost, raster_cost, perc_relu_losses_raw, perc_relu_losses_norm

    def get_raster_loss(self, last_step_num, pred_imgs, gt_imgs, loss_type, return_map_pred, return_map_gt,
                        raster_perc_loss_layer, perc_loss_mean_list):
        perc_layer_losses_raw = []
        perc_layer_losses_norm = []

        if loss_type == 'l1':
            ras_cost = torch.mean(torch.abs(torch.sub(gt_imgs, pred_imgs)))  # ()
        elif loss_type == 'mse':
            ras_cost = torch.mean(torch.pow(torch.sub(gt_imgs, pred_imgs), 2))  # ()
        elif loss_type == 'perceptual':
            perc_loss_type = 'l1'  # [l1, mse]
            perc_layers = raster_perc_loss_layer

            for perc_layer in perc_layers:
                if perc_loss_type == 'l1':
                    perc_layer_loss = torch.mean(torch.abs(torch.sub(return_map_pred[perc_layer],
                                                                     return_map_gt[perc_layer])))  # ()
                elif perc_loss_type == 'mse':
                    perc_layer_loss = torch.mean(torch.pow(torch.sub(return_map_pred[perc_layer],
                                                                     return_map_gt[perc_layer]), 2))  # ()
                else:
                    raise NameError('Unknown perceptual loss type:', perc_loss_type)
                perc_layer_losses_raw.append(perc_layer_loss)

            for loop_i in range(len(perc_layers)):
                perc_relu_loss_raw = perc_layer_losses_raw[loop_i]  # ()
                curr_relu_mean = (perc_loss_mean_list[loop_i] * last_step_num + perc_relu_loss_raw) / (
                            last_step_num + 1.0)
                relu_cost_norm = perc_relu_loss_raw / curr_relu_mean
                perc_layer_losses_norm.append(relu_cost_norm)

            perc_layer_losses_raw = torch.stack(perc_layer_losses_raw, dim=0)  # (n_layer)
            perc_layer_losses_norm = torch.stack(perc_layer_losses_norm, dim=0)  # (n_layer)

            if self.hps.perc_loss_fuse_type == 'max':
                ras_cost = torch.max(perc_layer_losses_norm)
            elif self.hps.perc_loss_fuse_type == 'add':
                ras_cost = torch.mean(perc_layer_losses_norm)
            elif self.hps.perc_loss_fuse_type == 'raw_add':
                ras_cost = torch.mean(perc_layer_losses_raw)
            else:
                raise NameError('Unknown perc_loss_fuse_type:', self.hps.perc_loss_fuse_type)
        else:
            raise NameError('Unknown loss type:', loss_type)

        return ras_cost, perc_layer_losses_raw, perc_layer_losses_norm

    def save_log_images(self, data_set, save_root, step_num, save_num=20):
        batch_num = save_num // data_set.batch_size

        with torch.no_grad():
            for batch_i in range(batch_num):
                reference_images, reference_components, reference_strokes, target_images, target_components, \
                    reference_centerpoints, target_centerpoints, base_window_size, image_ids, endpoint_ids, \
                    component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
                    target_transform_shear_x_angles, target_transform_shear_y_angles = \
                    data_set.get_batch(self.use_cuda, batch_idx=batch_i, all_example=False)
                # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_components: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
                # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                # target_components: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
                # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
                # base_window_size: (N, 1), in [0.0, 1.0]

                # component_centerpoints: (N, 1, 2), in [0.0, 1.0], relative to image size
                # component_win_sizes: (N, 1, 2), in image size
                # target_transform_cursors: (N, 1, 2), in [0.0, 1.0], relative to image size
                # target_transform_win_sizes: (N, 1, 2), in image size
                # target_transform_angles: (N, 1), in [-180.0, 180.0]
                # target_transform_shear_x_angles: (N, 1), in [-90.0, 90.0]
                # target_transform_shear_y_angles: (N, 1), in [-90.0, 90.0]

                image_size = reference_images.size()[1]

                assert len(image_ids) == 1
                img_index = image_ids[0]

                patch_reference, patch_stroke_reference, patch_component_reference, \
                    patch_target_trans0, patch_component_target_trans0, \
                    patch_target_trans1, patch_component_target_trans1, _, _, _, _, _ = \
                    self.transform_local_model(reference_images=reference_images, reference_components=reference_components, reference_strokes=reference_strokes,
                                        target_images=target_images, target_components=target_components,
                                        centerpoints_pos_ref=reference_centerpoints,
                                        centerpoints_pos_tar=target_centerpoints,
                                        base_window_size=base_window_size,
                                        image_size=image_size,
                                        component_centerpoints=component_centerpoints,
                                        component_win_sizes=component_win_sizes,
                                        target_transform_cursors=target_transform_cursors,
                                        target_transform_win_sizes=target_transform_win_sizes,
                                        target_transform_angles=target_transform_angles,
                                        target_transform_shear_x_angles=target_transform_shear_x_angles,
                                        target_transform_shear_y_angles=target_transform_shear_y_angles
                                        )
                # patch_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_stroke_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_component_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_target_trans0: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_component_target_trans0: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_target_trans1: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_component_target_trans1: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

                patch_reference_np = patch_reference.cpu().data.numpy()
                patch_stroke_reference_np = patch_stroke_reference.cpu().data.numpy()
                patch_component_reference_np = patch_component_reference.cpu().data.numpy()
                patch_target_trans0_np = patch_target_trans0.cpu().data.numpy()
                patch_component_target_trans0_np = patch_component_target_trans0.cpu().data.numpy()
                patch_target_trans1_np = patch_target_trans1.cpu().data.numpy()
                patch_component_target_trans1_np = patch_component_target_trans1.cpu().data.numpy()

                for p_i in range(patch_reference_np.shape[0]):
                    endpoint_id = endpoint_ids[p_i]

                    patch_reference_np_i = save_image(patch_reference_np[p_i], save_root, 'ref-' + str(img_index) + '-' + endpoint_id + '.png')
                    save_image(patch_stroke_reference_np[p_i], save_root, 'ref_stroke-' + str(img_index) + '-' + endpoint_id + '.png')
                    patch_component_reference_np_i = save_image(patch_component_reference_np[p_i], save_root, 'ref_comp-' + str(img_index) + '-' + endpoint_id + '.png')

                    save_image_overlap(patch_target_trans0_np[p_i], patch_reference_np_i, save_root,
                                       'tar_trans0-' + str(img_index) + '-' + endpoint_id + '.png')
                    save_image_overlap(patch_component_target_trans0_np[p_i], patch_component_reference_np_i, save_root,
                                       'tar_comp_trans0-' + str(img_index) + '-' + endpoint_id + '.png')

                    save_image_overlap(patch_target_trans1_np[p_i], patch_reference_np_i, save_root,
                                       'tar_trans1-' + str(img_index) + '-' + endpoint_id + '-step=' + str(step_num) + '.png')
                    save_image_overlap(patch_component_target_trans1_np[p_i], patch_component_reference_np_i, save_root,
                                       'tar_comp_trans1-' + str(img_index) + '-' + endpoint_id + '-step=' + str(step_num) + '.png')


    def save_model(self, step_num, save_root):
        if self.use_cuda:
            self.transform_local_model.cpu()

        save_dict = {}

        if self.hps.multi_gpu:
            model_state_dict = self.transform_local_model.module.state_dict()
        else:
            model_state_dict = self.transform_local_model.state_dict()
        # print('model_state_dict')
        # print(model_state_dict.keys())

        save_dict.update(model_state_dict)

        save_path = os.path.join(save_root, "sketch_transform_local_" + str(step_num) + ".pkl")
        torch.save(save_dict, save_path)
        print('Saved model:', save_path)
        if self.use_cuda:
            self.transform_local_model.cuda()

    def evaluate(self, load_trained_weights=False, occluded_only=False, test_max_batch_size=50):
        print('-' * 100)
        print('Evaluation begins ...')

        if load_trained_weights:
            print('-' * 100)
            trained_transform_local_model_path = os.path.join(self.snapshot_dir, "sketch_transform_local_" + str(self.hps.num_steps) + ".pkl")
            if self.hps.multi_gpu:
                load_weights(trained_transform_local_model_path, self.transform_local_model.module)
                load_weights(self.hps.perceptual_model_path, self.perceptual_model.module)
            else:
                load_weights(trained_transform_local_model_path, self.transform_local_model)
                load_weights(self.hps.perceptual_model_path, self.perceptual_model)
            print('-' * 100)

            if self.use_cuda:
                self.transform_local_model = self.transform_local_model.cuda()
                self.perceptual_model = self.perceptual_model.cuda()

        self.transform_local_model.eval()
        self.perceptual_model.eval()

        self.valid_set.batch_size = 1
        batch_num = self.valid_set.example_num // self.valid_set.batch_size
        print('batch_num:', batch_num)

        perc_loss_layers_eval = self.hps.perc_loss_layer_eval

        perc_score_set = {}
        for perc_layer in perc_loss_layers_eval:
            perc_score_set[perc_layer] = []
        total_endpoint_num = 0

        with (torch.no_grad()):
            for batch_i in range(batch_num):
                print('# batch_i', batch_i)
                batch_data = self.valid_set.get_batch(self.use_cuda, batch_idx=batch_i, all_example=True, occluded_only=occluded_only)
                if batch_data is None:
                    continue

                reference_images, reference_components, reference_strokes, target_images, target_components, \
                    reference_centerpoints, target_centerpoints, base_window_size, _, _, \
                    component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
                    target_transform_shear_x_angles, target_transform_shear_y_angles = batch_data
                # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_components: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
                # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                # target_components: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
                # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
                # base_window_size: (N, 1), in [0.0, 1.0]

                # component_centerpoints: (N, 1, 2), in [0.0, 1.0], relative to image size
                # component_win_sizes: (N, 1, 2), in image size
                # target_transform_cursors: (N, 1, 2), in [0.0, 1.0], relative to image size
                # target_transform_win_sizes: (N, 1, 2), in image size
                # target_transform_angles: (N, 1), in [-180.0, 180.0]
                # target_transform_shear_x_angles: (N, 1), in [-90.0, 90.0]
                # target_transform_shear_y_angles: (N, 1), in [-90.0, 90.0]

                image_size = reference_images.size()[1]

                endpoint_num = reference_images.size()[0]
                total_endpoint_num += endpoint_num

                chunks = endpoint_num // test_max_batch_size + 1

                reference_images_chunks = torch.chunk(reference_images, chunks, dim=0)
                reference_components_chunks = torch.chunk(reference_components, chunks, dim=0)
                reference_strokes_chunks = torch.chunk(reference_strokes, chunks, dim=0)
                target_images_chunks = torch.chunk(target_images, chunks, dim=0)
                target_components_chunks = torch.chunk(target_components, chunks, dim=0)
                reference_centerpoints_chunks = torch.chunk(reference_centerpoints, chunks, dim=0)
                target_centerpoints_chunks = torch.chunk(target_centerpoints, chunks, dim=0)
                base_window_size_chunks = torch.chunk(base_window_size, chunks, dim=0)
                component_centerpoints_chunks = torch.chunk(component_centerpoints, chunks, dim=0)
                component_win_sizes_chunks = torch.chunk(component_win_sizes, chunks, dim=0)
                target_transform_cursors_chunks = torch.chunk(target_transform_cursors, chunks, dim=0)
                target_transform_win_sizes_chunks = torch.chunk(target_transform_win_sizes, chunks, dim=0)
                target_transform_angles_chunks = torch.chunk(target_transform_angles, chunks, dim=0)
                target_transform_shear_x_angles_chunks = torch.chunk(target_transform_shear_x_angles, chunks, dim=0)
                target_transform_shear_y_angles_chunks = torch.chunk(target_transform_shear_y_angles, chunks, dim=0)

                for chunk_i in range(chunks):
                    _, _, patch_component_reference_ch, _, _, _, patch_component_target_trans1_ch, _, _, _, _, _ = \
                        self.transform_local_model(reference_images=reference_images_chunks[chunk_i],
                                            reference_components=reference_components_chunks[chunk_i],
                                            reference_strokes=reference_strokes_chunks[chunk_i],
                                            target_images=target_images_chunks[chunk_i],
                                            target_components=target_components_chunks[chunk_i],
                                            centerpoints_pos_ref=reference_centerpoints_chunks[chunk_i],
                                            centerpoints_pos_tar=target_centerpoints_chunks[chunk_i],
                                            base_window_size=base_window_size_chunks[chunk_i],
                                            image_size=image_size,
                                            component_centerpoints=component_centerpoints_chunks[chunk_i],
                                            component_win_sizes=component_win_sizes_chunks[chunk_i],
                                            target_transform_cursors=target_transform_cursors_chunks[chunk_i],
                                            target_transform_win_sizes=target_transform_win_sizes_chunks[chunk_i],
                                            target_transform_angles=target_transform_angles_chunks[chunk_i],
                                            target_transform_shear_x_angles=target_transform_shear_x_angles_chunks[chunk_i],
                                            target_transform_shear_y_angles=target_transform_shear_y_angles_chunks[chunk_i]
                                            )
                    # patch_component_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                    # patch_component_target_transformed: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

                    component_num_chunk = patch_component_reference_ch.shape[0]

                    perc_map_pred = self.perceptual_model(patch_component_target_trans1_ch)
                    perc_map_gt = self.perceptual_model(patch_component_reference_ch)

                    _, perc_relu_losses_raw, _ = \
                        self.get_raster_loss(0, patch_component_target_trans1_ch, patch_component_reference_ch,
                                             loss_type=self.hps.raster_loss_base_type,
                                             return_map_pred=perc_map_pred, return_map_gt=perc_map_gt,
                                             raster_perc_loss_layer=perc_loss_layers_eval,
                                             perc_loss_mean_list=[0.0 for _ in range(len(perc_loss_layers_eval))])
                    # perc_relu_losses_raw: (n_layer)

                    # Perceptual score (PS): use a single layer
                    for layer_i in range(len(perc_loss_layers_eval)):
                        perc_score = perc_relu_losses_raw[layer_i] * float(component_num_chunk)
                        perc_score = perc_score.cpu().data.numpy()
                        perc_score_set[perc_loss_layers_eval[layer_i]].append(perc_score)

            print('total_endpoint_num', total_endpoint_num)
            tf.get_logger().info('total_endpoint_num: ' + str(total_endpoint_num))
            for layer_i in range(len(perc_loss_layers_eval)):
                perc_score_avg = np.sum(perc_score_set[perc_loss_layers_eval[layer_i]]) / float(total_endpoint_num)
                print('Perceptual score (PS):', perc_loss_layers_eval[layer_i], ':', perc_score_avg * 100.0, 'e-2')
                tf.get_logger().info('Perceptual score (PS): ' + perc_loss_layers_eval[layer_i] + ': ' + str(perc_score_avg * 100.0) + ' e-2')

            print('snapshot_dir:', self.snapshot_dir)
            print('win size =', self.hps.window_size_scaling_ref)
            print('window_size_min =', self.hps.window_size_min)
            print('perc_loss_layer_eval =', self.hps.perc_loss_layer_eval)
            tf.get_logger().info('snapshot_dir: ' + self.snapshot_dir)
            tf.get_logger().info('win size = ' + str(self.hps.window_size_scaling_ref))
            tf.get_logger().info('window_size_min = ' + str(self.hps.window_size_min))
            tf.get_logger().info('perc_loss_layer_eval = ' + str(self.hps.perc_loss_layer_eval))

    def inference(self, save_root, show_data='selected', test_max_batch_size=50, do_inv=False):
        print('-' * 100)
        print('Inference begins ...')

        trained_transform_local_model_path = os.path.join(self.snapshot_dir, "sketch_transform_local_" + str(self.hps.num_steps) + ".pkl")
        if self.hps.multi_gpu:
            load_weights(trained_transform_local_model_path, self.transform_local_model.module)
        else:
            load_weights(trained_transform_local_model_path, self.transform_local_model)

        if self.use_cuda:
            self.transform_local_model = self.transform_local_model.cuda()

        self.transform_local_model.eval()

        if show_data == 'all':
            show_num = self.valid_set.example_num
            is_save_image = False
            batch_idx_offsets = [0]
            data_split = 'val'
            occluded_only = False
            stroke_fixing = True
        elif show_data == 'occluded':
            show_num = 50
            is_save_image = True
            batch_idx_offsets = [0, 737]
            data_split = 'val'
            occluded_only = True
            stroke_fixing = False
        elif show_data == 'selected':
            show_num = 20
            is_save_image = True
            batch_idx_offsets = [0, 737]
            data_split = 'val'
            occluded_only = False
            stroke_fixing = False
        else:
            raise Exception('Unknown show_data:', show_data)
        single_stroke_occlusion_threshold = 85.0

        batch_num = self.valid_set.example_num // self.valid_set.batch_size
        print('batch_num:', batch_num)

        with torch.no_grad():
            for batch_idx_offset in batch_idx_offsets:
                show_i = 0
                for batch_i in range(batch_num):
                    print('# batch_i', batch_i)
                    tf.get_logger().info('# batch_i: ' + str(batch_i))

                    batch_data = self.valid_set.get_batch(self.use_cuda, batch_idx=batch_i, all_example=True, batch_idx_offset=batch_idx_offset, occluded_only=occluded_only)
                    if batch_data is None:
                        continue

                    reference_images, reference_components, reference_strokes, target_images, target_components, target_layer_masks, \
                        reference_centerpoints, target_centerpoints, base_window_size, image_ids, endpoint_ids, \
                        component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
                        target_transform_shear_x_angles, target_transform_shear_y_angles = batch_data
                    # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                    # reference_components: (N, H, W, 1), [0-stroke, 1-BG]
                    # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
                    # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                    # target_components: (N, H, W, 1), [0-stroke, 1-BG]
                    # target_layer_masks: (N, H, W, 1), [0-FG, 1-BG]
                    # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
                    # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
                    # base_window_size: (N, 1), in [0.0, 1.0]

                    # component_centerpoints: (N, 1, 2), in [0.0, 1.0], relative to image size
                    # component_win_sizes: (N, 1, 2), in image size
                    # target_transform_cursors: (N, 1, 2), in [0.0, 1.0], relative to image size
                    # target_transform_win_sizes: (N, 1, 2), in image size
                    # target_transform_angles: (N, 1), in [-180.0, 180.0]
                    # target_transform_shear_x_angles: (N, 1), in [-90.0, 90.0]
                    # target_transform_shear_y_angles: (N, 1), in [-90.0, 90.0]

                    image_size = reference_images.size()[1]
                    if target_components is None:
                        target_components = reference_components
                        assert target_layer_masks is None
                        target_layer_masks = reference_components

                    assert len(image_ids) == 1
                    img_index = image_ids[0]
                    print(' >> img_index', img_index)
                    tf.get_logger().info(' >> img_index: ' + str(img_index))

                    stroke_num = reference_images.size()[0]
                    chunks = stroke_num // test_max_batch_size + 1

                    reference_images_chunks = torch.chunk(reference_images, chunks, dim=0)
                    reference_components_chunks = torch.chunk(reference_components, chunks, dim=0)
                    reference_strokes_chunks = torch.chunk(reference_strokes, chunks, dim=0)
                    target_images_chunks = torch.chunk(target_images, chunks, dim=0)
                    target_components_chunks = torch.chunk(target_components, chunks, dim=0)
                    target_layer_masks_chunks = torch.chunk(target_layer_masks, chunks, dim=0)
                    reference_centerpoints_chunks = torch.chunk(reference_centerpoints, chunks, dim=0)
                    target_centerpoints_chunks = torch.chunk(target_centerpoints, chunks, dim=0)
                    base_window_size_chunks = torch.chunk(base_window_size, chunks, dim=0)
                    component_centerpoints_chunks = torch.chunk(component_centerpoints, chunks, dim=0)
                    component_win_sizes_chunks = torch.chunk(component_win_sizes, chunks, dim=0)
                    target_transform_cursors_chunks = torch.chunk(target_transform_cursors, chunks, dim=0)
                    target_transform_win_sizes_chunks = torch.chunk(target_transform_win_sizes, chunks, dim=0)
                    target_transform_angles_chunks = torch.chunk(target_transform_angles, chunks, dim=0)
                    target_transform_shear_x_angles_chunks = torch.chunk(target_transform_shear_x_angles, chunks, dim=0)
                    target_transform_shear_y_angles_chunks = torch.chunk(target_transform_shear_y_angles, chunks, dim=0)

                    patch_reference_np = []
                    patch_stroke_reference_np = []
                    patch_stroke_prev_reference_np = []
                    patch_stroke_next_reference_np = []
                    patch_component_reference_np = []
                    patch_target_trans0_np = []
                    patch_component_target_trans0_np = []
                    patch_target_trans1_np = []
                    patch_component_target_trans1_np = []
                    patch_target_layer_mask_trans1_np = []

                    pred_translate_tar_np = []
                    pred_scaling_times_tar_np = []
                    pred_rotate_angle_tar_np = []
                    pred_shear_x_angle_tar_np = []
                    pred_shear_y_angle_tar_np = []

                    for chunk_i in range(chunks):
                        patch_reference_ch, patch_stroke_reference_ch, patch_stroke_prev_reference_ch, patch_stroke_next_reference_ch, \
                            patch_component_reference_ch, \
                            patch_target_trans0_ch, patch_component_target_trans0_ch, patch_target_layer_mask_trans0_ch, \
                            patch_target_trans1_ch, patch_component_target_trans1_ch, patch_target_layer_mask_trans1_ch, \
                            pred_translate_tar_ch, pred_scaling_times_tar_ch, pred_rotate_angle_tar_ch, \
                            pred_shear_x_angle_tar_ch, pred_shear_y_angle_tar_ch = \
                            self.transform_local_model(reference_images=reference_images_chunks[chunk_i],
                                                reference_components=reference_components_chunks[chunk_i],
                                                reference_strokes=reference_strokes_chunks[chunk_i],
                                                reference_strokes_prev=reference_strokes_chunks[chunk_i],
                                                reference_strokes_next=reference_strokes_chunks[chunk_i],
                                                target_images=target_images_chunks[chunk_i],
                                                target_components=target_components_chunks[chunk_i],
                                                target_layer_masks=target_layer_masks_chunks[chunk_i],
                                                centerpoints_pos_ref=reference_centerpoints_chunks[chunk_i],
                                                centerpoints_pos_tar=target_centerpoints_chunks[chunk_i],
                                                base_window_size=base_window_size_chunks[chunk_i],
                                                image_size=image_size,
                                                component_centerpoints=component_centerpoints_chunks[chunk_i],
                                                component_win_sizes=component_win_sizes_chunks[chunk_i],
                                                target_transform_cursors=target_transform_cursors_chunks[chunk_i],
                                                target_transform_win_sizes=target_transform_win_sizes_chunks[chunk_i],
                                                target_transform_angles=target_transform_angles_chunks[chunk_i],
                                                target_transform_shear_x_angles=target_transform_shear_x_angles_chunks[chunk_i],
                                                target_transform_shear_y_angles=target_transform_shear_y_angles_chunks[chunk_i]
                                                )
                        # patch_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                        # patch_stroke_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                        # patch_target_trans: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

                        # pred_translate_tar:(N, 1, 2), [-1.0, 1.0], relative to target trans0 window
                        # pred_scaling_times_tar: (N, 1, 2), [0.2, 2.0], relative to target trans0 window
                        # pred_rotate_angle_tar: (N, 1), [-180.0, 180.0]
                        # pred_shear_x_angle_tar / pred_shear_y_angle_tar: (N, 1), [-90.0, 90.0]

                        patch_reference_np_ch = patch_reference_ch.cpu().data.numpy()
                        patch_stroke_reference_np_ch = patch_stroke_reference_ch.cpu().data.numpy()
                        patch_stroke_prev_reference_np_ch = patch_stroke_prev_reference_ch.cpu().data.numpy()
                        patch_stroke_next_reference_np_ch = patch_stroke_next_reference_ch.cpu().data.numpy()
                        patch_component_reference_np_ch = patch_component_reference_ch.cpu().data.numpy()
                        patch_target_trans0_np_ch = patch_target_trans0_ch.cpu().data.numpy()
                        patch_component_target_trans0_np_ch = patch_component_target_trans0_ch.cpu().data.numpy()
                        patch_target_trans1_np_ch = patch_target_trans1_ch.cpu().data.numpy()
                        patch_component_target_trans1_np_ch = patch_component_target_trans1_ch.cpu().data.numpy()
                        patch_target_layer_mask_trans1_np_ch = 1.0 - patch_target_layer_mask_trans1_ch.cpu().data.numpy()

                        pred_translate_tar_ch = pred_translate_tar_ch.squeeze(dim=1).cpu().data.numpy()
                        pred_scaling_times_tar_ch = pred_scaling_times_tar_ch.squeeze(dim=1).cpu().data.numpy()
                        pred_rotate_angle_tar_ch = pred_rotate_angle_tar_ch.squeeze(dim=1).cpu().data.numpy()
                        pred_shear_x_angle_tar_ch = pred_shear_x_angle_tar_ch.squeeze(dim=1).cpu().data.numpy()
                        pred_shear_y_angle_tar_ch = pred_shear_y_angle_tar_ch.squeeze(dim=1).cpu().data.numpy()

                        patch_reference_np.append(patch_reference_np_ch)
                        patch_stroke_reference_np.append(patch_stroke_reference_np_ch)
                        patch_stroke_prev_reference_np.append(patch_stroke_prev_reference_np_ch)
                        patch_stroke_next_reference_np.append(patch_stroke_next_reference_np_ch)
                        patch_component_reference_np.append(patch_component_reference_np_ch)
                        patch_target_trans0_np.append(patch_target_trans0_np_ch)
                        patch_component_target_trans0_np.append(patch_component_target_trans0_np_ch)
                        patch_target_trans1_np.append(patch_target_trans1_np_ch)
                        patch_component_target_trans1_np.append(patch_component_target_trans1_np_ch)
                        patch_target_layer_mask_trans1_np.append(patch_target_layer_mask_trans1_np_ch)

                        pred_translate_tar_np.append(pred_translate_tar_ch)
                        pred_scaling_times_tar_np.append(pred_scaling_times_tar_ch)
                        pred_rotate_angle_tar_np.append(pred_rotate_angle_tar_ch)
                        pred_shear_x_angle_tar_np.append(pred_shear_x_angle_tar_ch)
                        pred_shear_y_angle_tar_np.append(pred_shear_y_angle_tar_ch)

                    patch_reference_np = np.concatenate(patch_reference_np, axis=0)
                    patch_stroke_reference_np = np.concatenate(patch_stroke_reference_np, axis=0)
                    patch_stroke_prev_reference_np = np.concatenate(patch_stroke_prev_reference_np, axis=0)
                    patch_stroke_next_reference_np = np.concatenate(patch_stroke_next_reference_np, axis=0)
                    patch_component_reference_np = np.concatenate(patch_component_reference_np, axis=0)
                    patch_target_trans0_np = np.concatenate(patch_target_trans0_np, axis=0)
                    patch_component_target_trans0_np = np.concatenate(patch_component_target_trans0_np, axis=0)
                    patch_target_trans1_np = np.concatenate(patch_target_trans1_np, axis=0)
                    patch_component_target_trans1_np = np.concatenate(patch_component_target_trans1_np, axis=0)
                    patch_target_layer_mask_trans1_np = np.concatenate(patch_target_layer_mask_trans1_np, axis=0)

                    pred_translate_tar_np = np.concatenate(pred_translate_tar_np, axis=0)
                    pred_scaling_times_tar_np = np.concatenate(pred_scaling_times_tar_np, axis=0)
                    pred_rotate_angle_tar_np = np.concatenate(pred_rotate_angle_tar_np, axis=0)
                    pred_shear_x_angle_tar_np = np.concatenate(pred_shear_x_angle_tar_np, axis=0)
                    pred_shear_y_angle_tar_np = np.concatenate(pred_shear_y_angle_tar_np, axis=0)

                    stroke_visible_pixel_percentages = {}
                    assert len(endpoint_ids) == patch_reference_np.shape[0]
                    for p_i in range(patch_reference_np.shape[0]):
                        endpoint_id = endpoint_ids[p_i]

                        if stroke_fixing:
                            if self.hps.use_target_layer_mask == 'stroke':
                                ref_stroke_prev_patch_i = 1.0 - patch_stroke_prev_reference_np[p_i]  # (raster_size, raster_size), [0.0-BG, 1.0-stroke]
                                ref_stroke_next_patch_i = 1.0 - patch_stroke_next_reference_np[p_i]  # (raster_size, raster_size), [0.0-BG, 1.0-stroke]

                                tar_layer_mask_trans1_patch_i = patch_target_layer_mask_trans1_np[p_i]  # (raster_size, raster_size), [0.0-BG, 1.0-FG]
                                visible_percentage_prev_i = np.sum(ref_stroke_prev_patch_i * tar_layer_mask_trans1_patch_i) / np.sum(ref_stroke_prev_patch_i)
                                visible_percentage_next_i = np.sum(ref_stroke_next_patch_i * tar_layer_mask_trans1_patch_i) / np.sum(ref_stroke_next_patch_i)
                                stroke_visible_pixel_percentages[endpoint_id] = [visible_percentage_prev_i, visible_percentage_next_i]

                        if is_save_image:
                            patch_reference_np_i = save_image(patch_reference_np[p_i], save_root, 'ref-' + str(img_index) + '-' + endpoint_id + '.png')
                            save_image(patch_stroke_reference_np[p_i], save_root, 'ref_stroke-' + str(img_index) + '-' + endpoint_id + '.png')
                            patch_component_reference_np_i = save_image(patch_component_reference_np[p_i], save_root,
                                                                        'ref_comp-' + str(img_index) + '-' + endpoint_id + '.png')

                            save_image_overlap(patch_target_trans0_np[p_i], patch_reference_np_i, save_root,
                                               'tar_trans0-' + str(img_index) + '-' + endpoint_id + '.png')
                            save_image_overlap(patch_component_target_trans0_np[p_i], patch_component_reference_np_i,
                                               save_root, 'tar_comp_trans0-' + str(img_index) + '-' + endpoint_id + '.png')

                            save_image_overlap(patch_target_trans1_np[p_i], patch_reference_np_i, save_root,
                                               'tar_trans1-' + str(img_index) + '-' + endpoint_id + '.png')
                            save_image_overlap(patch_component_target_trans1_np[p_i], patch_component_reference_np_i,
                                               save_root, 'tar_comp_trans1-' + str(img_index) + '-' + endpoint_id + '.png')

                        if not is_save_image:
                            transform_models_name_plus = '[' + self.hps.transform_global_model_name + ']' + '-[c_min=' + str(self.hps.window_size_min_comp) + ']-[' + self.hps.workspace + ']'
                            if self.hps.use_optical_flow:
                                transform_models_name_plus += '-[optical]'
                            selected_dataset_name = img_index[:img_index.find('-')]
                            selected_img_index = img_index[img_index.find('-') + 1:]
                            transform_params_save_base = os.path.join(self.hps.dataset_base, selected_dataset_name + '_512', data_split,
                                                                        'component_local_transform_params')
                            transform_params_save_base = os.path.join(transform_params_save_base, transform_models_name_plus, str(selected_img_index))
                            os.makedirs(transform_params_save_base, exist_ok=True)
                            transform_params_save_path = os.path.join(transform_params_save_base, endpoint_id + '.jsonl')
                            if os.path.exists(transform_params_save_path):
                                os.remove(transform_params_save_path)

                            transform_params_data = {}
                            transform_params_data['pred_translate'] = pred_translate_tar_np[p_i].tolist()
                            transform_params_data['pred_scaling_times'] = pred_scaling_times_tar_np[p_i].tolist()
                            transform_params_data['pred_rotate_angle'] = pred_rotate_angle_tar_np[p_i].tolist()
                            transform_params_data['pred_shear_x_angle'] = pred_shear_x_angle_tar_np[p_i].tolist()
                            transform_params_data['pred_shear_y_angle'] = pred_shear_y_angle_tar_np[p_i].tolist()
                            with jsonlines.open(transform_params_save_path, mode='w') as json_writer:
                                json_writer.write(transform_params_data)

                    if stroke_fixing:
                        if self.hps.use_target_layer_mask == 'stroke':
                            occluded_params = {}
                            num_component = 0
                            for endpoint_id in stroke_visible_pixel_percentages.keys():
                                occluded_params[endpoint_id] = {"endpoint": None, "stroke": None}
                                comp_curve_point = endpoint_id.split('_')
                                c_i = comp_curve_point[0]
                                num_component = max(num_component, int(c_i) + 1)

                            comp_occluded_status = [True for _ in range(num_component)]
                            for endpoint_id in stroke_visible_pixel_percentages.keys():
                                comp_curve_point = endpoint_id.split('_')
                                c_i, curve_i, point_i = comp_curve_point
                                endpoint_id_next = '_'.join([c_i, curve_i, str(int(point_i) + 1)])

                                if endpoint_id_next not in stroke_visible_pixel_percentages.keys():  # the last endpoint
                                    continue

                                if do_inv:
                                    fix_state_stroke = True
                                else:
                                    stroke_occluded_pixel_percent_i_prev = (1.0 - stroke_visible_pixel_percentages[endpoint_id][1]) * 100.0
                                    stroke_occluded_pixel_percent_i_next = (1.0 - stroke_visible_pixel_percentages[endpoint_id_next][0]) * 100.0

                                    if stroke_occluded_pixel_percent_i_prev > single_stroke_occlusion_threshold and stroke_occluded_pixel_percent_i_next > single_stroke_occlusion_threshold:
                                        fix_state_stroke = True
                                    else:
                                        fix_state_stroke = False
                                        comp_occluded_status[int(c_i)] = False
                                    # print(endpoint_id, ':', stroke_occluded_pixel_percent_i_prev, ',',
                                    #       stroke_occluded_pixel_percent_i_next, ', fix:', fix_state_stroke)

                                occluded_params[endpoint_id]["stroke"] = fix_state_stroke

                                if not fix_state_stroke:
                                    occluded_params[endpoint_id]["endpoint"] = False
                                    occluded_params[endpoint_id_next]["endpoint"] = False
                                else:
                                    if occluded_params[endpoint_id]["endpoint"] is None:
                                        occluded_params[endpoint_id]["endpoint"] = True
                                    if occluded_params[endpoint_id_next]["endpoint"] is None:
                                        occluded_params[endpoint_id_next]["endpoint"] = True

                            if not do_inv:
                                for endpoint_id in stroke_visible_pixel_percentages.keys():
                                    assert occluded_params[endpoint_id]["endpoint"] is not None, endpoint_id

                                    comp_curve_point = endpoint_id.split('_')
                                    c_i = comp_curve_point[0]

                                    # for fully occluded layer, should not be fixing
                                    if comp_occluded_status[int(c_i)]:
                                        occluded_params[endpoint_id]["stroke"] = False
                                        occluded_params[endpoint_id]["endpoint"] = False
                        else:
                            assert self.hps.use_target_layer_mask == 'none'
                            continue

                        transform_models_name_plus = '[' + self.hps.transform_global_model_name + ']' + '-[c_min=' + str(
                            self.hps.window_size_min_comp) + ']-[' + self.hps.workspace + ']'
                        if self.hps.use_optical_flow:
                            transform_models_name_plus += '-[optical]'
                        occlusion_params_save_base = os.path.join(self.hps.dataset_base, selected_dataset_name + '_512', data_split, 'occlusion_params', self.hps.use_target_layer_mask)
                        occlusion_params_save_base = os.path.join(occlusion_params_save_base, transform_models_name_plus)
                        occluded_params_save_path = os.path.join(occlusion_params_save_base, str(selected_img_index) + '.jsonl')
                        os.makedirs(occlusion_params_save_base, exist_ok=True)
                        if os.path.exists(occluded_params_save_path):
                            os.remove(occluded_params_save_path)

                        with jsonlines.open(occluded_params_save_path, mode='w') as json_writer:
                            json_writer.write(occluded_params)

                    show_i += 1
                    if show_i >= show_num:
                        break

    def inference_real(self, save_root, data_base, test_max_batch_size=50, do_inv=False):
        # print('-' * 100)
        print('Inference of [Local Layer Transformation (2/4)] begins ...')

        trained_transform_local_model_path = os.path.join(self.snapshot_dir, "sketch_transform_local_" + str(self.hps.num_steps) + ".pkl")
        if self.hps.multi_gpu:
            load_weights(trained_transform_local_model_path, self.transform_local_model.module)
        else:
            load_weights(trained_transform_local_model_path, self.transform_local_model)

        if self.use_cuda:
            self.transform_local_model = self.transform_local_model.cuda()

        self.transform_local_model.eval()

        is_save_image = False
        single_curve_occlusion_threshold = 90.0
        multi_curve_occlusion_threshold = 70.0
        single_stroke_occlusion_threshold = 85.0

        with torch.no_grad():
            batch_data = self.valid_set.get_batch(self.use_cuda, test_img_id=test_img_id)

            reference_images, reference_components, reference_strokes, reference_strokes_prev, reference_strokes_next, \
                target_images, target_layer_masks, \
                reference_centerpoints, target_centerpoints, base_window_size, image_ids, endpoint_ids, \
                component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
                target_transform_shear_x_angles, target_transform_shear_y_angles = batch_data
            # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
            # reference_components: (N, H, W, 1), [0-stroke, 1-BG]
            # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
            # reference_strokes_prev: (N, H, W, 1), [0-stroke, 1-BG]
            # reference_strokes_next: (N, H, W, 1), [0-stroke, 1-BG]
            # target_images: (N, H, W, 1), [0-stroke, 1-BG]
            # target_layer_masks: (N, H, W, 1), [0-FG, 1-BG]
            # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
            # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
            # base_window_size: (N, 1), in [0.0, 1.0]

            # component_centerpoints: (N, 1, 2), in [0.0, 1.0], relative to image size
            # component_win_sizes: (N, 1, 2), in image size
            # target_transform_cursors: (N, 1, 2), in [0.0, 1.0], relative to image size
            # target_transform_win_sizes: (N, 1, 2), in image size
            # target_transform_angles: (N, 1), in [-180.0, 180.0]
            # target_transform_shear_x_angles: (N, 1), in [-90.0, 90.0]
            # target_transform_shear_y_angles: (N, 1), in [-90.0, 90.0]

            image_size = reference_images.size()[1]

            assert len(image_ids) == 1
            img_index = image_ids[0]
            # print(' >> img_index', img_index)

            stroke_num = reference_images.size()[0]
            chunks = stroke_num // test_max_batch_size + 1

            reference_images_chunks = torch.chunk(reference_images, chunks, dim=0)
            reference_components_chunks = torch.chunk(reference_components, chunks, dim=0)
            reference_strokes_chunks = torch.chunk(reference_strokes, chunks, dim=0)
            reference_strokes_prev_chunks = torch.chunk(reference_strokes_prev, chunks, dim=0)
            reference_strokes_next_chunks = torch.chunk(reference_strokes_next, chunks, dim=0)
            target_images_chunks = torch.chunk(target_images, chunks, dim=0)
            target_layer_masks_chunks = torch.chunk(target_layer_masks, chunks, dim=0)
            reference_centerpoints_chunks = torch.chunk(reference_centerpoints, chunks, dim=0)
            target_centerpoints_chunks = torch.chunk(target_centerpoints, chunks, dim=0)
            base_window_size_chunks = torch.chunk(base_window_size, chunks, dim=0)
            component_centerpoints_chunks = torch.chunk(component_centerpoints, chunks, dim=0)
            component_win_sizes_chunks = torch.chunk(component_win_sizes, chunks, dim=0)
            target_transform_cursors_chunks = torch.chunk(target_transform_cursors, chunks, dim=0)
            target_transform_win_sizes_chunks = torch.chunk(target_transform_win_sizes, chunks, dim=0)
            target_transform_angles_chunks = torch.chunk(target_transform_angles, chunks, dim=0)
            target_transform_shear_x_angles_chunks = torch.chunk(target_transform_shear_x_angles, chunks, dim=0)
            target_transform_shear_y_angles_chunks = torch.chunk(target_transform_shear_y_angles, chunks, dim=0)

            patch_reference_np = []
            patch_stroke_reference_np = []
            patch_stroke_prev_reference_np = []
            patch_stroke_next_reference_np = []
            patch_component_reference_np = []
            patch_target_trans0_np = []
            patch_target_trans1_np = []
            patch_target_layer_mask_trans0_np = []
            patch_target_layer_mask_trans1_np = []

            pred_translate_tar_np = []
            pred_scaling_times_tar_np = []
            pred_rotate_angle_tar_np = []
            pred_shear_x_angle_tar_np = []
            pred_shear_y_angle_tar_np = []

            for chunk_i in range(chunks):
                patch_reference_ch, patch_stroke_reference_ch, patch_stroke_prev_reference_ch, patch_stroke_next_reference_ch,\
                    patch_component_reference_ch, \
                    patch_target_trans0_ch, _, patch_target_layer_mask_trans0_ch, \
                    patch_target_trans1_ch, _, patch_target_layer_mask_trans1_ch, \
                    pred_translate_tar_ch, pred_scaling_times_tar_ch, pred_rotate_angle_tar_ch, \
                    pred_shear_x_angle_tar_ch, pred_shear_y_angle_tar_ch = \
                    self.transform_local_model(reference_images=reference_images_chunks[chunk_i],
                                        reference_components=reference_components_chunks[chunk_i],
                                        reference_strokes=reference_strokes_chunks[chunk_i],
                                        reference_strokes_prev=reference_strokes_prev_chunks[chunk_i],
                                        reference_strokes_next=reference_strokes_next_chunks[chunk_i],
                                        target_images=target_images_chunks[chunk_i],
                                        target_components=target_layer_masks_chunks[chunk_i],
                                        target_layer_masks=target_layer_masks_chunks[chunk_i],
                                        centerpoints_pos_ref=reference_centerpoints_chunks[chunk_i],
                                        centerpoints_pos_tar=target_centerpoints_chunks[chunk_i],
                                        base_window_size=base_window_size_chunks[chunk_i],
                                        image_size=image_size,
                                        component_centerpoints=component_centerpoints_chunks[chunk_i],
                                        component_win_sizes=component_win_sizes_chunks[chunk_i],
                                        target_transform_cursors=target_transform_cursors_chunks[chunk_i],
                                        target_transform_win_sizes=target_transform_win_sizes_chunks[chunk_i],
                                        target_transform_angles=target_transform_angles_chunks[chunk_i],
                                        target_transform_shear_x_angles=target_transform_shear_x_angles_chunks[chunk_i],
                                        target_transform_shear_y_angles=target_transform_shear_y_angles_chunks[chunk_i]
                                        )
                # patch_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_stroke_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_target_trans: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

                # pred_translate_tar:(N, 1, 2), [-1.0, 1.0], relative to target trans0 window
                # pred_scaling_times_tar: (N, 1, 2), [0.2, 2.0], relative to target trans0 window
                # pred_rotate_angle_tar: (N, 1), [-180.0, 180.0]
                # pred_shear_x_angle_tar / pred_shear_y_angle_tar: (N, 1), [-90.0, 90.0]

                patch_reference_np_ch = patch_reference_ch.cpu().data.numpy()
                patch_stroke_reference_np_ch = patch_stroke_reference_ch.cpu().data.numpy()
                patch_stroke_prev_reference_np_ch = patch_stroke_prev_reference_ch.cpu().data.numpy()
                patch_stroke_next_reference_np_ch = patch_stroke_next_reference_ch.cpu().data.numpy()
                patch_component_reference_np_ch = patch_component_reference_ch.cpu().data.numpy()
                patch_target_trans0_np_ch = patch_target_trans0_ch.cpu().data.numpy()
                patch_target_trans1_np_ch = patch_target_trans1_ch.cpu().data.numpy()
                patch_target_layer_mask_trans0_np_ch = 1.0 - patch_target_layer_mask_trans0_ch.cpu().data.numpy()
                patch_target_layer_mask_trans1_np_ch = 1.0 - patch_target_layer_mask_trans1_ch.cpu().data.numpy()

                pred_translate_tar_ch = pred_translate_tar_ch.squeeze(dim=1).cpu().data.numpy()
                pred_scaling_times_tar_ch = pred_scaling_times_tar_ch.squeeze(dim=1).cpu().data.numpy()
                pred_rotate_angle_tar_ch = pred_rotate_angle_tar_ch.squeeze(dim=1).cpu().data.numpy()
                pred_shear_x_angle_tar_ch = pred_shear_x_angle_tar_ch.squeeze(dim=1).cpu().data.numpy()
                pred_shear_y_angle_tar_ch = pred_shear_y_angle_tar_ch.squeeze(dim=1).cpu().data.numpy()

                patch_reference_np.append(patch_reference_np_ch)
                patch_stroke_reference_np.append(patch_stroke_reference_np_ch)
                patch_stroke_prev_reference_np.append(patch_stroke_prev_reference_np_ch)
                patch_stroke_next_reference_np.append(patch_stroke_next_reference_np_ch)
                patch_component_reference_np.append(patch_component_reference_np_ch)
                patch_target_trans0_np.append(patch_target_trans0_np_ch)
                patch_target_trans1_np.append(patch_target_trans1_np_ch)
                patch_target_layer_mask_trans0_np.append(patch_target_layer_mask_trans0_np_ch)
                patch_target_layer_mask_trans1_np.append(patch_target_layer_mask_trans1_np_ch)

                pred_translate_tar_np.append(pred_translate_tar_ch)
                pred_scaling_times_tar_np.append(pred_scaling_times_tar_ch)
                pred_rotate_angle_tar_np.append(pred_rotate_angle_tar_ch)
                pred_shear_x_angle_tar_np.append(pred_shear_x_angle_tar_ch)
                pred_shear_y_angle_tar_np.append(pred_shear_y_angle_tar_ch)

            patch_reference_np = np.concatenate(patch_reference_np, axis=0)
            patch_stroke_reference_np = np.concatenate(patch_stroke_reference_np, axis=0)
            patch_stroke_prev_reference_np = np.concatenate(patch_stroke_prev_reference_np, axis=0)
            patch_stroke_next_reference_np = np.concatenate(patch_stroke_next_reference_np, axis=0)
            patch_component_reference_np = np.concatenate(patch_component_reference_np, axis=0)
            patch_target_trans0_np = np.concatenate(patch_target_trans0_np, axis=0)
            patch_target_trans1_np = np.concatenate(patch_target_trans1_np, axis=0)
            patch_target_layer_mask_trans0_np = np.concatenate(patch_target_layer_mask_trans0_np, axis=0)
            patch_target_layer_mask_trans1_np = np.concatenate(patch_target_layer_mask_trans1_np, axis=0)

            pred_translate_tar_np = np.concatenate(pred_translate_tar_np, axis=0)
            pred_scaling_times_tar_np = np.concatenate(pred_scaling_times_tar_np, axis=0)
            pred_rotate_angle_tar_np = np.concatenate(pred_rotate_angle_tar_np, axis=0)
            pred_shear_x_angle_tar_np = np.concatenate(pred_shear_x_angle_tar_np, axis=0)
            pred_shear_y_angle_tar_np = np.concatenate(pred_shear_y_angle_tar_np, axis=0)

            reference_strokes_np = reference_strokes.cpu().data.numpy()  # (N, H, W, 1), [0-stroke, 1-BG]

            curve_all_pixel_num = {}
            curve_visible_pixel_num = {}
            stroke_visible_pixel_percentages = {}
            for p_i in range(patch_reference_np.shape[0]):
                endpoint_id = endpoint_ids[p_i]
                comp_curve_point = endpoint_id.split('_')

                if self.hps.use_target_layer_mask == 'stroke':
                    ref_stroke_prev_patch_i = 1.0 - patch_stroke_prev_reference_np[p_i]  # (raster_size, raster_size), [0.0-BG, 1.0-stroke]
                    ref_stroke_next_patch_i = 1.0 - patch_stroke_next_reference_np[p_i]  # (raster_size, raster_size), [0.0-BG, 1.0-stroke]

                    tar_layer_mask_trans1_patch_i = patch_target_layer_mask_trans1_np[p_i]  # (raster_size, raster_size), [0.0-BG, 1.0-FG]
                    visible_percentage_prev_i = np.sum(ref_stroke_prev_patch_i * tar_layer_mask_trans1_patch_i) / np.sum(ref_stroke_prev_patch_i)
                    visible_percentage_next_i = np.sum(ref_stroke_next_patch_i * tar_layer_mask_trans1_patch_i) / np.sum(ref_stroke_next_patch_i)
                    stroke_visible_pixel_percentages[endpoint_id] = [visible_percentage_prev_i, visible_percentage_next_i]

                if is_save_image:
                    os.makedirs(save_root, exist_ok=True)
                    patch_reference_np_i = save_image(patch_reference_np[p_i], save_root, 'ref-' + str(img_index) + '-' + endpoint_id + '.png')
                    save_image(patch_stroke_reference_np[p_i], save_root, 'ref_stroke-' + str(img_index) + '-' + endpoint_id + '.png')
                    save_image(patch_stroke_prev_reference_np[p_i], save_root, 'ref_stroke_prev-' + str(img_index) + '-' + endpoint_id + '.png')
                    save_image(patch_stroke_next_reference_np[p_i], save_root, 'ref_stroke_next-' + str(img_index) + '-' + endpoint_id + '.png')
                    patch_component_reference_np_i = save_image(patch_component_reference_np[p_i], save_root,
                                                                'ref_comp-' + str(img_index) + '-' + endpoint_id + '.png')

                    save_image_overlap(patch_target_trans0_np[p_i], patch_reference_np_i, save_root,
                                        'tar_trans0-' + str(img_index) + '-' + endpoint_id + '.png')

                    save_image_overlap(patch_target_trans1_np[p_i], patch_reference_np_i, save_root,
                                        'tar_trans1-' + str(img_index) + '-' + endpoint_id + '.png')

                    save_image(patch_target_layer_mask_trans0_np[p_i], save_root,
                                'tar_layer_mask_trans0-' + str(img_index) + '-' + endpoint_id + '.png')
                    save_image(patch_target_layer_mask_trans1_np[p_i], save_root,
                                'tar_layer_mask_trans1-' + str(img_index) + '-' + endpoint_id + '.png')

                if not is_save_image:
                    selected_img_index = str(test_img_id)
                    transform_params_save_base = os.path.join(data_base, 'component_local_transform_params')
                    transform_models_name_plus = '[' + self.hps.transform_global_model_name + ']' + '-[c_min=' + str(self.hps.window_size_min_comp) + ']-[' + self.hps.workspace + ']'
                    if self.hps.use_optical_flow:
                        transform_models_name_plus += '-[optical]'
                    transform_params_save_base = os.path.join(transform_params_save_base, transform_models_name_plus, str(selected_img_index))
                    os.makedirs(transform_params_save_base, exist_ok=True)
                    transform_params_save_path = os.path.join(transform_params_save_base, endpoint_id + '.jsonl')
                    if os.path.exists(transform_params_save_path):
                        os.remove(transform_params_save_path)

                    transform_params_data = {}
                    transform_params_data['pred_translate'] = pred_translate_tar_np[p_i].tolist()
                    transform_params_data['pred_scaling_times'] = pred_scaling_times_tar_np[p_i].tolist()
                    transform_params_data['pred_rotate_angle'] = pred_rotate_angle_tar_np[p_i].tolist()
                    transform_params_data['pred_shear_x_angle'] = pred_shear_x_angle_tar_np[p_i].tolist()
                    transform_params_data['pred_shear_y_angle'] = pred_shear_y_angle_tar_np[p_i].tolist()
                    with jsonlines.open(transform_params_save_path, mode='w') as json_writer:
                        json_writer.write(transform_params_data)

            if self.hps.use_target_layer_mask == 'stroke':
                occluded_params = {}
                num_component = 0
                for endpoint_id in stroke_visible_pixel_percentages.keys():
                    occluded_params[endpoint_id] = {"endpoint": None, "stroke": None}
                    comp_curve_point = endpoint_id.split('_')
                    c_i = comp_curve_point[0]
                    num_component = max(num_component, int(c_i) + 1)

                comp_occluded_status = [True for _ in range(num_component)]
                for endpoint_id in stroke_visible_pixel_percentages.keys():
                    comp_curve_point = endpoint_id.split('_')
                    c_i, curve_i, point_i = comp_curve_point
                    endpoint_id_next = '_'.join([c_i, curve_i, str(int(point_i) + 1)])

                    if endpoint_id_next not in stroke_visible_pixel_percentages.keys():  # the last endpoint
                        continue

                    if do_inv:
                        fix_state_stroke = True
                    else:
                        stroke_occluded_pixel_percent_i_prev = (1.0 - stroke_visible_pixel_percentages[endpoint_id][1]) * 100.0
                        stroke_occluded_pixel_percent_i_next = (1.0 - stroke_visible_pixel_percentages[endpoint_id_next][0]) * 100.0

                        if stroke_occluded_pixel_percent_i_prev > single_stroke_occlusion_threshold and stroke_occluded_pixel_percent_i_next > single_stroke_occlusion_threshold:
                            fix_state_stroke = True
                        else:
                            fix_state_stroke = False
                            comp_occluded_status[int(c_i)] = False
                        # print(endpoint_id, ':', stroke_occluded_pixel_percent_i_prev, ',', stroke_occluded_pixel_percent_i_next, ', fix:', fix_state_stroke)

                    occluded_params[endpoint_id]["stroke"] = fix_state_stroke

                    if not fix_state_stroke:
                        occluded_params[endpoint_id]["endpoint"] = False
                        occluded_params[endpoint_id_next]["endpoint"] = False
                    else:
                        if occluded_params[endpoint_id]["endpoint"] is None:
                            occluded_params[endpoint_id]["endpoint"] = True
                        if occluded_params[endpoint_id_next]["endpoint"] is None:
                            occluded_params[endpoint_id_next]["endpoint"] = True

                if not do_inv:
                    for endpoint_id in stroke_visible_pixel_percentages.keys():
                        assert occluded_params[endpoint_id]["endpoint"] is not None, endpoint_id

                        comp_curve_point = endpoint_id.split('_')
                        c_i = comp_curve_point[0]

                        # for fully occluded layer, should not be fixing
                        if comp_occluded_status[int(c_i)]:
                            occluded_params[endpoint_id]["stroke"] = False
                            occluded_params[endpoint_id]["endpoint"] = False
            else:
                assert self.hps.use_target_layer_mask == 'none'

            selected_img_index = str(test_img_id)
            occlusion_params_save_base = os.path.join(data_base, 'occlusion_params', self.hps.use_target_layer_mask)
            transform_models_name_plus = '[' + self.hps.transform_global_model_name + ']' + '-[c_min=' + str(self.hps.window_size_min_comp) + ']-[' + self.hps.workspace + ']'
            if self.hps.use_optical_flow:
                transform_models_name_plus += '-[optical]'
            occlusion_params_save_base = os.path.join(occlusion_params_save_base, transform_models_name_plus)
            os.makedirs(occlusion_params_save_base, exist_ok=True)
            occluded_params_save_path = os.path.join(occlusion_params_save_base, str(selected_img_index) + '.jsonl')
            if os.path.exists(occluded_params_save_path):
                os.remove(occluded_params_save_path)

            with jsonlines.open(occluded_params_save_path, mode='w') as json_writer:
                json_writer.write(occluded_params)