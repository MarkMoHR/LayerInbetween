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
from image_utils.image_processing import save_image, save_image_overlap, draw_heatmap, draw_dot_full
from image_utils.model_processing import get_coordconv, add_coords, normalize_image_m1to1, image_cropping_stn, image_cropping_stn_multi, \
    load_weights, print_model_variables, spatial_transform_reverse_point_with_additional, spatial_transform_point_with_additional
from network.vanilla import CNN_Encoder, MLP_Decoder, CNN_SepEncoder
from configs.example_configs import test_img_id

tf.get_logger().setLevel('INFO')


def get_default_hparams():
    """Return default HParams for sketch-rnn."""
    hparams = HParams(
        workspace='FAD3-EPO35-1.5x-min=64',
        transform_model_name='FAD3-T12-2.0x-51-min=64',  # 'FAD-2.0x-51' / 'FAD2-2.0x-51-v3' / 'FAD3-2.0x-51-min=64' / 'FAD3-T12-2.0x-51-min=64'
        transform_local_model_name='FAD3-T13-2.0x-51',  # FAD3-T13-2.0x-51 / FAD3-T13-1.5x-51

        ############ For inference only ############
        do_dataset_filtering=True,  # set to True for training
        ############################################

        use_local_transform=True,

        dataset_base='/home/Datasets/CreativeSketch/proc_data3/',
        use_real_endpoints=False,  # whether use endpoints of curves only

        multi_gpu=False,

        num_steps=30000,
        save_every=10000,  # Number of steps per checkpoint creation.
        log_img_every=500,  # Number of steps per log image creation.

        batch_size=25,

        # image_size=512,

        enc_model_endpoint='combined',  # ['combined', 'separated']
        dec_model_endpoint='mlp',  # ['rnn', 'mlp']
        z_size=256,  # Size of latent vector z.
        endpoint_module_zero_init='last',  # ['none', 'last', 'all']
        add_coordconv=True,

        vector_loss_type='L1',  # ['MSE', 'L1']

        grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.

        learning_rate=1e-4,  # Learning rate.
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        decay_power=0.9,
        min_learning_rate=1e-6,  # Minimum learning rate.

        snapshot_root='outputs/endpoint/snapshot',
        log_img_root='outputs/endpoint/log_img',
        log_root='outputs/endpoint/log',
        inference_root='outputs/endpoint/inference_trans-FAD3',
        inference_full_root='outputs/endpoint/inference_FULL_trans-FAD3',
        inference_full_real_root='outputs/endpoint/inference_FULL_trans-Real',
    )
    return hparams


class Endpoint_Model(nn.Module):
    def __init__(self, hps):
        super(Endpoint_Model, self).__init__()
        self.hps = hps

        endpoint_out_size = 2  # offset
        cnn_out_size = self.hps.z_size

        # endpoint encoder
        if self.hps.enc_model_endpoint == 'combined':
            cnn_in_size = 3
            if self.hps.add_coordconv:
                cnn_in_size += 2
            self.encoder_endpoint = CNN_Encoder(cnn_in_size, cnn_out_size, input_size=self.hps.raster_size)
        elif self.hps.enc_model_endpoint == 'separated':
            cnn_in_size_ref = 2
            cnn_in_size_tar = 1
            if self.hps.add_coordconv:
                cnn_in_size_ref += 2
                cnn_in_size_tar += 2
            self.encoder_endpoint = CNN_SepEncoder(cnn_in_size_ref, cnn_in_size_tar, cnn_out_size, input_size=self.hps.raster_size)
        else:
            raise Exception('Unknown enc_model_endpoint:', self.hps.enc_model_endpoint)

        dec_in_size = self.hps.z_size
        if self.hps.dec_model_endpoint == 'mlp':
            self.decoder_endpoint = MLP_Decoder(dec_in_size, endpoint_out_size, zero_init=self.hps.endpoint_module_zero_init)
        else:
            raise Exception('Unknown dec_model_endpoint:', self.hps.dec_model_endpoint)

        if self.hps.add_coordconv:
            self.coordconv_input = get_coordconv(self.hps.raster_size)  # (2, raster_size, raster_size)

    def forward(self, reference_images, reference_components, reference_strokes, target_images,
                centerpoints_pos_ref, centerpoints_pos_tar, base_window_size, image_size,
                endpoint_offset_gt,
                component_centerpoints, component_win_sizes,
                target_transform_cursors, target_transform_win_sizes, target_transform_angles,
                target_transform_shear_x_angles, target_transform_shear_y_angles,
                target_transform1_translate, target_transform1_scaling,
                target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y,
                fixing_states
                ):
        """
        :param reference_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_components: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_strokes: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param centerpoints_pos_ref: (N, 1, 2), float32, in [0.0, 1.0]
        :param centerpoints_pos_tar: (N, 1, 2), float32, in [0.0, 1.0]
        :param base_window_size: (N, 1), float32, in [0.0, 1.0]
        :param endpoint_offset_gt: (N, 1, 2), float32, [-1.0, 1.0]
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

        patch_reference, patch_component_reference, patch_stroke_reference, patch_target_ori, patch_target_trans0, patch_target_trans1, \
            endpoint_offset, endpoint_offset_trans, endpoint_gt_offset_trans = \
            self.get_points_and_raster_image(reference_images, reference_components, reference_strokes, target_images,
                                             centerpoints_pos_ref, centerpoints_pos_tar,
                                             base_window_size,
                                             endpoint_offset_gt,
                                             component_centerpoints, component_win_sizes,
                                             target_transform_cursors, target_transform_win_sizes, target_transform_angles,
                                             target_transform_shear_x_angles, target_transform_shear_y_angles,
                                             target_transform1_translate, target_transform1_scaling,
                                             target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y,
                                             fixing_states
                                             )
        # patch_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_component_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_stroke_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_target_ori: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_target_trans0: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_target_trans1: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # endpoint_offset: (N, 2), [-1.0, 1.0]
        # endpoint_offset_trans: (N, 2), [-1.0, 1.0]
        # endpoint_gt_offset_trans: (N, 2), [-1.0, 1.0]

        return patch_reference, patch_component_reference, patch_stroke_reference, patch_target_ori, patch_target_trans0, patch_target_trans1, \
            endpoint_offset, endpoint_offset_trans, endpoint_gt_offset_trans

    def get_points_and_raster_image(self, reference_images, reference_components, reference_strokes, target_images,
                                    centerpoints_pos_ref, centerpoints_pos_tar,
                                    base_window_size,
                                    endpoint_offset_gt,
                                    component_centerpoints, component_win_sizes,
                                    target_transform_cursors, target_transform_win_sizes, target_transform_angles,
                                    target_transform_shear_x_angles, target_transform_shear_y_angles,
                                    target_transform1_translate, target_transform1_scaling,
                                    target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y,
                                    fixing_states):
        """
        :param reference_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_components: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_strokes: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param centerpoints_pos_ref: (N, 1, 2), float32, in [0.0, 1.0]
        :param centerpoints_pos_tar: (N, 1, 2), float32, in [0.0, 1.0]
        :param base_window_size: (N, 1), float32, in [0.0, 1.0]
        :param endpoint_offset_gt: (N, 1, 2), float32, [-1.0, 1.0]
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
        crop_inputs_ref = torch.cat([reference_images, reference_strokes, reference_components], dim=-1)  # (N, H, W, *)
        cropped_outputs = image_cropping_stn(cursor_position_loop_ref, crop_inputs_ref, self.image_size, self.hps.raster_size, curr_window_size)

        curr_patch_image_ref = cropped_outputs[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_stroke_ref = cropped_outputs[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_component_ref = cropped_outputs[:, :, :, 2:3]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]

        curr_patch_image_ref_out = torch.squeeze(curr_patch_image_ref, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_stroke_ref_out = torch.squeeze(curr_patch_stroke_ref, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_component_ref_out = torch.squeeze(curr_patch_component_ref, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        curr_patch_image_ref = normalize_image_m1to1(curr_patch_image_ref)
        curr_patch_stroke_ref = normalize_image_m1to1(curr_patch_stroke_ref)
        # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

        ## target_images: (N, H, W, 1), [0.0-stroke, 1.0-BG]
        crop_inputs_tar = target_images  # (N, H, W, *)

        # crop without transform
        cropped_outputs = image_cropping_stn(cursor_position_loop_tar, crop_inputs_tar, self.image_size, self.hps.raster_size, curr_window_size)
        curr_patch_image_tar_ori = cropped_outputs[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_image_tar_ori_out = torch.squeeze(curr_patch_image_tar_ori, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        # crop with transform0
        additional_offset = (cursor_position_loop_ref - component_centerpoints) * self.image_size / (component_win_sizes / 2.0)  # (N, 1, 2), [-1, 1]
        additional_scale = curr_window_size / component_win_sizes  # (N, 1, 2), [0, 1+]
        cropped_outputs = image_cropping_stn(target_transform_cursors, crop_inputs_tar, self.image_size, self.hps.raster_size, target_transform_win_sizes,
                                             rotation_angle=target_transform_angles,
                                             shear_x_angle=target_transform_shear_x_angles, shear_y_angle=target_transform_shear_y_angles,
                                             additional_transform=True, addi_offset=additional_offset, addi_scale=additional_scale)
        curr_patch_image_tar_trans0_temp = cropped_outputs[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_image_tar_trans0_out = torch.squeeze(curr_patch_image_tar_trans0_temp, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_image_tar_trans0 = normalize_image_m1to1(curr_patch_image_tar_trans0_temp)

        # crop with transform1
        curr_window_size_local_trans = base_window_size.unsqueeze(dim=-1)  # (N, 1, 1), in [0.0, 1.0]
        curr_window_size_local_trans = torch.mul(curr_window_size_local_trans, self.image_size)  # (N, 1, 1), in full size
        curr_window_size_local_trans = torch.mul(curr_window_size_local_trans, self.hps.window_size_scaling_ref_comp_local)  # (N, 1, 1), in full size
        curr_window_size_local_trans = torch.max(curr_window_size_local_trans, torch.tensor(self.hps.window_size_min).float().cuda())
        curr_window_size_local_trans = torch.min(curr_window_size_local_trans, torch.tensor(self.image_size * 1.5).float().cuda())
        curr_window_size_local_trans = torch.cat([curr_window_size_local_trans, curr_window_size_local_trans], dim=-1)  # (N, 1, 2), in full size

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
        curr_patch_image_tar_trans1 = normalize_image_m1to1(curr_patch_image_tar_trans1_temp)
        # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

        if self.hps.use_local_transform:
            encoded_z = self.build_encoder_endpoint(curr_patch_image_ref, curr_patch_stroke_ref, curr_patch_image_tar_trans1)  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        else:
            encoded_z = self.build_encoder_endpoint(curr_patch_image_ref, curr_patch_stroke_ref, curr_patch_image_tar_trans0)  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        endpoint_offset_trans = self.build_decoder_endpoint(encoded_z)
        # endpoint_offset_trans: (N, 2), [-1.0, 1.0]

        ## Fixing endpoints of fully occluded curves
        endpoint_offset_trans = endpoint_offset_trans * (1.0 - fixing_states)  # (N, 2), [-1.0, 1.0]

        if self.hps.use_local_transform:
            additional_scale_trans0 = additional_scale_local_trans
        else:
            additional_scale_trans0 = additional_scale

        ## inverse transform 1 -> 0
        if self.hps.use_local_transform:
            target_transform_cursors0 = target_transform_cursors * float(self.image_size) + additional_offset * (component_win_sizes / 2.0)  # (N, 1, 2), in image size
            target_transform_win_sizes0 = target_transform_win_sizes * additional_scale_local_trans  # (N, 1, 2), in image size
            target_transform_cursors1 = target_transform_cursors0 + (target_transform1_translate * target_transform_win_sizes0 / 2.0)   # (N, 1, 2), in image size
            target_transform_win_sizes1 = target_transform_win_sizes0 * target_transform1_scaling  # (N, 1, 2), in image size

            endpoint_offset_trans0 = spatial_transform_reverse_point_with_additional(
                endpoint_offset_trans.unsqueeze(dim=1), target_transform_win_sizes0, target_transform_cursors0,
                target_transform_cursors1, target_transform_win_sizes1,
                target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y,
                torch.zeros_like(additional_offset), additional_scale_stroke
            )  # (N, 1, 2), [-1.0, 1.0]
        else:
            endpoint_offset_trans0 = endpoint_offset_trans.unsqueeze(dim=1)  # (N, 1, 2), [-1.0, 1.0]

        ## inverse transform 0 -> ori
        endpoint_offset = spatial_transform_reverse_point_with_additional(
            endpoint_offset_trans0, curr_window_size, cursor_position_loop_tar * float(self.image_size),
            target_transform_cursors * float(self.image_size), target_transform_win_sizes,
            target_transform_angles, target_transform_shear_x_angles, target_transform_shear_y_angles,
            additional_offset, additional_scale_trans0
        )  # (N, 1, 2), [-1.0, 1.0]
        endpoint_offset = endpoint_offset.squeeze(dim=1)  # (N, 2), [-1.0, 1.0]

        ## Convert gt_offset in normal coordinate to transformed coordinate
        # ori -> trans0
        endpoint_gt_offset_trans0 = spatial_transform_point_with_additional(endpoint_offset_gt, curr_window_size, cursor_position_loop_tar * float(self.image_size),
                                                                            target_transform_cursors * float(self.image_size), target_transform_win_sizes,
                                                                            target_transform_angles, target_transform_shear_x_angles, target_transform_shear_y_angles,
                                                                            additional_offset, additional_scale_trans0)  # (N, 1, 2), [-1.0, 1.0]

        # trans0 -> trans1
        if self.hps.use_local_transform:
            endpoint_gt_offset_trans = spatial_transform_point_with_additional(endpoint_gt_offset_trans0, target_transform_win_sizes0, target_transform_cursors0,
                                                                               target_transform_cursors1, target_transform_win_sizes1,
                                                                               target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y,
                                                                               torch.zeros_like(additional_offset), additional_scale_stroke)
            endpoint_gt_offset_trans = endpoint_gt_offset_trans.squeeze(dim=1)  # (N, 2), [-1.0, 1.0]
        else:
            endpoint_gt_offset_trans = endpoint_gt_offset_trans0.squeeze(dim=1)

        return curr_patch_image_ref_out, curr_patch_component_ref_out, curr_patch_stroke_ref_out, \
               curr_patch_image_tar_ori_out, curr_patch_image_tar_trans0_out, curr_patch_image_tar_trans1_out, \
               endpoint_offset, endpoint_offset_trans, endpoint_gt_offset_trans

    def build_encoder_endpoint(self, patch_image_ref, patch_stroke_ref, patch_image_tar):
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

        if self.hps.enc_model_endpoint == 'combined':
            batch_input = torch.cat([patch_images_ref, patch_strokes_ref, patch_images_tar], dim=1)  # (N, 3, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
            if self.hps.add_coordconv:
                batch_input = add_coords(batch_input, self.coordconv_input)  # (N, in_dim + 2, in_H, in_W)
            output = self.encoder_endpoint(batch_input)  # (N, z_size)
        elif self.hps.enc_model_endpoint == 'separated':
            batch_input_ref = torch.cat([patch_images_ref, patch_strokes_ref], dim=1)  # (N, 2, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
            batch_input_tar = patch_images_tar
            if self.hps.add_coordconv:
                batch_input_ref = add_coords(batch_input_ref, self.coordconv_input)  # (N, in_dim + 2, in_H, in_W)
                batch_input_tar = add_coords(batch_input_tar, self.coordconv_input)  # (N, in_dim + 2, in_H, in_W)
            output = self.encoder_endpoint(batch_input_ref, batch_input_tar)  # (N, z_size)
        else:
            raise Exception('Unknown enc_model_endpoint:', self.hps.enc_model_endpoint)

        return output

    def build_decoder_endpoint(self, dec_input):
        """
        :param dec_input: (N, in_dim)
        :return:
        """
        output = self.decoder_endpoint(dec_input)
        output = torch.tanh(output)  # (N, 2), [-1.0, 1.0]
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

        self.endpoint_model = Endpoint_Model(hps)

        params_list = []

        if self.hps.multi_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.endpoint_model = nn.DataParallel(self.endpoint_model)
            params_list.append({'params': self.endpoint_model.module.parameters()})
        else:
            params_list.append({'params': self.endpoint_model.parameters()})
        self.optimizer = optim.Adam(params_list, lr=hps.learning_rate)

        self.start_step = 0
        self.use_cuda = torch.cuda.is_available()

    def train(self):
        # load weight
        print('-' * 100)

        print('## All variables:')
        if self.hps.multi_gpu:
            gen_num_param = print_model_variables(self.endpoint_model.module.named_parameters(), 'Endpoint_Model')
        else:
            gen_num_param = print_model_variables(self.endpoint_model.named_parameters(), 'Endpoint_Model')
        total_num_param = gen_num_param
        print('Total trainable variables %i.' % total_num_param)

        # print('## Trainable variables:')
        # for param_group in self.optimizer.param_groups:
        #     print(param_group["params"])

        # setup tensorboards
        train_summary_writer = Logger(self.log_dir)

        mean_vector_loss = 0.0

        if self.use_cuda:
            self.endpoint_model = self.endpoint_model.cuda()

        start = time.time()

        for step in range(self.start_step, self.hps.num_steps):
            # print('## Step:', step)
            self.endpoint_model.train()

            curr_learning_rate = ((self.hps.learning_rate - self.hps.min_learning_rate) *
                                  (1 - step / self.hps.num_steps) ** self.hps.decay_power + self.hps.min_learning_rate)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = curr_learning_rate

            train_cost, vector_cost, vector_cost_raw = \
                self.train_step(step, self.train_set, mean_vector_loss)

            ## update mean_vector_loss
            vector_cost_raw_numpy = vector_cost_raw.cpu().detach().numpy()
            mean_vector_loss = (mean_vector_loss * step + vector_cost_raw_numpy) / float(step + 1)

            if (step + 1) % 20 == 0:
                end = time.time()
                time_taken = end - start

                train_summary_writer.scalar_summary('Train_cost', train_cost.item(), step=step + 1)
                train_summary_writer.scalar_summary('Train_vector_cost', vector_cost_raw.item(), step=step + 1)
                train_summary_writer.scalar_summary('Train_vector_cost_norm', vector_cost.item(), step=step + 1)
                train_summary_writer.scalar_summary('Learning_Rate', curr_learning_rate, step=step + 1)
                train_summary_writer.scalar_summary('Time_Taken_Train', time_taken, step=step + 1)

                output_format = ('step: %d, lr: %.6f, cost: %.6f, vec: %.6f, '
                                 'time: %.1f')
                output_values = ((step + 1), curr_learning_rate, train_cost.item(), vector_cost.item(),
                                 time_taken)
                output_log = output_format % output_values
                print(output_log)
                tf.get_logger().info(output_log)
                start = time.time()

            if (step + 1) % self.hps.log_img_every == 0:
                self.endpoint_model.eval()
                self.save_log_images(self.valid_set, self.log_img_dir, (step + 1))

            if (step + 1) % self.hps.save_every == 0:
                self.save_model(step_num=step + 1, save_root=self.snapshot_dir)

        # save model for final step
        self.save_model(step_num=self.hps.num_steps, save_root=self.snapshot_dir)

    def train_step(self, step, data_set, vec_loss_mean):
        reference_images, reference_strokes, target_images, \
            reference_centerpoints, target_centerpoints, target_endpoints_offset_gt, base_window_size, _, _, \
            component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
            target_transform_shear_x_angles, target_transform_shear_y_angles,\
            target_transform1_translate, target_transform1_scaling, \
            target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y, fixing_states = \
            data_set.get_batch(self.use_cuda)
        # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
        # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
        # target_images: (N, H, W, 1), [0-stroke, 1-BG]
        # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
        # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
        # target_endpoints_offset_gt: (N, 1, 2), in [-1.0, 1.0]
        # base_window_size: (N, 1), in [0.0, 1.0]

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

        _, _, _, _, _, endpoints_offset_pred, _, _ = \
            self.endpoint_model(reference_images=reference_images, reference_strokes=reference_strokes,
                                target_images=target_images,
                                centerpoints_pos_ref=reference_centerpoints,
                                centerpoints_pos_tar=target_centerpoints,
                                endpoint_offset_gt=target_endpoints_offset_gt,
                                base_window_size=base_window_size,
                                image_size=image_size,
                                component_centerpoints=component_centerpoints,
                                component_win_sizes=component_win_sizes,
                                target_transform_cursors=target_transform_cursors,
                                target_transform_win_sizes=target_transform_win_sizes,
                                target_transform_angles=target_transform_angles,
                                target_transform_shear_x_angles=target_transform_shear_x_angles,
                                target_transform_shear_y_angles=target_transform_shear_y_angles,
                                target_transform1_translate=target_transform1_translate,
                                target_transform1_scaling=target_transform1_scaling,
                                target_transform1_rotate=target_transform1_rotate,
                                target_transform1_shear_x=target_transform1_shear_x,
                                target_transform1_shear_y=target_transform1_shear_y,
                                fixing_states=fixing_states,
                                )
        # endpoints_offset_pred: (N, 2), [-1.0, 1.0]

        vector_cost_raw, vector_cost = self.get_vector_loss(endpoints_offset_pred, target_endpoints_offset_gt.squeeze(dim=1),
                                                            vec_loss_mean, step)
        cost = vector_cost

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost, vector_cost, vector_cost_raw

    def get_vector_loss(self, pred_params, gt_params, vector_loss_mean, last_step_num):
        """
        :param pred_params: (N, 2), [-1.0, 1.0]
        :param gt_params: (N, 2), [-1.0, 1.0]
        """
        if self.hps.vector_loss_type == 'MSE':
            vector_penalty = torch.sum(torch.pow((gt_params - pred_params), 2), dim=-1)  # (N)
        elif self.hps.vector_loss_type == 'L1':
            vector_penalty = torch.sum(torch.abs(gt_params - pred_params), dim=-1)  # (N)
        else:
            raise Exception('Unknown vector_loss_type:', self.hps.vector_loss_type)

        vector_loss_raw = torch.mean(vector_penalty)
        curr_relu_mean = (vector_loss_mean * last_step_num + vector_loss_raw) / (last_step_num + 1.0)
        vector_loss_norm = vector_loss_raw / curr_relu_mean
        return vector_loss_raw, vector_loss_norm

    def save_log_images(self, data_set, save_root, step_num, save_num=20):
        batch_num = save_num // data_set.batch_size

        with torch.no_grad():
            for batch_i in range(batch_num):
                reference_images, reference_strokes, target_images, \
                    reference_centerpoints, target_centerpoints, target_endpoints_offset_gt, base_window_size, image_ids, endpoint_ids, \
                    component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
                    target_transform_shear_x_angles, target_transform_shear_y_angles,\
                    target_transform1_translate, target_transform1_scaling, \
                    target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y, fixing_states = \
                    data_set.get_batch(self.use_cuda, batch_idx=batch_i, all_example=False)
                # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
                # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
                # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
                # target_endpoints_offset_gt: (N, 1, 2), in [-1.0, 1.0]
                # base_window_size: (N, 1), in [0.0, 1.0]

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

                patch_reference, patch_stroke_reference, patch_target_ori, patch_target_trans0, patch_target_trans1, \
                    endpoints_offset_pred, endpoints_offset_pred_trans, endpoints_offset_gt_trans = \
                    self.endpoint_model(reference_images=reference_images, reference_strokes=reference_strokes,
                                        target_images=target_images,
                                        centerpoints_pos_ref=reference_centerpoints,
                                        centerpoints_pos_tar=target_centerpoints,
                                        endpoint_offset_gt=target_endpoints_offset_gt,
                                        base_window_size=base_window_size,
                                        image_size=image_size,
                                        component_centerpoints=component_centerpoints,
                                        component_win_sizes=component_win_sizes,
                                        target_transform_cursors=target_transform_cursors,
                                        target_transform_win_sizes=target_transform_win_sizes,
                                        target_transform_angles=target_transform_angles,
                                        target_transform_shear_x_angles=target_transform_shear_x_angles,
                                        target_transform_shear_y_angles=target_transform_shear_y_angles,
                                        target_transform1_translate=target_transform1_translate,
                                        target_transform1_scaling=target_transform1_scaling,
                                        target_transform1_rotate=target_transform1_rotate,
                                        target_transform1_shear_x=target_transform1_shear_x,
                                        target_transform1_shear_y=target_transform1_shear_y,
                                        fixing_states=fixing_states,
                                        )
                # patch_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_stroke_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_target_ori: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_target_trans0: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_target_trans1: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # endpoints_offset_pred: (N, 2), [-1.0, 1.0]
                # endpoints_offset_pred_trans: (N, 2), [-1.0, 1.0]
                # endpoints_offset_gt_trans: (N, 2), [-1.0, 1.0]

                patch_reference_np = patch_reference.cpu().data.numpy()
                patch_stroke_reference_np = patch_stroke_reference.cpu().data.numpy()
                patch_target_ori_np = patch_target_ori.cpu().data.numpy()
                patch_target_trans0_np = patch_target_trans0.cpu().data.numpy()
                patch_target_trans1_np = patch_target_trans1.cpu().data.numpy()
                endpoints_offset_pred_np = endpoints_offset_pred.cpu().data.numpy()
                endpoints_offset_pred_trans_np = endpoints_offset_pred_trans.cpu().data.numpy()
                endpoints_offset_gt_trans_np = endpoints_offset_gt_trans.cpu().data.numpy()

                endpoints_offset_gt = torch.squeeze(target_endpoints_offset_gt, dim=1)  # (N, 2), in [-1.0, 1.0]
                endpoints_offset_gt_np = endpoints_offset_gt.cpu().data.numpy()

                for p_i in range(patch_reference_np.shape[0]):
                    endpoint_id = endpoint_ids[p_i]

                    patch_reference_np_i = save_image(patch_reference_np[p_i], save_root, 'ref-' + str(img_index) + '-' + endpoint_id + '.png')
                    save_image(patch_stroke_reference_np[p_i], save_root, 'ref_stroke-' + str(img_index) + '-' + endpoint_id + '.png')
                    patch_target_ori_np_i = save_image(patch_target_ori_np[p_i], save_root, 'tar_ori-' + str(img_index) + '-' + endpoint_id + '.png')
                    patch_target_trans1_np_i = save_image(patch_target_trans1_np[p_i], save_root, 'tar_trans0-' + str(img_index) + '-' + endpoint_id + '.png')

                    save_image_overlap(patch_target_trans0_np[p_i], patch_reference_np_i, save_root,
                                       'tar_trans0_vis-' + str(img_index) + '-' + endpoint_id + '.png')
                    save_image_overlap(patch_target_trans1_np[p_i], patch_reference_np_i, save_root,
                                       'tar_trans1_vis-' + str(img_index) + '-' + endpoint_id + '.png')
                    save_image_overlap(patch_stroke_reference_np[p_i], patch_target_trans1_np_i, save_root,
                                       'ref_stroke_vis-' + str(img_index) + '-' + endpoint_id + '.png')

                    draw_heatmap(endpoints_offset_gt_np[p_i], self.hps.raster_size, save_root,
                                 'tar_ep_gt-' + str(img_index) + '-' + endpoint_id + '.png',
                                 background=patch_target_ori_np_i)
                    draw_heatmap(endpoints_offset_gt_trans_np[p_i], self.hps.raster_size, save_root,
                                 'tar_ep_gt_trans-' + str(img_index) + '-' + endpoint_id + '-step=' + str(step_num) + '.png',
                                 background=patch_target_trans1_np_i)
                    draw_heatmap(endpoints_offset_pred_trans_np[p_i], self.hps.raster_size, save_root,
                                 'tar_ep_pred_trans-' + str(img_index) + '-' + endpoint_id + '-step=' + str(step_num) + '.png',
                                 background=patch_target_trans1_np_i)
                    draw_heatmap(endpoints_offset_pred_np[p_i], self.hps.raster_size, save_root,
                                 'tar_ep_pred-' + str(img_index) + '-' + endpoint_id + '-step=' + str(step_num) + '.png',
                                 background=patch_target_ori_np_i)

    def save_model(self, step_num, save_root):
        if self.use_cuda:
            self.endpoint_model.cpu()

        save_dict = {}

        if self.hps.multi_gpu:
            model_state_dict = self.endpoint_model.module.state_dict()
        else:
            model_state_dict = self.endpoint_model.state_dict()
        # print('model_state_dict')
        # print(model_state_dict.keys())

        save_dict.update(model_state_dict)

        save_path = os.path.join(save_root, "sketch_endpoint_" + str(step_num) + ".pkl")
        torch.save(save_dict, save_path)
        print('Saved model:', save_path)
        if self.use_cuda:
            self.endpoint_model.cuda()

    def evaluate(self, load_trained_weights=False, occluded_only=False, test_max_batch_size=180, distance_threshold=5):
        print('-' * 100)
        print('Evaluation begins ...')

        if load_trained_weights:
            print('-' * 100)
            trained_endpoint_model_path = os.path.join(self.snapshot_dir, "sketch_endpoint_" + str(self.hps.num_steps) + ".pkl")
            if self.hps.multi_gpu:
                load_weights(trained_endpoint_model_path, self.endpoint_model.module)
            else:
                load_weights(trained_endpoint_model_path, self.endpoint_model)
            print('-' * 100)

            if self.use_cuda:
                self.endpoint_model = self.endpoint_model.cuda()

        self.endpoint_model.eval()

        self.valid_set.batch_size = 1
        batch_num = self.valid_set.example_num // self.valid_set.batch_size
        print('batch_num:', batch_num)

        point_error_set_local = []
        point_error_set_global = []
        point_incorrect_state_set = []
        total_endpoint_num = 0

        with (torch.no_grad()):
            for batch_i in range(batch_num):
                print('# batch_i', batch_i)
                batch_data = self.valid_set.get_batch(self.use_cuda, batch_idx=batch_i, all_example=True, occluded_only=occluded_only)
                if batch_data is None:
                    continue

                reference_images, reference_components, reference_strokes, target_images, \
                    reference_centerpoints, target_centerpoints, target_endpoints_offset_gt, base_window_size, _, _, \
                    component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
                    target_transform_shear_x_angles, target_transform_shear_y_angles,\
                    target_transform1_translate, target_transform1_scaling, \
                    target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y, fixing_states = batch_data
                # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_components: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
                # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
                # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
                # target_endpoints_offset_gt: (N, 1, 2), in [-1.0, 1.0]
                # base_window_size: (N, 1), in [0.0, 1.0]

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

                endpoint_num = reference_images.size()[0]
                total_endpoint_num += endpoint_num

                chunks = endpoint_num // test_max_batch_size + 1

                reference_images_chunks = torch.chunk(reference_images, chunks, dim=0)
                reference_components_chunks = torch.chunk(reference_components, chunks, dim=0)
                reference_strokes_chunks = torch.chunk(reference_strokes, chunks, dim=0)
                target_images_chunks = torch.chunk(target_images, chunks, dim=0)
                reference_centerpoints_chunks = torch.chunk(reference_centerpoints, chunks, dim=0)
                target_centerpoints_chunks = torch.chunk(target_centerpoints, chunks, dim=0)
                target_endpoints_offset_gt_chunks = torch.chunk(target_endpoints_offset_gt, chunks, dim=0)
                base_window_size_chunks = torch.chunk(base_window_size, chunks, dim=0)
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

                for chunk_i in range(chunks):
                    _, _, _, _, _, _, endpoints_offset_pred_ch, _, _ = \
                        self.endpoint_model(reference_images=reference_images_chunks[chunk_i],
                                            reference_components=reference_components_chunks[chunk_i],
                                            reference_strokes=reference_strokes_chunks[chunk_i],
                                            target_images=target_images_chunks[chunk_i],
                                            centerpoints_pos_ref=reference_centerpoints_chunks[chunk_i],
                                            centerpoints_pos_tar=target_centerpoints_chunks[chunk_i],
                                            endpoint_offset_gt=target_endpoints_offset_gt_chunks[chunk_i],
                                            base_window_size=base_window_size_chunks[chunk_i],
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
                    # endpoints_offset_pred: (N, 2), [-1.0, 1.0]

                    endpoints_offset_pred_np_ch = endpoints_offset_pred_ch.cpu().data.numpy()  # (N, 2), [-1.0, 1.0]
                    target_endpoints_offset_gt_ch = torch.squeeze(target_endpoints_offset_gt_chunks[chunk_i], dim=1)
                    target_endpoints_offset_gt_np_ch = target_endpoints_offset_gt_ch.cpu().data.numpy()  # (N, 2), [-1.0, 1.0]

                    # Local Point error (LPE):
                    point_error_local = np.sqrt(np.sum(np.power(endpoints_offset_pred_np_ch - target_endpoints_offset_gt_np_ch, 2), axis=-1))  # (N)
                    point_error_set_local += point_error_local.tolist()

                    # Global Point error (GPE):
                    base_window_size_ch = torch.squeeze(base_window_size_chunks[chunk_i], dim=-1)  # (N), in [0.0, 1.0]
                    base_window_size_np_ch = base_window_size_ch.cpu().data.numpy()  # (N), in [0.0, 1.0]
                    centerpoints_pos_tar_ch = torch.squeeze(target_centerpoints_chunks[chunk_i], dim=1)  # (N, 2), in [0.0, 1.0], relative to full size
                    centerpoints_pos_tar_np_ch = centerpoints_pos_tar_ch.cpu().data.numpy()  # (N, 2), in [0.0, 1.0], relative to full size

                    base_window_size_np_ch_scaled = base_window_size_np_ch * image_size * self.hps.window_size_scaling_ref  # (N), in full size
                    base_window_size_np_ch_scaled = np.clip(base_window_size_np_ch_scaled,
                                                            self.hps.window_size_min, image_size * 1.5)
                    base_window_size_np_ch_scaled = np.expand_dims(base_window_size_np_ch_scaled, axis=-1)
                    base_window_size_np_ch_scaled = np.tile(base_window_size_np_ch_scaled, (1, 2))  # (N, 2)

                    points_pred_rel = endpoints_offset_pred_np_ch  # (N, 2), [-1.0, 1.0] relative to window
                    points_pred_offset_global = points_pred_rel * (base_window_size_np_ch_scaled / 2.0)
                    points_pred_global = points_pred_offset_global + centerpoints_pos_tar_np_ch * image_size  # (N, 2), in full size

                    points_gt_rel = target_endpoints_offset_gt_np_ch  # (N, 2), [-1.0, 1.0] relative to window
                    points_gt_offset_global = points_gt_rel * (base_window_size_np_ch_scaled / 2.0)
                    points_gt_global = points_gt_offset_global + centerpoints_pos_tar_np_ch * image_size  # (N, 2), in full size

                    point_error_global = np.sqrt(np.sum(np.power(points_pred_global - points_gt_global, 2), axis=-1))  # (N)
                    point_error_set_global += point_error_global.tolist()

                    incorrect_point_states = point_error_global > distance_threshold
                    point_incorrect_state_set += incorrect_point_states.tolist()

            print('total_endpoint_num', total_endpoint_num)
            tf.get_logger().info('total_endpoint_num: ' + str(total_endpoint_num))
            assert len(point_error_set_local) == len(point_error_set_global) == total_endpoint_num == len(point_incorrect_state_set)
            point_error_local_avg = np.mean(point_error_set_local)
            point_error_global_avg = np.mean(point_error_set_global)
            accurate_rate = 1.0 - np.mean(point_incorrect_state_set)
            print('Local Point error (LPE): win size =', self.hps.window_size_scaling_ref, ':', point_error_local_avg * 100.0, 'e-2')
            print('Global Point error (GPE): win size =', self.hps.window_size_scaling_ref, ':', point_error_global_avg)
            print('Accuracy_rate (ACC): win size =', self.hps.window_size_scaling_ref, ':', accurate_rate * 100.0, '%')
            tf.get_logger().info('Local Point error (LPE): win size = ' + str(self.hps.window_size_scaling_ref) + ' : ' + str(point_error_local_avg * 100.0) + ' e-2')
            tf.get_logger().info('Global Point error (GPE): win size = ' + str(self.hps.window_size_scaling_ref) + ' : ' + str(point_error_global_avg))
            tf.get_logger().info('Accuracy_rate (ACC): win size = ' + str(self.hps.window_size_scaling_ref) + ' : ' + str(accurate_rate * 100.0) + '%')
            print('snapshot_dir:', self.snapshot_dir)
            print('transform_model_name:', self.hps.transform_model_name)
            print('transform_local_model_name:', self.hps.transform_local_model_name)
            tf.get_logger().info('snapshot_dir: ' + self.snapshot_dir)
            tf.get_logger().info('transform_model_name: ' + self.hps.transform_model_name)
            tf.get_logger().info('transform_local_model_name: ' + self.hps.transform_local_model_name)

    def inference(self, save_root, show_data='selected', test_max_batch_size=180):
        print('-' * 100)
        print('Inference begins ...')

        trained_endpoint_model_path = os.path.join(self.snapshot_dir, "sketch_endpoint_" + str(self.hps.num_steps) + ".pkl")
        if self.hps.multi_gpu:
            load_weights(trained_endpoint_model_path, self.endpoint_model.module)
        else:
            load_weights(trained_endpoint_model_path, self.endpoint_model)

        if self.use_cuda:
            self.endpoint_model = self.endpoint_model.cuda()

        self.endpoint_model.eval()

        if show_data == 'all':
            show_num = self.valid_set.example_num
            batch_idx_offsets = [0]
            occluded_only = False
        elif show_data == 'occluded':
            show_num = 50
            batch_idx_offsets = [0, 737]
            occluded_only = True
        elif show_data == 'selected':
            show_num = 20
            batch_idx_offsets = [0, 737]
            occluded_only = False
        else:
            raise Exception('Unknown show_data:', show_data)

        batch_num = self.valid_set.example_num // self.valid_set.batch_size
        print('batch_num:', batch_num)

        with torch.no_grad():
            for batch_idx_offset in batch_idx_offsets:
                show_i = 0
                for batch_i in range(batch_num):
                    print('# batch_i', batch_i)

                    batch_data = self.valid_set.get_batch(self.use_cuda, batch_idx=batch_i, all_example=True, batch_idx_offset=batch_idx_offset, occluded_only=occluded_only)
                    if batch_data is None:
                        continue

                    reference_images, reference_components, reference_strokes, target_images, \
                        reference_centerpoints, target_centerpoints, target_endpoints_offset_gt, base_window_size, image_ids, endpoint_ids, \
                        component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
                        target_transform_shear_x_angles, target_transform_shear_y_angles,\
                        target_transform1_translate, target_transform1_scaling, \
                        target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y, fixing_states = batch_data
                    # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                    # reference_components: (N, H, W, 1), [0-stroke, 1-BG]
                    # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
                    # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                    # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
                    # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
                    # target_endpoints_offset_gt: (N, 1, 2), in [-1.0, 1.0]
                    # base_window_size: (N, 1), in [0.0, 1.0]

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
                    print(' >> img_index', img_index)

                    stroke_num = reference_images.size()[0]
                    chunks = stroke_num // test_max_batch_size + 1

                    reference_images_chunks = torch.chunk(reference_images, chunks, dim=0)
                    reference_components_chunks = torch.chunk(reference_components, chunks, dim=0)
                    reference_strokes_chunks = torch.chunk(reference_strokes, chunks, dim=0)
                    target_images_chunks = torch.chunk(target_images, chunks, dim=0)
                    reference_centerpoints_chunks = torch.chunk(reference_centerpoints, chunks, dim=0)
                    target_centerpoints_chunks = torch.chunk(target_centerpoints, chunks, dim=0)
                    target_endpoints_offset_gt_chunks = torch.chunk(target_endpoints_offset_gt, chunks, dim=0)
                    base_window_size_chunks = torch.chunk(base_window_size, chunks, dim=0)
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
                    patch_component_reference_np = []
                    patch_stroke_reference_np = []
                    patch_target_ori_np = []
                    patch_target_trans0_np = []
                    patch_target_trans1_np = []
                    endpoints_offset_pred_np = []
                    endpoints_offset_pred_trans_np = []
                    endpoints_offset_gt_trans_np = []

                    for chunk_i in range(chunks):
                        patch_reference_ch, patch_component_reference_ch, patch_stroke_reference_ch, patch_target_ori_ch, patch_target_trans0_ch, patch_target_trans1_ch, \
                            endpoints_offset_pred_ch, endpoints_offset_pred_trans_ch, endpoints_offset_gt_trans_ch = \
                            self.endpoint_model(reference_images=reference_images_chunks[chunk_i],
                                                reference_components=reference_components_chunks[chunk_i],
                                                reference_strokes=reference_strokes_chunks[chunk_i],
                                                target_images=target_images_chunks[chunk_i],
                                                centerpoints_pos_ref=reference_centerpoints_chunks[chunk_i],
                                                centerpoints_pos_tar=target_centerpoints_chunks[chunk_i],
                                                endpoint_offset_gt=target_endpoints_offset_gt_chunks[chunk_i],
                                                base_window_size=base_window_size_chunks[chunk_i],
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
                        # patch_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                        # patch_stroke_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                        # patch_target_ori: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                        # patch_target_trans: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                        # endpoints_offset_pred: (N, 2), [-1.0, 1.0]
                        # endpoints_offset_pred_trans: (N, 2), [-1.0, 1.0]
                        # endpoints_offset_gt_trans: (N, 2), [-1.0, 1.0]

                        patch_reference_np_ch = patch_reference_ch.cpu().data.numpy()
                        patch_component_reference_np_ch = patch_component_reference_ch.cpu().data.numpy()
                        patch_stroke_reference_np_ch = patch_stroke_reference_ch.cpu().data.numpy()
                        patch_target_ori_np_ch = patch_target_ori_ch.cpu().data.numpy()
                        patch_target_trans0_ch_ch = patch_target_trans0_ch.cpu().data.numpy()
                        patch_target_trans1_ch_ch = patch_target_trans1_ch.cpu().data.numpy()
                        endpoints_offset_pred_np_ch = endpoints_offset_pred_ch.cpu().data.numpy()
                        endpoints_offset_pred_trans_np_ch = endpoints_offset_pred_trans_ch.cpu().data.numpy()
                        endpoints_offset_gt_trans_np_ch = endpoints_offset_gt_trans_ch.cpu().data.numpy()

                        patch_reference_np.append(patch_reference_np_ch)
                        patch_component_reference_np.append(patch_component_reference_np_ch)
                        patch_stroke_reference_np.append(patch_stroke_reference_np_ch)
                        patch_target_ori_np.append(patch_target_ori_np_ch)
                        patch_target_trans0_np.append(patch_target_trans0_ch_ch)
                        patch_target_trans1_np.append(patch_target_trans1_ch_ch)
                        endpoints_offset_pred_np.append(endpoints_offset_pred_np_ch)
                        endpoints_offset_pred_trans_np.append(endpoints_offset_pred_trans_np_ch)
                        endpoints_offset_gt_trans_np.append(endpoints_offset_gt_trans_np_ch)

                    patch_reference_np = np.concatenate(patch_reference_np, axis=0)
                    patch_component_reference_np = np.concatenate(patch_component_reference_np, axis=0)
                    patch_stroke_reference_np = np.concatenate(patch_stroke_reference_np, axis=0)
                    patch_target_ori_np = np.concatenate(patch_target_ori_np, axis=0)
                    patch_target_trans0_np = np.concatenate(patch_target_trans0_np, axis=0)
                    patch_target_trans1_np = np.concatenate(patch_target_trans1_np, axis=0)
                    endpoints_offset_pred_np = np.concatenate(endpoints_offset_pred_np, axis=0)
                    endpoints_offset_pred_trans_np = np.concatenate(endpoints_offset_pred_trans_np, axis=0)
                    endpoints_offset_gt_trans_np = np.concatenate(endpoints_offset_gt_trans_np, axis=0)

                    endpoints_offset_gt = torch.squeeze(target_endpoints_offset_gt, dim=1)  # (N, 2), in [-1.0, 1.0]
                    endpoints_offset_gt_np = endpoints_offset_gt.cpu().data.numpy()

                    for p_i in range(patch_reference_np.shape[0]):
                        endpoint_id = endpoint_ids[p_i]

                        patch_reference_np_i = save_image(patch_reference_np[p_i], save_root, 'ref-' + str(img_index) + '-' + endpoint_id + '.png')
                        patch_component_reference_np_i = save_image(patch_component_reference_np[p_i], save_root, 'ref_comp-' + str(img_index) + '-' + endpoint_id + '.png')
                        save_image(patch_stroke_reference_np[p_i], save_root, 'ref_stroke-' + str(img_index) + '-' + endpoint_id + '.png')
                        patch_target_ori_np_i = save_image(patch_target_ori_np[p_i], save_root, 'tar_ori-' + str(img_index) + '-' + endpoint_id + '.png')
                        if self.hps.use_local_transform:
                            patch_target_trans_np_i = save_image(patch_target_trans1_np[p_i], save_root, 'tar_trans1-' + str(img_index) + '-' + endpoint_id + '.png')
                        else:
                            patch_target_trans_np_i = save_image(patch_target_trans0_np[p_i], save_root, 'tar_trans0-' + str(img_index) + '-' + endpoint_id + '.png')

                        save_image_overlap(patch_target_trans0_np[p_i], patch_reference_np_i, save_root,
                                           'tar_trans0_vis-' + str(img_index) + '-' + endpoint_id + '.png')
                        save_image_overlap(patch_target_trans0_np[p_i], patch_component_reference_np_i, save_root,
                                           'tar_trans0_comp_vis-' + str(img_index) + '-' + endpoint_id + '.png')
                        save_image_overlap(patch_target_trans1_np[p_i], patch_reference_np_i, save_root,
                                           'tar_trans1_vis-' + str(img_index) + '-' + endpoint_id + '.png')
                        save_image_overlap(patch_target_trans1_np[p_i], patch_component_reference_np_i, save_root,
                                           'tar_trans1_comp_vis-' + str(img_index) + '-' + endpoint_id + '.png')
                        save_image_overlap(patch_stroke_reference_np[p_i], patch_target_trans_np_i, save_root,
                                           'ref_stroke_vis-' + str(img_index) + '-' + endpoint_id + '.png')

                        draw_heatmap(endpoints_offset_gt_np[p_i], self.hps.raster_size, save_root,
                                     'tar_ep_gt-' + str(img_index) + '-' + endpoint_id + '.png',
                                     background=patch_target_ori_np_i)
                        draw_heatmap(endpoints_offset_gt_trans_np[p_i], self.hps.raster_size, save_root,
                                     'tar_ep_gt_trans-' + str(img_index) + '-' + endpoint_id + '.png',
                                     background=patch_target_trans_np_i)
                        draw_heatmap(endpoints_offset_pred_trans_np[p_i], self.hps.raster_size, save_root,
                                     'tar_ep_pred_trans-' + str(img_index) + '-' + endpoint_id + '.png',
                                     background=patch_target_trans_np_i)
                        draw_heatmap(endpoints_offset_pred_np[p_i], self.hps.raster_size, save_root,
                                     'tar_ep_pred-' + str(img_index) + '-' + endpoint_id + '.png',
                                     background=patch_target_ori_np_i)

                    show_i += 1
                    if show_i >= show_num:
                        break

    def inference_full(self, save_root, show_data='selected', stroke_fixing=False, test_max_batch_size=180, distance_threshold=5):
        print('-' * 100)
        print('Inference begins ...')

        trained_endpoint_model_path = os.path.join(self.snapshot_dir, "sketch_endpoint_" + str(self.hps.num_steps) + ".pkl")
        if self.hps.multi_gpu:
            load_weights(trained_endpoint_model_path, self.endpoint_model.module)
        else:
            load_weights(trained_endpoint_model_path, self.endpoint_model)

        if self.use_cuda:
            self.endpoint_model = self.endpoint_model.cuda()

        self.endpoint_model.eval()

        if show_data == 'all':
            show_num = self.valid_set.example_num
            is_save_image = False
            batch_idx_offsets = [0]
            data_split = 'val'
            occluded_only = False
        elif show_data == 'selected':
            show_num = 50
            is_save_image = True
            batch_idx_offsets = [0, 737]
            data_split = 'val'
            occluded_only = False
        else:
            raise Exception('Unknown show_data:', show_data)

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

                    reference_images, reference_strokes, target_images, \
                        reference_centerpoints, target_centerpoints, target_endpoints_offset_gt, base_window_size, image_ids, endpoint_ids, \
                        component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
                        target_transform_shear_x_angles, target_transform_shear_y_angles,\
                        target_transform1_translate, target_transform1_scaling, \
                        target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y, fixing_states = batch_data
                    # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                    # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
                    # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                    # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
                    # target_centerpoints: (N, 1, 2), in [0.0, 1.0]
                    # target_endpoints_offset_gt: (N, 1, 2), in [-1.0, 1.0]
                    # base_window_size: (N, 1), in [0.0, 1.0]

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
                    if target_endpoints_offset_gt is None:
                        target_endpoints_offset_gt = target_centerpoints

                    assert len(image_ids) == 1
                    img_index = image_ids[0]
                    print(' >> img_index', img_index)
                    tf.get_logger().info(' >> img_index: ' + str(img_index))

                    stroke_num = reference_images.size()[0]
                    chunks = stroke_num // test_max_batch_size + 1

                    reference_images_chunks = torch.chunk(reference_images, chunks, dim=0)
                    reference_strokes_chunks = torch.chunk(reference_strokes, chunks, dim=0)
                    target_images_chunks = torch.chunk(target_images, chunks, dim=0)
                    reference_centerpoints_chunks = torch.chunk(reference_centerpoints, chunks, dim=0)
                    target_centerpoints_chunks = torch.chunk(target_centerpoints, chunks, dim=0)
                    target_endpoints_offset_gt_chunks = torch.chunk(target_endpoints_offset_gt, chunks, dim=0)
                    base_window_size_chunks = torch.chunk(base_window_size, chunks, dim=0)
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

                    endpoints_offset_pred_np = []

                    for chunk_i in range(chunks):
                        _, _, _, _, _, _, endpoints_offset_pred_ch, _, _ = \
                            self.endpoint_model(reference_images=reference_images_chunks[chunk_i],
                                                reference_components=reference_strokes_chunks[chunk_i],
                                                reference_strokes=reference_strokes_chunks[chunk_i],
                                                target_images=target_images_chunks[chunk_i],
                                                centerpoints_pos_ref=reference_centerpoints_chunks[chunk_i],
                                                centerpoints_pos_tar=target_centerpoints_chunks[chunk_i],
                                                endpoint_offset_gt=target_endpoints_offset_gt_chunks[chunk_i],
                                                base_window_size=base_window_size_chunks[chunk_i],
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
                        # endpoints_offset_pred: (N, 2), [-1.0, 1.0]

                        endpoints_offset_pred_np_ch = endpoints_offset_pred_ch.cpu().data.numpy()
                        endpoints_offset_pred_np.append(endpoints_offset_pred_np_ch)

                    ## convert to global coordinate
                    reference_endpoints = torch.squeeze(reference_centerpoints, dim=1)  # (N, 2), in [0.0, 1.0]
                    reference_endpoints_np = reference_endpoints.cpu().data.numpy()
                    reference_endpoints_global_np = reference_endpoints_np * image_size  # (N, 2), in full size

                    endpoints_offset_pred_np = np.concatenate(endpoints_offset_pred_np, axis=0)  # (N, 2), in [-1.0, 1.0]

                    endpoints_offset_gt = torch.squeeze(target_endpoints_offset_gt, dim=1)  # (N, 2), in [-1.0, 1.0]
                    endpoints_offset_gt_np = endpoints_offset_gt.cpu().data.numpy()

                    base_window_size = torch.squeeze(base_window_size, dim=-1)  # (N), in [0.0, 1.0]
                    base_window_size_np = base_window_size.cpu().data.numpy()  # (N), in [0.0, 1.0]
                    base_window_size_np_scaled = base_window_size_np * image_size * self.hps.window_size_scaling_ref  # (N), in full size
                    base_window_size_np_scaled = np.clip(base_window_size_np_scaled, self.hps.window_size_min, image_size * 1.5)
                    base_window_size_np_scaled = np.expand_dims(base_window_size_np_scaled, axis=-1)
                    base_window_size_np_scaled = np.tile(base_window_size_np_scaled, (1, 2))  # (N, 2)

                    centerpoints_pos_tar = torch.squeeze(target_centerpoints, dim=1)  # (N, 2), in [0.0, 1.0], relative to full size
                    centerpoints_pos_tar_np = centerpoints_pos_tar.cpu().data.numpy()  # (N, 2), in [0.0, 1.0], relative to full size

                    points_pred_rel = endpoints_offset_pred_np  # (N, 2), [-1.0, 1.0] relative to window
                    points_pred_offset_global = points_pred_rel * (base_window_size_np_scaled / 2.0)
                    points_pred_global = points_pred_offset_global + centerpoints_pos_tar_np * image_size  # (N, 2), in full size

                    if not is_save_image:
                        selected_dataset_name = img_index[:img_index.find('-')]
                        selected_img_index = img_index[img_index.find('-') + 1:]
                        endpoints_pred_data_save_base = os.path.join(self.hps.dataset_base, selected_dataset_name + '_512', data_split,
                                                                        'vector-endpoint-prediction', self.hps.workspace)
                        endpoints_pred_data_save_base += '-[c_min=' + str(self.hps.window_size_min_comp) + ']'
                        if self.hps.use_optical_flow:
                            endpoints_pred_data_save_base += '-[optical]'
                        if stroke_fixing:
                            endpoints_pred_data_save_base += '-[fixing]'
                        os.makedirs(endpoints_pred_data_save_base, exist_ok=True)
                        endpoints_pred_data_save_path = os.path.join(endpoints_pred_data_save_base, selected_img_index + '.jsonl')

                        assert len(points_pred_global) == len(endpoint_ids)
                        endpoints_params_data = {}
                        endpoints_params_data['endpoints_pred'] = points_pred_global.tolist()
                        endpoints_params_data['endpoint_ids'] = endpoint_ids
                        with jsonlines.open(endpoints_pred_data_save_path, mode='w') as json_writer:
                            json_writer.write(endpoints_params_data)
                    else:
                        points_gt_rel = endpoints_offset_gt_np  # (N, 2), [-1.0, 1.0] relative to window
                        points_gt_offset_global = points_gt_rel * (base_window_size_np_scaled / 2.0)
                        points_gt_global = points_gt_offset_global + centerpoints_pos_tar_np * image_size  # (N, 2), in full size

                        reference_images_np = reference_images.cpu().data.numpy()  # (N, H, W, 1), [0-stroke, 1-BG]
                        target_images_np = target_images.cpu().data.numpy()  # (N, H, W, 1), [0-stroke, 1-BG]

                        reference_image_np = reference_images_np[0, :, :, 0] * 255.0  # (H, W), [0-stroke, 255-BG]
                        target_image_np = target_images_np[0, :, :, 0] * 255.0  # (H, W), [0-stroke, 255-BG]

                        draw_dot_full(reference_image_np, reference_endpoints_global_np, save_root, 'ref-' + str(img_index) + '.png')
                        draw_dot_full(target_image_np, points_gt_global, save_root, 'tar_gt-' + str(img_index) + '.png')
                        draw_dot_full(target_image_np, points_pred_global, save_root, 'tar_pred-' + str(img_index) + '.png')

                        # save incorrect endpoints
                        point_error_global = np.sqrt(np.sum(np.power(points_pred_global - points_gt_global, 2), axis=-1))  # (N)
                        incorrect_point_states = point_error_global > distance_threshold
                        # incorrect_point_indices = np.argwhere(point_error_global > distance_threshold)
                        # ref_incorrect_endpoints = reference_endpoints_global_np[incorrect_point_indices].squeeze(axis=1)
                        # gt_incorrect_endpoints = points_gt_global[incorrect_point_indices].squeeze(axis=1)
                        # pred_incorrect_endpoints = points_pred_global[incorrect_point_indices].squeeze(axis=1)

                        save_dir = os.path.join(save_root, 'incorrect')
                        os.makedirs(save_dir, exist_ok=True)
                        draw_dot_full(reference_image_np, reference_endpoints_global_np, save_dir, 'ref-' + str(img_index) + '.png', drawn_states=incorrect_point_states)
                        draw_dot_full(target_image_np, points_gt_global, save_dir, 'tar_gt-' + str(img_index) + '.png', drawn_states=incorrect_point_states)
                        draw_dot_full(target_image_np, points_pred_global, save_dir, 'tar_pred-' + str(img_index) + '.png', drawn_states=incorrect_point_states)

                    show_i += 1
                    if show_i >= show_num:
                        break

    def inference_full_real(self, save_root, data_base, test_max_batch_size=180, do_inv=False):
        # print('-' * 100)
        print('Inference of [Endpoint Matching (3/4)] begins ...')

        trained_endpoint_model_path = os.path.join(self.snapshot_dir, "sketch_endpoint_" + str(self.hps.num_steps) + ".pkl")
        if self.hps.multi_gpu:
            load_weights(trained_endpoint_model_path, self.endpoint_model.module)
        else:
            load_weights(trained_endpoint_model_path, self.endpoint_model)

        if self.use_cuda:
            self.endpoint_model = self.endpoint_model.cuda()

        self.endpoint_model.eval()

        is_save_image = False

        with torch.no_grad():
            batch_data = self.valid_set.get_batch(self.use_cuda, test_img_id=test_img_id)

            reference_images, reference_strokes, target_images, target_images_ori, \
                reference_centerpoints, target_centerpoints, base_window_size, image_ids, endpoint_ids, \
                component_centerpoints, component_win_sizes, target_transform_cursors, target_transform_win_sizes, target_transform_angles, \
                target_transform_shear_x_angles, target_transform_shear_y_angles,\
                target_transform1_translate, target_transform1_scaling, \
                target_transform1_rotate, target_transform1_shear_x, target_transform1_shear_y, fixing_states, connect_states = batch_data
            # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
            # reference_strokes: (N, H, W, 1), [0-stroke, 1-BG]
            # target_images: (N, H, W, 1), [0-stroke, 1-BG]
            # target_images_ori: (N, H, W, 1), [0-stroke, 1-BG]
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

            # target_transform1_translate:(N, 1, 2), [-1.0, 1.0], relative to target trans0 window
            # target_transform1_scaling: (N, 1, 2), [0.2, 2.0], relative to target trans0 window
            # target_transform1_rotate: (N, 1), [-180.0, 180.0]
            # target_transform1_shear_x / target_transform1_shear_y: (N, 1), [-90.0, 90.0]

            # fixing_states: (N, 1), [0-nonfix, 1-fix]
            # connect_states: list (N) of '0_1_2' / None

            image_size = reference_images.size()[1]

            assert len(image_ids) == 1
            img_index = image_ids[0]
            selected_img_index = str(test_img_id)
            # print(' >> img_index', img_index)

            stroke_num = reference_images.size()[0]
            chunks = stroke_num // test_max_batch_size + 1

            reference_images_chunks = torch.chunk(reference_images, chunks, dim=0)
            reference_strokes_chunks = torch.chunk(reference_strokes, chunks, dim=0)
            target_images_chunks = torch.chunk(target_images, chunks, dim=0)
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
            target_transform1_translate_chunks = torch.chunk(target_transform1_translate, chunks, dim=0)
            target_transform1_scaling_chunks = torch.chunk(target_transform1_scaling, chunks, dim=0)
            target_transform1_rotate_chunks = torch.chunk(target_transform1_rotate, chunks, dim=0)
            target_transform1_shear_x_chunks = torch.chunk(target_transform1_shear_x, chunks, dim=0)
            target_transform1_shear_y_chunks = torch.chunk(target_transform1_shear_y, chunks, dim=0)
            fixing_states_chunks = torch.chunk(fixing_states, chunks, dim=0)

            endpoints_offset_pred_np = []

            for chunk_i in range(chunks):
                _, _, _, _, _, _, endpoints_offset_pred_ch, _, _ = \
                    self.endpoint_model(reference_images=reference_images_chunks[chunk_i],
                                        reference_components=reference_strokes_chunks[chunk_i],
                                        reference_strokes=reference_strokes_chunks[chunk_i],
                                        target_images=target_images_chunks[chunk_i],
                                        centerpoints_pos_ref=reference_centerpoints_chunks[chunk_i],
                                        centerpoints_pos_tar=target_centerpoints_chunks[chunk_i],
                                        endpoint_offset_gt=target_centerpoints_chunks[chunk_i],
                                        base_window_size=base_window_size_chunks[chunk_i],
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
                # endpoints_offset_pred: (N, 2), [-1.0, 1.0]

                endpoints_offset_pred_np_ch = endpoints_offset_pred_ch.cpu().data.numpy()
                endpoints_offset_pred_np.append(endpoints_offset_pred_np_ch)

            ## convert to global coordinate
            reference_endpoints = torch.squeeze(reference_centerpoints, dim=1)  # (N, 2), in [0.0, 1.0]
            reference_endpoints_np = reference_endpoints.cpu().data.numpy()
            reference_endpoints_global_np = reference_endpoints_np * image_size  # (N, 2), in full size

            endpoints_offset_pred_np = np.concatenate(endpoints_offset_pred_np, axis=0)  # (N, 2), in [-1.0, 1.0]

            base_window_size = torch.squeeze(base_window_size, dim=-1)  # (N), in [0.0, 1.0]
            base_window_size_np = base_window_size.cpu().data.numpy()  # (N), in [0.0, 1.0]
            base_window_size_np_scaled = base_window_size_np * image_size * self.hps.window_size_scaling_ref  # (N), in full size
            base_window_size_np_scaled = np.clip(base_window_size_np_scaled, self.hps.window_size_min, image_size * 1.5)
            base_window_size_np_scaled = np.expand_dims(base_window_size_np_scaled, axis=-1)
            base_window_size_np_scaled = np.tile(base_window_size_np_scaled, (1, 2))  # (N, 2)

            centerpoints_pos_tar = torch.squeeze(target_centerpoints, dim=1)  # (N, 2), in [0.0, 1.0], relative to full size
            centerpoints_pos_tar_np = centerpoints_pos_tar.cpu().data.numpy()  # (N, 2), in [0.0, 1.0], relative to full size

            points_pred_rel = endpoints_offset_pred_np  # (N, 2), [-1.0, 1.0] relative to window
            points_pred_offset_global = points_pred_rel * (base_window_size_np_scaled / 2.0)
            points_pred_global = points_pred_offset_global + centerpoints_pos_tar_np * image_size  # (N, 2), in full size

            # Use ref0 position for connected points
            if do_inv:
                if data_base.endswith('/'):
                    data_base_ref0 = data_base[:data_base[:-1].rfind('/')]
                else:
                    data_base_ref0 = data_base[:data_base.rfind('/')]
                vector_data_path_ref0 = os.path.join(data_base_ref0, 'vector-params', str(img_index) + '_ref.jsonl')
                with open(vector_data_path_ref0, "r+") as f:
                    for item in jsonlines.Reader(f):
                        stroke_data_b_ref0 = item['stroke_params']

                assert len(connect_states) == len(points_pred_global)

                for p_i in range(len(connect_states)):
                    connect_state = connect_states[p_i]
                    if connect_state is None:
                        continue

                    corr_comp_i, corr_curve_i, corr_point_i = connect_state.split('_')
                    corr_curve = stroke_data_b_ref0[int(corr_comp_i)][int(corr_curve_i)]  # (N', 4, 2)
                    corr_curve_endpoints = [corr_curve[0][0]]
                    for ce_i in range(len(corr_curve)):
                        corr_curve_endpoints.append(corr_curve[ce_i][-1])
                    corr_point_ref0 = corr_curve_endpoints[int(corr_point_i)]
                    points_pred_global[p_i] = corr_point_ref0

            if not is_save_image:
                endpoints_pred_data_save_base = os.path.join(data_base,
                                                                'vector-endpoint-prediction', self.hps.workspace)
                endpoints_pred_data_save_base += '-[c_min=' + str(self.hps.window_size_min_comp) + ']'
                if self.hps.use_optical_flow:
                    endpoints_pred_data_save_base += '-[optical]'
                os.makedirs(endpoints_pred_data_save_base, exist_ok=True)
                endpoints_pred_data_save_path = os.path.join(endpoints_pred_data_save_base, selected_img_index + '.jsonl')

                assert len(points_pred_global) == len(endpoint_ids)
                endpoints_params_data = {}
                endpoints_params_data['endpoints_pred'] = points_pred_global.tolist()
                endpoints_params_data['endpoint_ids'] = endpoint_ids
                with jsonlines.open(endpoints_pred_data_save_path, mode='w') as json_writer:
                    json_writer.write(endpoints_params_data)
            else:
                reference_images_np = reference_images.cpu().data.numpy()  # (N, H, W, 1), [0-stroke, 1-BG]
                target_images_np = target_images_ori.cpu().data.numpy()  # (N, H, W, 1), [0-stroke, 1-BG]

                reference_image_np = reference_images_np[0, :, :, 0] * 255.0  # (H, W), [0-stroke, 255-BG]
                target_image_np = target_images_np[0, :, :, 0] * 255.0  # (H, W), [0-stroke, 255-BG]

                os.makedirs(save_root, exist_ok=True)
                draw_dot_full(reference_image_np, reference_endpoints_global_np, save_root, 'ref-' + str(img_index) + '.png')
                draw_dot_full(target_image_np, points_pred_global, save_root, 'tar_pred-' + str(img_index) + '.png')
