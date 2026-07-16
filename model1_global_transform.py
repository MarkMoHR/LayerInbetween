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
from image_utils.model_processing import get_coordconv, add_coords, normalize_image_m1to1, image_cropping_stn, \
    load_weights, print_model_variables
from network.vanilla import CNN_Encoder, MLP_Decoder
from configs.example_configs import test_img_id

tf.get_logger().setLevel('INFO')


def get_default_hparams():
    """Return default HParams for sketch-rnn."""
    hparams = HParams(
        workspace='FAD3-T12-2.0x-51-min=64',
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

        transform_module_zero_init='last',  # ['none', 'last', 'all']
        transform_with_rotation=True,
        transform_with_shear=True,
        # window_size_cropping_grad=False,

        use_ref_ori_img=True,  # whether use original reference patch as input

        enc_model_transform='combined',  # ['combined', 'separated', 'separated_fully']
        dec_model_transform='mlp',  # ['rnn', 'mlp']

        add_coordconv=True,
        z_size=256,  # Size of latent vector z.

        stroke_thickness=1.2,  # 2.0 for toy; 1.2 for TUB

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


class Transformation_Model(nn.Module):
    def __init__(self, hps):
        super(Transformation_Model, self).__init__()
        self.hps = hps
        self.stroke_thickness = hps.stroke_thickness

        cnn_in_size = 3 if self.hps.use_ref_ori_img else 2

        transform_out_size = 2  # scaling
        transform_out_size += 2  # translation
        if self.hps.transform_with_rotation:
            transform_out_size += 1
        if self.hps.transform_with_shear:
            transform_out_size += 2

        # transform encoder
        if self.hps.enc_model_transform == 'combined':
            if self.hps.add_coordconv:
                cnn_in_size += 2
            cnn_out_size = self.hps.z_size

            self.encoder_transform = CNN_Encoder(cnn_in_size, cnn_out_size, input_size=self.hps.raster_size)
        else:
            raise Exception('Unknown enc_model_transform:', self.hps.enc_model_transform)

        if self.hps.add_coordconv:
            self.coordconv_input = get_coordconv(self.hps.raster_size)  # (2, raster_size, raster_size)

        dec_in_size = self.hps.z_size

        if self.hps.dec_model_transform == 'mlp':
            self.decoder_transform = MLP_Decoder(dec_in_size, transform_out_size, zero_init=self.hps.transform_module_zero_init)
        else:
            raise Exception('Unknown dec_model_transform:', self.hps.dec_model_transform)

    def forward(self, reference_images, reference_components, target_images, target_components,
                centerpoints_pos_ref, centerpoints_pos_tar, base_window_size, image_size, model_mode):
        """
        :param reference_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_components: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_components: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param centerpoints_pos_ref: (N, 1, 2), float32, in [0.0, 1.0]
        :param centerpoints_pos_tar: (N, 1, 2), float32, in [0.0, 1.0]
        :param base_window_size: (N, 1), float32, in [0.0, 1.0]
        :return:
        """
        self.model_mode = model_mode
        self.image_size = image_size
        assert model_mode in ['train', 'valid', 'eval', 'inference']

        patch_reference, patch_component_reference, patch_target_original, patch_component_target_original, \
        patch_target_transformed, patch_component_target_transformed, \
        pred_cursor_position_tar, pred_window_size_tar, pred_rotate_angle_tar, \
        pred_shear_x_angle_tar, pred_shear_y_angle_tar, component_win_size = \
            self.get_points_and_raster_image(reference_images, reference_components, target_images, target_components,
                                             centerpoints_pos_ref, centerpoints_pos_tar,
                                             base_window_size)
        # patch_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_component_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_target_original: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_component_target_original: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_target_transformed: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_component_target_transformed: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        # pred_cursor_position_tar: (N, 1, 2), [0.0, 1.0], relative to image size
        # pred_window_size_tar: (N, 1, 2), in image size
        # pred_rotate_angle_tar: (N, 1), [-180.0, 180.0]
        # pred_shear_x_angle_tar / pred_shear_y_angle_tar: (N, 1), [-90.0, 90.0]
        # component_win_size: (N, 1, 2), in full size

        return patch_reference, patch_component_reference, patch_target_original, patch_component_target_original, \
            patch_target_transformed, patch_component_target_transformed, \
            pred_cursor_position_tar, pred_window_size_tar, pred_rotate_angle_tar, \
            pred_shear_x_angle_tar, pred_shear_y_angle_tar, component_win_size

    def get_points_and_raster_image(self, reference_images, reference_components, target_images, target_components,
                                    centerpoints_pos_ref, centerpoints_pos_tar,
                                    base_window_size):
        """
        :param reference_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param reference_components: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_images: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param target_components: (N, H, W, 1), float32, [0.0-stroke, 1.0-BG]
        :param centerpoints_pos_ref: (N, 1, 2), float32, in [0.0, 1.0]
        :param centerpoints_pos_tar: (N, 1, 2), float32, in [0.0, 1.0]
        :param base_window_size: (N, 1), float32, in [0.0, 1.0]
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
        crop_inputs_ref = torch.cat([reference_images, reference_components], dim=-1)  # (N, H, W, *)
        cropped_outputs = image_cropping_stn(cursor_position_loop_ref, crop_inputs_ref, self.image_size, self.hps.raster_size, curr_window_size)

        curr_patch_image_ref = cropped_outputs[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_component_ref = cropped_outputs[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]

        curr_patch_image_ref_out = torch.squeeze(curr_patch_image_ref, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_component_ref_out = torch.squeeze(curr_patch_component_ref, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        curr_patch_image_ref = normalize_image_m1to1(curr_patch_image_ref)
        curr_patch_component_ref = normalize_image_m1to1(curr_patch_component_ref)
        # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

        ## target_images: (N, H, W, 1), [0.0-stroke, 1.0-BG]
        crop_inputs_tar = torch.cat([target_images, target_components], dim=-1)  # (N, H, W, *)
        cropped_outputs = image_cropping_stn(cursor_position_loop_tar, crop_inputs_tar, self.image_size, self.hps.raster_size, curr_window_size)

        curr_patch_image_tar_temp = cropped_outputs[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_component_tar_temp = cropped_outputs[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]

        curr_patch_image_tar_out = torch.squeeze(curr_patch_image_tar_temp, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_component_tar_out = torch.squeeze(curr_patch_component_tar_temp, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        curr_patch_image_tar_temp = normalize_image_m1to1(curr_patch_image_tar_temp)
        # (N, raster_size, raster_size, 1), [-1.0-stroke, 1.0-BG]

        ## generate the transformation of target window size
        transform_z = self.build_encoder_transform(curr_patch_image_ref, curr_patch_component_ref, curr_patch_image_tar_temp)
        transform_output, transform_next_state = self.build_decoder_transform(transform_z)
        # transform_output: (N, 5)

        transform_output_translation = transform_output[:, 0:2]  # (N, 2)
        transform_output_scaling = transform_output[:, 2:4]  # (N, 2)
        if self.hps.transform_with_rotation:
            transform_output_rotate_angle = transform_output[:, 4:5]  # (N, 1)
        if self.hps.transform_with_shear:
            transform_output_shear_x_angle = transform_output[:, 5:6]  # (N, 1)
            transform_output_shear_y_angle = transform_output[:, 6:7]  # (N, 1)

        # Translation
        pred_window_translate = torch.tanh(transform_output_translation)  # (N, 2), [-1.0, 1.0]
        pred_window_translate = pred_window_translate.unsqueeze(dim=1) * (curr_window_size / 2.0)  # (N, 1, 2), in full size
        pred_cursor_position_tar = cursor_position_loop_tar * self.image_size + pred_window_translate  # (N, 1, 2), in full size
        pred_cursor_position_tar = pred_cursor_position_tar / float(self.image_size)  # (N, 1, 2), [0.0, 1.0]

        # Scaling
        pred_window_scaling_times_tar = torch.tanh(transform_output_scaling)  # (N, 2), [-1.0, 1.0]
        pred_window_scaling_times_tar = (pred_window_scaling_times_tar + 1.0) / 2.0 * self.hps.window_size_scaling_times_tar[1]  # (N, 2), [0.0, 2.0]
        pred_window_scaling_times_tar = torch.clamp(pred_window_scaling_times_tar,
                                                    self.hps.window_size_scaling_times_tar[0], self.hps.window_size_scaling_times_tar[1])  # (N, 2), [0.2, 2.0]

        curr_window_size_tar_pred = pred_window_scaling_times_tar.unsqueeze(dim=1) * curr_window_size  # (N, 1, 2), in full size
        curr_window_size_tar_pred = torch.max(curr_window_size_tar_pred, torch.tensor(self.hps.window_size_min).float().cuda())
        curr_window_size_tar_pred = torch.min(curr_window_size_tar_pred, torch.tensor(self.image_size * 1.5).float().cuda())

        # Rotation
        if self.hps.transform_with_rotation:
            pred_window_rotate_angle_tar = torch.tanh(transform_output_rotate_angle)  # (N, 1), [-1.0, 1.0]
            pred_window_rotate_angle_tar = torch.mul(pred_window_rotate_angle_tar, 180.0)  # (N, 1), [-180.0, 180.0]
        else:
            pred_window_rotate_angle_tar = None

        if self.hps.transform_with_shear:
            pred_window_shear_x_angle_tar = torch.tanh(transform_output_shear_x_angle)  # (N, 1), [-1.0, 1.0]
            pred_window_shear_x_angle_tar = torch.mul(pred_window_shear_x_angle_tar, 90.0)  # (N, 1), [-90.0, 90.0]
            pred_window_shear_y_angle_tar = torch.tanh(transform_output_shear_y_angle)  # (N, 1), [-1.0, 1.0]
            pred_window_shear_y_angle_tar = torch.mul(pred_window_shear_y_angle_tar, 90.0)  # (N, 1), [-90.0, 90.0]
        else:
            pred_window_shear_x_angle_tar = None
            pred_window_shear_y_angle_tar = None

        ## crop the target again
        crop_inputs_tar = torch.cat([target_images, target_components], dim=-1)  # (N, H, W, *)
        cropped_outputs_tar = image_cropping_stn(pred_cursor_position_tar, crop_inputs_tar, self.image_size, self.hps.raster_size, curr_window_size_tar_pred,
                                                 rotation_angle=pred_window_rotate_angle_tar,
                                                 shear_x_angle=pred_window_shear_x_angle_tar, shear_y_angle=pred_window_shear_y_angle_tar)
        curr_patch_image_tar = cropped_outputs_tar[:, :, :, 0:1]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]
        curr_patch_component_tar = cropped_outputs_tar[:, :, :, 1:2]  # (N, raster_size, raster_size, 1), [0.0-stroke, 1.0-BG]

        curr_patch_image_tar_trans = torch.squeeze(curr_patch_image_tar, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        curr_patch_component_tar_trans = torch.squeeze(curr_patch_component_tar, dim=-1)  # (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        return curr_patch_image_ref_out, curr_patch_component_ref_out, \
               curr_patch_image_tar_out, curr_patch_component_tar_out, \
               curr_patch_image_tar_trans, curr_patch_component_tar_trans, \
               pred_cursor_position_tar, curr_window_size_tar_pred, pred_window_rotate_angle_tar, \
               pred_window_shear_x_angle_tar, pred_window_shear_y_angle_tar, curr_window_size

    def build_encoder_transform(self, patch_image_ref, patch_component_ref, patch_image_tar):
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

        if self.hps.enc_model_transform == 'combined':
            if self.hps.use_ref_ori_img:
                batch_input = torch.cat([patch_images_ref, patch_components_ref, patch_images_tar], dim=1)  # (N, 3, raster_size, raster_size), [-1.0-stroke, 1.0-BG]
            else:
                batch_input = torch.cat([patch_components_ref, patch_images_tar], dim=1)  # (N, 2, raster_size, raster_size), [-1.0-stroke, 1.0-BG]

            if self.hps.add_coordconv:
                batch_input = add_coords(batch_input, self.coordconv_input)  # (N, in_dim + 2, in_H, in_W)
            output = self.encoder_transform(batch_input)  # (N, z_size)
        else:
            raise Exception('Unknown enc_model_transform:', self.hps.enc_model_transform)

        return output

    def build_decoder_transform(self, dec_input):
        """
        :param dec_input: (N, in_dim)
        :return:
        """
        h_output = self.decoder_transform(dec_input)
        next_state = None

        return h_output, next_state


class FullModel(object):
    def __init__(self, hps, train_set, valid_set,
                 log_dir, snapshot_dir, log_img_dir):
        self.hps = hps
        self.train_set = train_set
        self.valid_set = valid_set
        self.log_dir = log_dir
        self.snapshot_dir = snapshot_dir
        self.log_img_dir = log_img_dir

        self.transformation_model = Transformation_Model(hps)

        self.perceptual_model = VGG_Slim()

        params_list = []

        if self.hps.multi_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.transformation_model = nn.DataParallel(self.transformation_model)
            self.perceptual_model = nn.DataParallel(self.perceptual_model)

            params_list.append({'params': self.transformation_model.module.parameters()})
        else:
            params_list.append({'params': self.transformation_model.parameters()})
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
            gen_num_param = print_model_variables(self.transformation_model.module.named_parameters(), 'Transformation_Model')
            vgg_num_param = print_model_variables(self.perceptual_model.module.named_parameters(), 'Perceptual model')
        else:
            gen_num_param = print_model_variables(self.transformation_model.named_parameters(), 'Transformation_Model')
            vgg_num_param = print_model_variables(self.perceptual_model.named_parameters(), 'Perceptual model')
        total_num_param = gen_num_param
        total_num_param += vgg_num_param
        print('Total trainable variables %i.' % gen_num_param)

        # print('## Trainable variables:')
        # for param_group in self.optimizer.param_groups:
        #     print(param_group["params"])

        # setup tensorboards
        train_summary_writer = Logger(self.log_dir)

        mean_perc_relu_losses = [0.0 for _ in range(len(self.hps.perc_loss_layers))]

        if self.use_cuda:
            self.transformation_model = self.transformation_model.cuda()
            self.perceptual_model = self.perceptual_model.cuda()

        start = time.time()

        self.perceptual_model.eval()

        for step in range(self.start_step, self.hps.num_steps):
            # print('## Step:', step)
            self.transformation_model.train()

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
                output_values = ((step + 1), curr_learning_rate,
                                 train_cost.item(), raster_cost.item(),
                                 time_taken)
                output_log = output_format % output_values
                print(output_log)
                tf.get_logger().info(output_log)
                start = time.time()

            if (step + 1) % self.hps.log_img_every == 0:
                self.transformation_model.eval()
                self.save_log_images(self.valid_set, self.log_img_dir, (step + 1))

            if (step + 1) % self.hps.save_every == 0:
                self.save_model(step_num=step + 1, save_root=self.snapshot_dir)
                print('iter:', step + 1, ', mean_perc_relu_losses', mean_perc_relu_losses)
                filename = os.path.join(self.snapshot_dir, 'mean_perc_relu_losses.txt')
                write_txt = 'iter: ' + str(step + 1) + ', ' + str(mean_perc_relu_losses) + '\n'
                with open(filename, "a", encoding="utf-8") as f:
                    f.write(write_txt)
                f.close()

        # save model for final step
        self.save_model(step_num=self.hps.num_steps, save_root=self.snapshot_dir)
        print('iter:', self.hps.num_steps, ', mean_perc_relu_losses', mean_perc_relu_losses)
        filename = os.path.join(self.snapshot_dir, 'mean_perc_relu_losses.txt')
        write_txt = 'iter: ' + str(self.hps.num_steps) + ', ' + str(mean_perc_relu_losses) + '\n'
        with open(filename, "a", encoding="utf-8") as f:
            f.write(write_txt)
        f.close()

    def train_step(self, step, data_set, perc_loss_mean_list):
        reference_images, reference_components, target_images, target_components, reference_centerpoints, reference_centerpoints_offset, base_window_size, _, _ = \
            data_set.get_batch(self.use_cuda)
        # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
        # reference_components: (N, H, W, 1), [0-stroke, 1-BG]
        # target_images: (N, H, W, 1), [0-stroke, 1-BG]
        # target_components: (N, H, W, 1), [0-stroke, 1-BG]
        # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
        # reference_centerpoints_offset: (N, 1, 2), in [0.0, 1.0]
        # base_window_size: (N, 1), in [0.0, 1.0]

        image_size = reference_images.size()[1]

        _, patch_component_reference, _, _, _, patch_component_target_transformed, _, _, _, _, _, _ = \
            self.transformation_model(reference_images=reference_images, reference_components=reference_components,
                                      target_images=target_images, target_components=target_components,
                                      centerpoints_pos_ref=reference_centerpoints,
                                      centerpoints_pos_tar=reference_centerpoints_offset,
                                      base_window_size=base_window_size,
                                      image_size=image_size,
                                      model_mode='train')
        # patch_component_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
        # patch_component_target_transformed: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

        perc_map_pred = self.perceptual_model(patch_component_target_transformed)
        perc_map_gt = self.perceptual_model(patch_component_reference)

        raster_cost, perc_relu_losses_raw, perc_relu_losses_norm = \
            self.get_raster_loss(step, patch_component_target_transformed, patch_component_reference,
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
                curr_relu_mean = (perc_loss_mean_list[loop_i] * last_step_num + perc_relu_loss_raw) / (last_step_num + 1.0)
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
                reference_images, reference_components, target_images, target_components, reference_centerpoints, reference_centerpoints_offset, base_window_size, image_ids, _ = \
                    data_set.get_batch(self.use_cuda, batch_idx=batch_i, all_example=False)
                # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_components: (N, H, W, 1), [0-stroke, 1-BG]
                # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                # target_components: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
                # reference_centerpoints_offset: (N, 1, 2), in [0.0, 1.0]
                # base_window_size: (N, 1), in [0.0, 1.0]

                image_size = reference_images.size()[1]

                patch_reference, patch_component_reference, patch_target_original, patch_component_target_original, \
                    patch_target_transformed, patch_component_target_transformed, _, _, _, _, _, _ = \
                    self.transformation_model(reference_images=reference_images, reference_components=reference_components,
                                              target_images=target_images, target_components=target_components,
                                              centerpoints_pos_ref=reference_centerpoints,
                                              centerpoints_pos_tar=reference_centerpoints_offset,
                                              base_window_size=base_window_size,
                                              image_size=image_size,
                                              model_mode='valid')
                # patch_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_component_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_target_original: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_component_target_original: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_target_transformed: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_component_target_transformed: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

                patch_reference_np = patch_reference.cpu().data.numpy()
                patch_component_reference_np = patch_component_reference.cpu().data.numpy()
                patch_target_original_np = patch_target_original.cpu().data.numpy()
                patch_component_target_original_np = patch_component_target_original.cpu().data.numpy()
                patch_target_transformed_np = patch_target_transformed.cpu().data.numpy()
                patch_component_target_transformed_np = patch_component_target_transformed.cpu().data.numpy()

                assert len(image_ids) == 1
                img_index = image_ids[0]

                for c_i in range(patch_reference_np.shape[0]):
                    patch_reference_np_i = save_image(patch_reference_np[c_i], save_root,
                                                           'ref-0' + str(img_index) + '-' + str(c_i) + '.png')
                    patch_component_reference_np_i = save_image(patch_component_reference_np[c_i], save_root,
                                                                     'ref_comp-0' + str(img_index) + '-' + str(c_i) + '.png')

                    patch_target_original_np_i = save_image(patch_target_original_np[c_i], save_root,
                                                                 'tar_ori-0' + str(img_index) + '-' + str(c_i) + '.png')
                    patch_component_target_original_np_i = save_image(patch_component_target_original_np[c_i], save_root,
                                                                           'tar_comp_ori-0' + str(img_index) + '-' + str(c_i) + '.png')

                    save_image_overlap(patch_target_transformed_np[c_i], patch_reference_np_i, save_root,
                                            'tar_trans-0' + str(img_index) + '-' + str(c_i) + '-step=' + str(step_num) + '.png')
                    save_image_overlap(patch_component_target_transformed_np[c_i], patch_component_reference_np_i, save_root,
                                            'tar_comp_trans-0' + str(img_index) + '-' + str(c_i) + '-step=' + str(step_num) + '.png')

    def save_model(self, step_num, save_root):
        if self.use_cuda:
            self.transformation_model.cpu()

        save_dict = {}

        if self.hps.multi_gpu:
            model_state_dict = self.transformation_model.module.state_dict()
        else:
            model_state_dict = self.transformation_model.state_dict()
        # print('model_state_dict')
        # print(model_state_dict.keys())

        save_dict.update(model_state_dict)

        save_path = os.path.join(save_root, "sketch_transform_" + str(step_num) + ".pkl")
        torch.save(save_dict, save_path)
        print('Saved model:', save_path)
        if self.use_cuda:
            self.transformation_model.cuda()

    def evaluate(self, load_trained_weights=False, occluded_only=False):
        print('-' * 100)
        print('Evaluation begins ...')

        if load_trained_weights:
            print('-' * 100)
            trained_transformation_model_path = os.path.join(self.snapshot_dir, "sketch_transform_" + str(self.hps.num_steps) + ".pkl")
            if self.hps.multi_gpu:
                load_weights(trained_transformation_model_path, self.transformation_model.module)
                load_weights(self.hps.perceptual_model_path, self.perceptual_model.module)
            else:
                load_weights(trained_transformation_model_path, self.transformation_model)
                load_weights(self.hps.perceptual_model_path, self.perceptual_model)
            print('-' * 100)

            if self.use_cuda:
                self.transformation_model = self.transformation_model.cuda()
                self.perceptual_model = self.perceptual_model.cuda()

        self.transformation_model.eval()
        self.perceptual_model.eval()

        self.valid_set.batch_size = 1
        batch_num = self.valid_set.example_num // self.valid_set.batch_size
        print('batch_num:', batch_num)

        perc_loss_layers_eval = self.hps.perc_loss_layer_eval

        perc_score_set = {}
        for perc_layer in perc_loss_layers_eval:
            perc_score_set[perc_layer] = []
        total_component_num = 0

        with torch.no_grad():
            for batch_i in range(batch_num):
                print('# batch_i', batch_i)
                batch_data = self.valid_set.get_batch(self.use_cuda, batch_idx=batch_i, all_example=True, occluded_only=occluded_only)
                if batch_data is None:
                    continue

                reference_images, reference_components, target_images, target_components, \
                    reference_centerpoints, reference_centerpoints_offset, base_window_size, _, _ = batch_data
                # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_components: (N, H, W, 1), [0-stroke, 1-BG]
                # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                # target_components: (N, H, W, 1), [0-stroke, 1-BG]
                # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
                # reference_centerpoints_offset: (N, 1, 2), in [0.0, 1.0]
                # base_window_size: (N, 1), in [0.0, 1.0]

                component_num = reference_images.size()[0]
                total_component_num += component_num

                image_size = reference_images.size()[1]

                _, patch_component_reference, _, _, _, patch_component_target_transformed, _, _, _, _, _, _ = \
                    self.transformation_model(reference_images=reference_images, reference_components=reference_components,
                                              target_images=target_images, target_components=target_components,
                                              centerpoints_pos_ref=reference_centerpoints,
                                              centerpoints_pos_tar=reference_centerpoints_offset,
                                              base_window_size=base_window_size,
                                              image_size=image_size,
                                              model_mode='eval')
                # patch_component_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                # patch_component_target_transformed: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

                perc_map_pred = self.perceptual_model(patch_component_target_transformed)
                perc_map_gt = self.perceptual_model(patch_component_reference)

                _, perc_relu_losses_raw, _ = \
                    self.get_raster_loss(0, patch_component_target_transformed, patch_component_reference,
                                         loss_type=self.hps.raster_loss_base_type,
                                         return_map_pred=perc_map_pred, return_map_gt=perc_map_gt,
                                         raster_perc_loss_layer=perc_loss_layers_eval,
                                         perc_loss_mean_list=[0.0 for _ in range(len(perc_loss_layers_eval))])
                # perc_relu_losses_raw: (n_layer)

                # Perceptual score (PS): use a single layer
                for layer_i in range(len(perc_loss_layers_eval)):
                    perc_score = perc_relu_losses_raw[layer_i] * float(component_num)
                    perc_score = perc_score.cpu().data.numpy()
                    perc_score_set[perc_loss_layers_eval[layer_i]].append(perc_score)

            print('total_component_num', total_component_num)
            tf.get_logger().info('total_component_num: ' + str(total_component_num))
            for layer_i in range(len(perc_loss_layers_eval)):
                perc_score_avg = np.sum(perc_score_set[perc_loss_layers_eval[layer_i]]) / float(total_component_num)
                print('Perceptual score (PS):', perc_loss_layers_eval[layer_i], ':', perc_score_avg * 100.0, 'e-2')
                tf.get_logger().info('Perceptual score (PS): ' + perc_loss_layers_eval[layer_i] + ': ' + str(perc_score_avg * 100.0) + ' e-2')
            print('snapshot_dir:', self.snapshot_dir)
            print('win size =', self.hps.window_size_scaling_ref)
            print('window_size_min =', self.hps.window_size_min)
            print('perc_loss_layer_eval =', self.hps.perc_loss_layer_eval)

    def inference(self, save_root, show_data='selected'):
        print('-' * 100)
        print('Inference begins ...')

        trained_transformation_model_path = os.path.join(self.snapshot_dir, "sketch_transform_" + str(self.hps.num_steps) + ".pkl")
        if self.hps.multi_gpu:
            load_weights(trained_transformation_model_path, self.transformation_model.module)
        else:
            load_weights(trained_transformation_model_path, self.transformation_model)

        if self.use_cuda:
            self.transformation_model = self.transformation_model.cuda()

        self.transformation_model.eval()

        if show_data == 'all':
            show_num = self.valid_set.example_num
            is_save_image = False
            batch_idx_offsets = [0]
            data_split = 'val'
            occluded_only = False
        elif show_data == 'occluded':
            show_num = 50
            is_save_image = True
            batch_idx_offsets = [0, 737]
            data_split = 'val'
            occluded_only = True
        elif show_data == 'selected':
            show_num = 20
            is_save_image = True
            batch_idx_offsets = [0, 737]
            data_split = 'val'
            occluded_only = False
        else:
            raise Exception('Unknown show_data:', show_data)
        batch_num = self.valid_set.example_num // self.valid_set.batch_size
        print('batch_num:', batch_num)

        with (torch.no_grad()):
            for batch_idx_offset in batch_idx_offsets:
                show_i = 0
                for batch_i in range(batch_num):
                    print('# batch_i', batch_i)

                    batch_data = self.valid_set.get_batch(self.use_cuda, batch_idx=batch_i, all_example=True, batch_idx_offset=batch_idx_offset,
                                                          occluded_only=occluded_only)
                    if batch_data is None:
                        continue

                    reference_images, reference_components, target_images, target_components, \
                        reference_centerpoints, reference_centerpoints_offset, base_window_size, image_ids, component_ids = batch_data
                    # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
                    # reference_components: (N, H, W, 1), [0-stroke, 1-BG]
                    # target_images: (N, H, W, 1), [0-stroke, 1-BG]
                    # target_components: (N, H, W, 1), [0-stroke, 1-BG]
                    # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
                    # reference_centerpoints_offset: (N, 1, 2), in [0.0, 1.0]
                    # base_window_size: (N, 1), in [0.0, 1.0]

                    image_size = reference_images.size()[1]
                    if target_components is None:
                        target_components = reference_components

                    assert len(image_ids) == 1
                    img_index = image_ids[0]
                    selected_dataset_name = img_index[:img_index.find('-')]
                    selected_img_index = img_index[img_index.find('-') + 1:]

                    if not is_save_image:
                        transform_params_save_base = os.path.join(self.hps.dataset_base, selected_dataset_name + '_512', data_split,
                                                                    'component_transform_params',
                                                                    self.hps.workspace + '-[c_min=' + str(self.hps.window_size_min) + ']')
                        if self.hps.use_optical_flow:
                            transform_params_save_base += '-[optical]'
                        os.makedirs(transform_params_save_base, exist_ok=True)
                        transform_params_save_path = os.path.join(transform_params_save_base, selected_img_index + '.jsonl')
                        if os.path.exists(transform_params_save_path):
                            os.remove(transform_params_save_path)

                    patch_reference, patch_component_reference, patch_target_original, patch_component_target_original, \
                        patch_target_transformed, patch_component_target_transformed, \
                        pred_cursor_position_tar, pred_window_size_tar, pred_rotate_angle_tar, \
                        pred_shear_x_angle_tar, pred_shear_y_angle_tar, component_win_size = \
                        self.transformation_model(reference_images=reference_images, reference_components=reference_components,
                                                  target_images=target_images, target_components=target_components,
                                                  centerpoints_pos_ref=reference_centerpoints,
                                                  centerpoints_pos_tar=reference_centerpoints_offset,
                                                  base_window_size=base_window_size,
                                                  image_size=image_size,
                                                  model_mode='inference')
                    # patch_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                    # patch_component_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                    # patch_target_original: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                    # patch_component_target_original: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                    # patch_target_transformed: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
                    # patch_component_target_transformed: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

                    # pred_cursor_position_tar: (N, 1, 2), [0.0, 1.0], relative to image size
                    # pred_window_size_tar: (N, 1, 2), in image size
                    # pred_rotate_angle_tar: (N, 1), [-180.0, 180.0]
                    # pred_shear_x_angle_tar / pred_shear_y_angle_tar: (N, 1), [-90.0, 90.0]
                    # component_win_size: (N, 1, 2), in full size

                    reference_centerpoints = reference_centerpoints.squeeze(dim=1)  # (N, 2)
                    pred_cursor_position_tar = pred_cursor_position_tar.squeeze(dim=1)  # (N, 2)
                    pred_window_size_tar = pred_window_size_tar.squeeze(dim=1)  # (N, 2)
                    pred_rotate_angle_tar = pred_rotate_angle_tar.squeeze(dim=1)  # (N)
                    pred_shear_x_angle_tar = pred_shear_x_angle_tar.squeeze(dim=1)  # (N)
                    pred_shear_y_angle_tar = pred_shear_y_angle_tar.squeeze(dim=1)  # (N)
                    component_win_size = component_win_size.squeeze(dim=1)  # (N, 2)

                    patch_reference_np = patch_reference.cpu().data.numpy()
                    patch_component_reference_np = patch_component_reference.cpu().data.numpy()
                    patch_target_original_np = patch_target_original.cpu().data.numpy()
                    patch_component_target_original_np = patch_component_target_original.cpu().data.numpy()
                    patch_target_transformed_np = patch_target_transformed.cpu().data.numpy()
                    patch_component_target_transformed_np = patch_component_target_transformed.cpu().data.numpy()

                    reference_centerpoints_np = reference_centerpoints.cpu().data.numpy()
                    pred_cursor_position_tar_np = pred_cursor_position_tar.cpu().data.numpy()
                    pred_window_size_tar_np = pred_window_size_tar.cpu().data.numpy()
                    pred_rotate_angle_tar_np = pred_rotate_angle_tar.cpu().data.numpy()
                    pred_shear_x_angle_tar_np = pred_shear_x_angle_tar.cpu().data.numpy()
                    pred_shear_y_angle_tar_np = pred_shear_y_angle_tar.cpu().data.numpy()
                    component_win_size_np = component_win_size.cpu().data.numpy()

                    assert len(component_ids) == patch_reference_np.shape[0]
                    for c_i in range(patch_reference_np.shape[0]):
                        component_idx_real = component_ids[c_i]
                        if is_save_image:
                            patch_reference_np_i = save_image(patch_reference_np[c_i], save_root,
                                                              'ref-0' + str(img_index) + '-' + str(component_idx_real) + '.png')
                            patch_component_reference_np_i = save_image(patch_component_reference_np[c_i], save_root,
                                                                        'ref_comp-0' + str(img_index) + '-' + str(component_idx_real) + '.png')

                            patch_target_original_np_i = save_image(patch_target_original_np[c_i], save_root,
                                                                    'tar_ori-0' + str(img_index) + '-' + str(component_idx_real) + '.png')
                            patch_component_target_original_np_i = save_image(patch_component_target_original_np[c_i], save_root,
                                                                              'tar_comp_ori-0' + str(img_index) + '-' + str(component_idx_real) + '.png')

                            save_image_overlap(patch_target_original_np[c_i], patch_reference_np_i, save_root,
                                               'tar_ori_vis-0' + str(img_index) + '-' + str(component_idx_real) + '.png')
                            save_image_overlap(patch_component_target_original_np[c_i], patch_component_reference_np_i, save_root,
                                               'tar_comp_ori_vis-0' + str(img_index) + '-' + str(component_idx_real) + '.png')

                            save_image_overlap(patch_target_transformed_np[c_i], patch_reference_np_i, save_root,
                                               'tar_trans-0' + str(img_index) + '-' + str(component_idx_real) + '.png')
                            save_image_overlap(patch_target_transformed_np[c_i], patch_component_reference_np_i, save_root,
                                               'tar_trans_comp-0' + str(img_index) + '-' + str(component_idx_real) + '.png')
                            save_image_overlap(patch_component_target_transformed_np[c_i], patch_component_reference_np_i, save_root,
                                               'tar_comp_trans-0' + str(img_index) + '-' + str(component_idx_real) + '.png')

                        if not is_save_image:
                            transform_params_data = {}
                            transform_params_data['component_index'] = component_idx_real
                            transform_params_data['component_center'] = reference_centerpoints_np[c_i].tolist()
                            transform_params_data['component_win_size'] = component_win_size_np[c_i].tolist()
                            transform_params_data['pred_cursor'] = pred_cursor_position_tar_np[c_i].tolist()
                            transform_params_data['pred_window_size'] = pred_window_size_tar_np[c_i].tolist()
                            transform_params_data['pred_rotate_angle'] = pred_rotate_angle_tar_np[c_i].tolist()
                            transform_params_data['pred_shear_x_angle'] = pred_shear_x_angle_tar_np[c_i].tolist()
                            transform_params_data['pred_shear_y_angle'] = pred_shear_y_angle_tar_np[c_i].tolist()
                            with jsonlines.open(transform_params_save_path, mode='a') as json_writer:
                                json_writer.write(transform_params_data)

                    show_i += 1
                    if show_i >= show_num:
                        break

    def inference_real(self, save_root, data_base):
        # print('-' * 100)
        print('Inference of [Global Layer Transformation (1/4)] begins ...')

        trained_transformation_model_path = os.path.join(self.snapshot_dir, "sketch_transform_" + str(self.hps.num_steps) + ".pkl")
        if self.hps.multi_gpu:
            load_weights(trained_transformation_model_path, self.transformation_model.module)
        else:
            load_weights(trained_transformation_model_path, self.transformation_model)

        if self.use_cuda:
            self.transformation_model = self.transformation_model.cuda()

        self.transformation_model.eval()

        is_save_image = False

        with (torch.no_grad()):
            batch_data = self.valid_set.get_batch(self.use_cuda, test_img_id=test_img_id)

            reference_images, reference_components, target_images, \
                reference_centerpoints, reference_centerpoints_offset, base_window_size, image_ids, component_ids = batch_data
            # reference_images: (N, H, W, 1), [0-stroke, 1-BG]
            # reference_components: (N, H, W, 1), [0-stroke, 1-BG]
            # target_images: (N, H, W, 1), [0-stroke, 1-BG]
            # reference_centerpoints: (N, 1, 2), in [0.0, 1.0]
            # reference_centerpoints_offset: (N, 1, 2), in [0.0, 1.0]
            # base_window_size: (N, 1), in [0.0, 1.0]

            image_size = reference_images.size()[1]

            assert len(image_ids) == 1
            selected_img_index = str(test_img_id)

            if not is_save_image:
                transform_params_save_base = os.path.join(data_base, 'component_transform_params',
                                                            self.hps.workspace + '-[c_min=' + str(self.hps.window_size_min) + ']')
                if self.hps.use_optical_flow:
                    transform_params_save_base += '-[optical]'
                os.makedirs(transform_params_save_base, exist_ok=True)
                transform_params_save_path = os.path.join(transform_params_save_base, selected_img_index + '.jsonl')
                if os.path.exists(transform_params_save_path):
                    os.remove(transform_params_save_path)

            patch_reference, patch_component_reference, patch_target_original, _, \
                patch_target_transformed, _, \
                pred_cursor_position_tar, pred_window_size_tar, pred_rotate_angle_tar, \
                pred_shear_x_angle_tar, pred_shear_y_angle_tar, component_win_size = \
                self.transformation_model(reference_images=reference_images, reference_components=reference_components,
                                            target_images=target_images, target_components=reference_components,
                                            centerpoints_pos_ref=reference_centerpoints,
                                            centerpoints_pos_tar=reference_centerpoints_offset,
                                            base_window_size=base_window_size,
                                            image_size=image_size,
                                            model_mode='inference')
            # patch_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
            # patch_component_reference: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
            # patch_target_original: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]
            # patch_target_transformed: (N, raster_size, raster_size), [0.0-stroke, 1.0-BG]

            # pred_cursor_position_tar: (N, 1, 2), [0.0, 1.0], relative to image size
            # pred_window_size_tar: (N, 1, 2), in image size
            # pred_rotate_angle_tar: (N, 1), [-180.0, 180.0]
            # pred_shear_x_angle_tar / pred_shear_y_angle_tar: (N, 1), [-90.0, 90.0]
            # component_win_size: (N, 1, 2), in full size

            reference_centerpoints = reference_centerpoints.squeeze(dim=1)  # (N, 2)
            pred_cursor_position_tar = pred_cursor_position_tar.squeeze(dim=1)  # (N, 2)
            pred_window_size_tar = pred_window_size_tar.squeeze(dim=1)  # (N, 2)
            pred_rotate_angle_tar = pred_rotate_angle_tar.squeeze(dim=1)  # (N)
            pred_shear_x_angle_tar = pred_shear_x_angle_tar.squeeze(dim=1)  # (N)
            pred_shear_y_angle_tar = pred_shear_y_angle_tar.squeeze(dim=1)  # (N)
            component_win_size = component_win_size.squeeze(dim=1)  # (N, 2)

            patch_reference_np = patch_reference.cpu().data.numpy()
            patch_component_reference_np = patch_component_reference.cpu().data.numpy()
            patch_target_original_np = patch_target_original.cpu().data.numpy()
            patch_target_transformed_np = patch_target_transformed.cpu().data.numpy()

            reference_centerpoints_np = reference_centerpoints.cpu().data.numpy()
            pred_cursor_position_tar_np = pred_cursor_position_tar.cpu().data.numpy()
            pred_window_size_tar_np = pred_window_size_tar.cpu().data.numpy()
            pred_rotate_angle_tar_np = pred_rotate_angle_tar.cpu().data.numpy()
            pred_shear_x_angle_tar_np = pred_shear_x_angle_tar.cpu().data.numpy()
            pred_shear_y_angle_tar_np = pred_shear_y_angle_tar.cpu().data.numpy()
            component_win_size_np = component_win_size.cpu().data.numpy()

            for c_i in range(patch_reference_np.shape[0]):
                component_idx_real = component_ids[c_i]
                if is_save_image:
                    os.makedirs(save_root, exist_ok=True)
                    patch_reference_np_i = save_image(patch_reference_np[c_i], save_root,
                                                        'ref-' + str(selected_img_index) + '-' + str(component_idx_real) + '.png')
                    patch_component_reference_np_i = save_image(patch_component_reference_np[c_i], save_root,
                                                                'ref_comp-' + str(selected_img_index) + '-' + str(component_idx_real) + '.png')

                    patch_target_original_np_i = save_image(patch_target_original_np[c_i], save_root,
                                                            'tar_ori-' + str(selected_img_index) + '-' + str(component_idx_real) + '.png')

                    save_image_overlap(patch_target_original_np[c_i], patch_reference_np_i, save_root,
                                        'tar_ori_vis-' + str(selected_img_index) + '-' + str(component_idx_real) + '.png')

                    save_image_overlap(patch_target_transformed_np[c_i], patch_reference_np_i, save_root,
                                        'tar_trans-' + str(selected_img_index) + '-' + str(component_idx_real) + '.png')

                if not is_save_image:
                    transform_params_data = {}
                    transform_params_data['component_index'] = component_idx_real
                    transform_params_data['component_center'] = reference_centerpoints_np[c_i].tolist()
                    transform_params_data['component_win_size'] = component_win_size_np[c_i].tolist()
                    transform_params_data['pred_cursor'] = pred_cursor_position_tar_np[c_i].tolist()
                    transform_params_data['pred_window_size'] = pred_window_size_tar_np[c_i].tolist()
                    transform_params_data['pred_rotate_angle'] = pred_rotate_angle_tar_np[c_i].tolist()
                    transform_params_data['pred_shear_x_angle'] = pred_shear_x_angle_tar_np[c_i].tolist()
                    transform_params_data['pred_shear_y_angle'] = pred_shear_y_angle_tar_np[c_i].tolist()
                    with jsonlines.open(transform_params_save_path, mode='a') as json_writer:
                        json_writer.write(transform_params_data)
