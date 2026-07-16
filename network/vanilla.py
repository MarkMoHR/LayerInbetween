import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def general_conv2d(in_dim, output_dim, kernel_size, stride, do_norm=True, norm_type='instance_norm', padding=1,
                   atrous=False, atrous_rate=1):
    if atrous:
        conv = nn.Conv2d(in_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=atrous_rate, dilation=atrous_rate)
    else:
        conv = nn.Conv2d(in_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    if do_norm:
        if norm_type == 'instance_norm':
            norm = nn.InstanceNorm2d(output_dim, affine=True)
        elif norm_type == 'batch_norm':
            norm = nn.BatchNorm2d(output_dim)
        elif norm_type == 'layer_norm':
            norm = nn.LayerNorm(output_dim)
        else:
            raise Exception('Unknown norm_type:', norm_type)

        return nn.Sequential(
            OrderedDict([
                ('conv', conv),
                (norm_type, norm)
            ]))
    else:
        return conv


class CNN_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, input_size,
                 first_kernel_size=3, first_padding=1, use_atrous=False):
        super(CNN_Encoder, self).__init__()

        if use_atrous:
            atrou_rates = [1, 1, 2, 4, 4]
        else:
            atrou_rates = [1, 1, 1, 1, 1]

        self.cnn_enc_11 = general_conv2d(input_dim, 32, kernel_size=first_kernel_size, stride=2, padding=first_padding, atrous=use_atrous, atrous_rate=atrou_rates[0])
        self.cnn_enc_12 = general_conv2d(32, 32, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[0])

        self.cnn_enc_21 = general_conv2d(32, 64, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[1])
        self.cnn_enc_22 = general_conv2d(64, 64, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[1])

        self.cnn_enc_31 = general_conv2d(64, 128, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_32 = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_33 = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])

        self.cnn_enc_41 = general_conv2d(128, 256, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_42 = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_43 = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])

        self.cnn_enc_51 = general_conv2d(256, 512, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[4])
        self.cnn_enc_52 = general_conv2d(512, 512, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[4])
        self.cnn_enc_53 = general_conv2d(512, 512, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[4])

        assert input_size % 32 == 0
        self.feature_size = input_size // 32
        self.fc = nn.Linear(512 * self.feature_size * self.feature_size, output_dim)

    def forward(self, inputs):
        x = inputs

        x = F.relu(self.cnn_enc_11(x))
        # print('cnn_enc_11', x.size())
        x = F.relu(self.cnn_enc_12(x))
        # print('cnn_enc_12', x.size())

        x = F.relu(self.cnn_enc_21(x))
        # print('cnn_enc_21', x.size())
        x = F.relu(self.cnn_enc_22(x))
        # print('cnn_enc_22', x.size())

        x = F.relu(self.cnn_enc_31(x))
        # print('cnn_enc_31', x.size())
        x = F.relu(self.cnn_enc_32(x))
        # print('cnn_enc_32', x.size())
        x = F.relu(self.cnn_enc_33(x))
        # print('cnn_enc_33', x.size())

        x = F.relu(self.cnn_enc_41(x))
        # print('cnn_enc_41', x.size())
        x = F.relu(self.cnn_enc_42(x))
        # print('cnn_enc_42', x.size())
        x = F.relu(self.cnn_enc_43(x))
        # print('cnn_enc_43', x.size())

        x = F.relu(self.cnn_enc_51(x))
        # print('cnn_enc_51', x.size())
        x = F.relu(self.cnn_enc_52(x))
        # print('cnn_enc_52', x.size())
        x = F.relu(self.cnn_enc_53(x))
        # print('cnn_enc_53', x.size())

        # x = x.view(-1, 512 * 4 * 4)
        x = torch.reshape(x, (-1, 512 * self.feature_size * self.feature_size))
        # print('x', x.size())

        x = self.fc(x)
        # print('fc', x.size())

        return x


class CNN_SepEncoder(nn.Module):
    def __init__(self, input_dim_ref, input_dim_tar, output_dim, input_size,
                 first_kernel_size=3, first_padding=1, use_atrous=False):
        super(CNN_SepEncoder, self).__init__()

        if use_atrous:
            atrou_rates = [1, 1, 2, 4, 4]
        else:
            atrou_rates = [1, 1, 1, 1, 1]

        # reference
        self.cnn_enc_11_ref = general_conv2d(input_dim_ref, 32, kernel_size=first_kernel_size, stride=2, padding=first_padding, atrous=use_atrous, atrous_rate=atrou_rates[0])
        self.cnn_enc_12_ref = general_conv2d(32, 32, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[0])

        self.cnn_enc_21_ref = general_conv2d(32, 64, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[1])
        self.cnn_enc_22_ref = general_conv2d(64, 64, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[1])

        self.cnn_enc_31_ref = general_conv2d(64, 128, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_32_ref = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_33_ref = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])

        self.cnn_enc_41_ref = general_conv2d(128, 256, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_42_ref = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_43_ref = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])

        # target
        self.cnn_enc_11_tar = general_conv2d(input_dim_tar, 32, kernel_size=first_kernel_size, stride=2, padding=first_padding, atrous=use_atrous, atrous_rate=atrou_rates[0])
        self.cnn_enc_12_tar = general_conv2d(32, 32, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[0])

        self.cnn_enc_21_tar = general_conv2d(64, 64, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[1])
        self.cnn_enc_22_tar = general_conv2d(64, 64, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[1])

        self.cnn_enc_31_tar = general_conv2d(128, 128, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_32_tar = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])
        self.cnn_enc_33_tar = general_conv2d(128, 128, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[2])

        self.cnn_enc_41_tar = general_conv2d(256, 256, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_42_tar = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])
        self.cnn_enc_43_tar = general_conv2d(256, 256, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[3])

        self.cnn_enc_51_tar = general_conv2d(512, 512, kernel_size=3, stride=2, atrous=use_atrous, atrous_rate=atrou_rates[4])
        self.cnn_enc_52_tar = general_conv2d(512, 512, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[4])
        self.cnn_enc_53_tar = general_conv2d(512, 512, kernel_size=3, stride=1, atrous=use_atrous, atrous_rate=atrou_rates[4])

        assert input_size % 32 == 0
        self.feature_size = input_size // 32
        self.fc = nn.Linear(512 * self.feature_size * self.feature_size, output_dim)

    def forward(self, inputs_ref, inputs_tar):
        x_r = inputs_ref

        x_r11 = F.relu(self.cnn_enc_11_ref(x_r))
        x_r12 = F.relu(self.cnn_enc_12_ref(x_r11))

        x_r21 = F.relu(self.cnn_enc_21_ref(x_r12))
        x_r22 = F.relu(self.cnn_enc_22_ref(x_r21))

        x_r31 = F.relu(self.cnn_enc_31_ref(x_r22))
        x_r32 = F.relu(self.cnn_enc_32_ref(x_r31))
        x_r33 = F.relu(self.cnn_enc_33_ref(x_r32))

        x_r41 = F.relu(self.cnn_enc_41_ref(x_r33))
        x_r42 = F.relu(self.cnn_enc_42_ref(x_r41))
        x_r43 = F.relu(self.cnn_enc_43_ref(x_r42))

        x = inputs_tar

        x = F.relu(self.cnn_enc_11_tar(x))
        x = F.relu(self.cnn_enc_12_tar(x))
        x = torch.cat([x, x_r12], dim=1)

        x = F.relu(self.cnn_enc_21_tar(x))
        x = F.relu(self.cnn_enc_22_tar(x))
        x = torch.cat([x, x_r22], dim=1)

        x = F.relu(self.cnn_enc_31_tar(x))
        x = F.relu(self.cnn_enc_32_tar(x))
        x = F.relu(self.cnn_enc_33_tar(x))
        x = torch.cat([x, x_r33], dim=1)

        x = F.relu(self.cnn_enc_41_tar(x))
        x = F.relu(self.cnn_enc_42_tar(x))
        x = F.relu(self.cnn_enc_43_tar(x))
        x = torch.cat([x, x_r43], dim=1)

        x = F.relu(self.cnn_enc_51_tar(x))
        x = F.relu(self.cnn_enc_52_tar(x))
        x = F.relu(self.cnn_enc_53_tar(x))

        # x = x.view(-1, 512 * 4 * 4)
        x = torch.reshape(x, (-1, 512 * self.feature_size * self.feature_size))

        x = self.fc(x)

        return x


class MLP_Decoder(nn.Module):
    def __init__(self, input_size, output_size, zero_init):
        super(MLP_Decoder, self).__init__()
        self.input_size = input_size

        hidden_size = 128
        self.dec_fc_1 = nn.Linear(input_size, hidden_size)
        # self.dec_fc_2 = nn.Linear(hidden_size, hidden_size)
        self.dec_fc_params = nn.Linear(hidden_size, output_size)

        if zero_init == 'last':
            for (m_name, m) in self.named_modules():
                if isinstance(m, nn.Linear) and m_name == 'dec_fc_params':
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
        elif zero_init == 'all':
            for (m_name, m) in self.named_modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
        else:
            assert zero_init == 'none'
        # print('MLP_Decoder zero init:', zero_init)

    def forward(self, input_x):
        """
        :param input_x: (batch_size, input_size)
        :return:
        """
        features_1 = self.dec_fc_1(input_x)
        # features_2 = self.dec_fc_2(features_1)
        output = self.dec_fc_params(features_1)  # (N, n_out)
        return output