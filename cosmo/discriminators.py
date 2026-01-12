import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from cosmo.layers import *


class PatchDiscriminator(nn.Module):

    def __init__(
        self,
        in_channels,
        num_layers,
        num_filters,
        max_num_filters,
        kernel_size,
        activation_norm_type,
        weight_norm_type
        ):
                 
        super().__init__()
        
        padding = int(np.floor((kernel_size - 1.0) / 2))
        nonlinearity = 'leakyrelu'
        base_conv2d_block_params = {
            'kernel_size': kernel_size,
            'padding': padding,
            'weight_norm_type': weight_norm_type,
            'activation_norm_type': activation_norm_type,
            'nonlinearity': nonlinearity,
            }

        blocks = [Conv2dBlock(in_channels, num_filters, stride=2, **base_conv2d_block_params)]
        for n in range(num_layers):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            stride = 2 if n < (num_layers - 1) else 1
            blocks += [Conv2dBlock(num_filters_prev, num_filters, stride=stride, **base_conv2d_block_params)]
        blocks += [Conv2dBlock(num_filters, 1, 3, 1, padding, weight_norm_type=weight_norm_type)]
        self.model = nn.Sequential(*blocks)

    def forward(self, input):
        return self.model(input)


class MultiScalePatchDiscriminator(nn.Module):

    def __init__(
        self,
        num_discriminators=3,
        in_channels=1,
        num_layers=4,
        num_filters=64,
        max_num_filters=512,
        kernel_size=3,
        activation_norm_type=None,
        weight_norm_type=None
        ):
        
        super().__init__()

        self.discriminators = nn.ModuleList()
        for _ in range(num_discriminators):
            discriminator = PatchDiscriminator(in_channels, num_layers, num_filters, max_num_filters, kernel_size, activation_norm_type, weight_norm_type)
            self.discriminators.append(discriminator)

    def forward(self, input):
        output_list = []
        input_downsampled = input
        for discriminator in self.discriminators:
            output = discriminator(input_downsampled)
            output_list.append(output)
            input_downsampled = F.interpolate(input_downsampled, scale_factor=0.5, mode='bilinear',
                                              align_corners=True, recompute_scale_factor=True)
        return output_list