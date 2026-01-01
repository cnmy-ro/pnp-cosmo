import functools

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class LinearBlock(nn.Module):

    def __init__(
        self, 
        in_channels, out_channels, bias=True, 
        weight_norm_type=None, weight_norm_params=None, activation_norm_type=None, activation_norm_params=None, 
        nonlinearity='leakyrelu'):

        super().__init__()
        
        # Linear layer
        self.weight_norm_type = weight_norm_type
        linear_layer = nn.Linear(in_channels, out_channels, bias)
        if self.weight_norm_type == 'spectral':
            weight_norm_params = {} if weight_norm_params is None else weight_norm_params
            weight_norm = functools.partial(spectral_norm, **weight_norm_params)
            linear_layer = weight_norm(linear_layer)

        # Normalization layer
        if activation_norm_type == 'instance':
            affine = activation_norm_params.pop('affine', True)
            activation_norm_layer = nn.InstanceNorm1d(out_channels, affine=affine, **activation_norm_params)
        else:
            activation_norm_layer = nn.Identity()
        
        # Nonlinearity layer
        if nonlinearity == 'relu':        nonlinearity_layer = nn.ReLU()
        elif nonlinearity == 'leakyrelu': nonlinearity_layer = nn.LeakyReLU(0.2)

        # All layers
        self.layers = nn.Sequential(linear_layer, activation_norm_layer, nonlinearity_layer)

    def forward(self, input):
        return self.layers(input)


class Conv2dBlock(nn.Module):

    def __init__(
        self, 
        in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='reflect',
        weight_norm_type=None, weight_norm_params=None, activation_norm_type=None, activation_norm_params=None, 
        nonlinearity='leakyrelu'):

        super().__init__()
        
        # Conv layer
        self.weight_norm_type = weight_norm_type
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, padding_mode=padding_mode)
        if self.weight_norm_type == 'spectral':
            weight_norm_params = {} if weight_norm_params is None else weight_norm_params
            weight_norm = functools.partial(spectral_norm, **weight_norm_params)
            conv_layer = weight_norm(conv_layer)

        # Normalization layer
        if activation_norm_type == 'instance':
            affine = activation_norm_params.pop('affine', True)
            activation_norm_layer = nn.InstanceNorm2d(out_channels, affine=affine, **activation_norm_params)
        elif activation_norm_type == 'adain':
            activation_norm_layer = AdaptiveInstanceNorm2d(out_channels, **activation_norm_params)
        else:
            activation_norm_layer = nn.Identity()

        # Nonlinearity layer
        if nonlinearity == 'relu':        nonlinearity_layer = nn.ReLU()
        elif nonlinearity == 'leakyrelu': nonlinearity_layer = nn.LeakyReLU(0.2)
        elif nonlinearity == 'tanh':      nonlinearity_layer = nn.Tanh()
        elif nonlinearity is None:        nonlinearity_layer = nn.Identity()

        # All layers
        self.layers = nn.ModuleDict({'conv': conv_layer, 'norm': activation_norm_layer, 'nonlinearity': nonlinearity_layer})

        # Conditional block?
        self.conditional = \
            getattr(conv_layer, 'conditional', False) or \
            getattr(activation_norm_layer, 'conditional', False)

    def forward(self, input, *cond_inputs, **kw_cond_inputs):
        output = input
        for layer in self.layers.values():
            if getattr(layer, 'conditional', False): output = layer(output, *cond_inputs, **kw_cond_inputs)
            else:                                    output = layer(output)
        return output


class ResConv2dBlock(nn.Module):

    def __init__(
        self, 
        in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='reflect',
        weight_norm_type=None, weight_norm_params=None, activation_norm_type=None, activation_norm_params=None, 
        nonlinearity='leakyrelu'):
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Parameters
        residual_params = {
            'dilation': dilation, 'padding_mode': padding_mode,
            'activation_norm_type': activation_norm_type,
            'activation_norm_params': activation_norm_params,
            'weight_norm_type': weight_norm_type,
            'weight_norm_params': weight_norm_params,
            'padding': padding
            }

        # Residual conv blocks
        hidden_channels = min(in_channels, out_channels)
        self.conv_block_0 = Conv2dBlock(in_channels, hidden_channels, kernel_size=kernel_size, bias=bias, nonlinearity=nonlinearity, stride=stride, **residual_params)
        self.conv_block_1 = Conv2dBlock(hidden_channels, out_channels, kernel_size=kernel_size, bias=bias, nonlinearity=nonlinearity, stride=stride, **residual_params)

        # Conditional block?
        self.conditional = \
            getattr(self.conv_block_0, 'conditional', False) or \
            getattr(self.conv_block_1, 'conditional', False)

    def apply_conv_blocks(self, x, *cond_inputs, **kw_cond_inputs):
        dx = self.conv_block_0(x, *cond_inputs, **kw_cond_inputs)
        dx = self.conv_block_1(dx, *cond_inputs, **kw_cond_inputs)
        return dx

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        dx = self.apply_conv_blocks(x, *cond_inputs, **kw_cond_inputs)
        output = x + dx
        return output


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(
        self, 
        num_features, cond_dims, 
        weight_norm_type=None,
        instance_norm_params=None):

        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, **instance_norm_params)
        self.fc = LinearBlock(cond_dims, num_features * 2, bias=True, weight_norm_type=weight_norm_type)
        self.conditional = True

    def forward(self, x, y):
        y = self.fc(y)
        for _ in range(x.dim() - y.dim()): y = y.unsqueeze(-1)
        gamma, beta = y.chunk(2, 1)
        x = self.norm(x)
        x = torch.addcmul(beta, x, 1 + gamma)
        return x
