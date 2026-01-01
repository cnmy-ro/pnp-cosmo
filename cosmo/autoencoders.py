from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import Upsample as Upsample

from cosmo.layers import *



class SCMUNITAutoEncoder(nn.Module):
    
    def __init__(
        self,
        in_channels=1, 
        content_channels=4,
        style_latent_size=2,
        num_filters=32,
        max_num_filters=256,
        num_features_mlp=256,
        num_res_blocks=4,
        num_downsamples_style=4,
        num_downsamples_content=0,
        num_layers_mlp=8,
        content_norm_type='instance',
        style_norm_type=None,
        weight_norm_type=None,
        output_nonlinearity='tanh'
        ):

        super().__init__()

        self.style_encoder = StyleEncoder(
            in_channels=in_channels,
            style_latent_size=style_latent_size,  
            num_filters=num_filters,
            num_downsamples=num_downsamples_style, 
            padding_mode='reflect', 
            activation_norm_type=style_norm_type, 
            weight_norm_type=weight_norm_type, 
            nonlinearity='relu')

        self.content_encoder = ContentEncoder(
            in_channels=in_channels,
            content_channels=content_channels,
            num_filters=num_filters,
            max_num_filters=max_num_filters,     
            num_res_blocks=num_res_blocks,
            num_downsamples=num_downsamples_content,
            padding_mode='reflect',
            activation_norm_type=content_norm_type,
            weight_norm_type=weight_norm_type,
            nonlinearity='relu')
        
        self.decoder = Decoder(
            in_channels=in_channels,
            content_channels=content_channels,
            num_features_style=num_features_mlp,
            num_filters=num_filters,
            num_res_blocks=num_res_blocks,
            num_upsamples=num_downsamples_content,
            padding_mode='reflect',
            weight_norm_type=weight_norm_type,
            nonlinearity='relu',
            output_nonlinearity=output_nonlinearity)
        
        self.mlp = MLP(
            in_features=style_latent_size, 
            out_features=num_features_mlp, 
            latent_features=num_features_mlp, 
            num_layers=num_layers_mlp, 
            norm=None, 
            nonlinearity='relu')
        
        self.style_channels = style_latent_size

    def forward(self, image):
        content_mean, content_logvar, style = self.encode(image)
        content = content_mean + torch.exp(0.5 * content_logvar) * torch.randn_like(content_mean)
        image_self = self.decode(content, style)
        return image_self

    def encode(self, image):
        style = self.style_encoder(image)
        content_mean, content_logvar = self.content_encoder(image)
        return content_mean, content_logvar, style

    def decode(self, content, style):
        style = self.mlp(style)
        image = self.decoder(content, style)
        return image


class Decoder(nn.Module):

    def __init__(
        self,
        in_channels,
        content_channels,
        num_features_style,
        num_filters,
        num_res_blocks,
        num_upsamples,
        padding_mode,
        weight_norm_type,
        nonlinearity,
        output_nonlinearity
        ):
        
        super().__init__()
        
        adain_params = {
            'cond_dims': num_features_style,
            'instance_norm_params': {'affine': False}}
        conv_params = {
            'padding_mode': padding_mode,
            'nonlinearity': nonlinearity,
            'weight_norm_type': weight_norm_type,
            'activation_norm_type': 'adain',
            'activation_norm_params': adain_params}

        # Residual blocks with AdaIN.
        self.model = nn.ModuleList()
        self.model += [Conv2dBlock(content_channels, num_filters, 1, 1, 0, **conv_params)]
        for _ in range(num_res_blocks):
            self.model += [ResConv2dBlock(num_filters, num_filters, 3, 1, 1, **conv_params)]

        # Convolutional blocks with upsampling.
        for _ in range(num_upsamples):
            self.model += [Upsample(scale_factor=2)]
            self.model += [Conv2dBlock(num_filters, num_filters // 2, 5, 1, 2, **conv_params)]
            num_filters //= 2
        self.model += [Conv2dBlock(num_filters, in_channels, 7, 1, 3, nonlinearity=output_nonlinearity, padding_mode=padding_mode)]

    def forward(self, x, style):
        for block in self.model:
            if getattr(block, 'conditional', False): x = block(x, style)
            else:                                    x = block(x)
        return x


class MLP(nn.Module):

    def __init__(self, in_features, out_features, latent_features, num_layers, norm, nonlinearity):
                 
        super().__init__()
        model = []
        model += [LinearBlock(in_features, latent_features, activation_norm_type=norm, nonlinearity=nonlinearity)]
        for _ in range(num_layers - 2):
            model += [LinearBlock(latent_features, latent_features, activation_norm_type=norm, nonlinearity=nonlinearity)]
        model += [LinearBlock(latent_features, out_features, activation_norm_type=norm, nonlinearity=nonlinearity)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    

class ContentEncoder(nn.Module):

    def __init__(
        self,
        in_channels,
        content_channels,
        num_filters,
        max_num_filters,
        num_res_blocks,
        num_downsamples,
        padding_mode,
        activation_norm_type,
        weight_norm_type,
        nonlinearity
        ):
    
        super().__init__()
        
        conv_params = {
            'padding_mode': padding_mode,
            'activation_norm_type': activation_norm_type,
            'activation_norm_params': {'affine': True},
            'weight_norm_type': weight_norm_type,
            'nonlinearity': nonlinearity}

        backbone = []
        backbone += [Conv2dBlock(in_channels, num_filters, 7, 1, 3, **conv_params)]

        # Downsampling blocks.
        for _ in range(num_downsamples):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            backbone += [Conv2dBlock(num_filters_prev, num_filters, 4, 2, 1, **conv_params)]

        # Residual blocks.
        for _ in range(num_res_blocks):
            backbone += [ResConv2dBlock(num_filters, num_filters, 3, 1, 1, **conv_params)]        
        backbone = nn.Sequential(*backbone)
        content_mean_layer = Conv2dBlock(num_filters, content_channels, 1, 1, 0, weight_norm_type=weight_norm_type)
        content_logvar_layer = Conv2dBlock(num_filters, content_channels, 1, 1, 0, weight_norm_type=weight_norm_type)
        self.output_dim = content_channels
        self.model = nn.ModuleDict({'backbone': backbone, 'content_mean_layer': content_mean_layer, 'content_logvar_layer': content_logvar_layer})

    def forward(self, x):
        x = self.model['backbone'](x)
        content_mean = self.model['content_mean_layer'](x)
        content_logvar = self.model['content_logvar_layer'](x)
        return content_mean, content_logvar


class StyleEncoder(nn.Module):

    def __init__(
        self, 
        in_channels,
        style_latent_size,  
        num_filters,
        num_downsamples, 
        padding_mode, 
        activation_norm_type, 
        weight_norm_type, 
        nonlinearity
        
        ):

        super().__init__()

        conv_params = {
            'padding_mode': padding_mode,
            'activation_norm_type': activation_norm_type,
            'weight_norm_type': weight_norm_type,
            'nonlinearity': nonlinearity}

        model = []
        model += [Conv2dBlock(in_channels, num_filters, 7, 1, 3, **conv_params)]
        for _ in range(2):
            model += [Conv2dBlock(num_filters, 2 * num_filters, 4, 2, 1, **conv_params)]
            num_filters *= 2
        for _ in range(num_downsamples - 2):
            model += [Conv2dBlock(num_filters, num_filters, 4, 2, 1, **conv_params)]
        model += [nn.AdaptiveAvgPool2d(1)]
        model += [nn.Conv2d(num_filters, style_latent_size, 1, 1, 0)]
        self.model = nn.Sequential(*model)
        self.output_dim = num_filters

    def forward(self, x):
        return self.model(x)
    

if __name__ == '__main__':
    ae = SCMUNITAutoEncoder().to('cuda')
    num_params = sum([p.numel() for p in ae.parameters()])
    print(num_params)
    
    x = torch.zeros((1,1,128,128), device='cuda')
    cmu,clv = ae.content_encoder(x)
    s = ae.style_encoder(x)
    xr = ae.decode(cmu,s)
    