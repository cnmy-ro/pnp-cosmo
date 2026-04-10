from abc import ABC, abstractmethod
import itertools

import torch
from torch.optim import Adam

from cosmo.autoencoders import MUNITAutoEncoder, SCMUNITAutoEncoder
from cosmo.discriminators import *
from cosmo.criteria import *


class BaseCoSMo(ABC):

    def __init__(self, conf, mode='infer'):

        self.mode = mode  # train, infer
        self.conf = conf
        self.device = conf['device']

        self._init_networks()

        if self.mode == 'train':
            self._init_criteria()
            self._init_optimizers()

        self.input = None
        self.output = None
        self.losses = None


    def _init_networks(self):
        ...
        self.networks = {k: net.to(self.device) for k, net in self.networks.items()}
        if self.mode == 'train':
            self.set_net_mode_train()
        elif self.mode in ['infer']:
            self.set_net_mode_eval()


    def _init_criteria(self):
        self.criteria = {}
        self.loss_weights = {}        
        for loss_name in self.conf['criteria'].keys():
            self.loss_weights[loss_name] = self.conf['criteria'][loss_name]['weight']
        ...


    @abstractmethod
    def _init_optimizers(self):
        self.optimizers = {}
        ...


    @abstractmethod
    def get_visuals(self):
        ...


    def get_losses(self):
        return self.losses


    def get_output(self):
        return self.output


    def set_input(self, input):
        self.input = input
        self.input = {k: v.to(self.device) for k,v in self.input.items()}


    def set_net_mode_train(self):
        for k in self.networks.keys():
            self.networks[k].train()


    def set_net_mode_eval(self):
        for k in self.networks.keys():
            self.networks[k].eval()


    def set_requires_grad(self, network_keys, require=True):
        for k in network_keys:
            network = self.networks[k]
            for p in network.parameters():
                p.requires_grad = require


    @abstractmethod
    def training_step(self):
        ...


class MUNIT(BaseCoSMo):
    
    def __init__(self, conf, mode='infer'):

        self.domains = (1, 2)

        # Training options
        if mode == 'train':
            self.body_conditioned_dis = conf['body_conditioned_dis']
            self.paired_finetuning = conf['paired_finetuning']

        super().__init__(conf, mode)


    def _init_networks(self):

        # Common settings for both autoencs
        self.networks = {f'autoenc_{domain}': MUNITAutoEncoder(**self.conf['autoencoder']) for domain in self.domains}

        # Common settings for both discriminators
        if self.mode == 'train':
            discriminators = {f'dis_{domain}': MultiScalePatchDiscriminator(**self.conf['discriminator']) for domain in self.domains}
            self.networks.update(discriminators)

        super()._init_networks()


    def _init_criteria(self):
        
        super()._init_criteria()
        self.criteria['gan'] = GANLoss('lsgan')
        self.criteria['image_self'] = torch.nn.L1Loss(reduction='mean')
        self.criteria['content_self'] = torch.nn.L1Loss(reduction='mean')
        self.criteria['style_self'] = torch.nn.L1Loss(reduction='mean')

        # Optional losses
        if self.loss_weights['image_cycle'] > 0:
            self.criteria['image_cycle'] = torch.nn.L1Loss(reduction='mean')
        
        # Paired losses
        if self.paired_finetuning:
            self.criteria['image_cross'] = torch.nn.L1Loss(reduction='mean')
            self.criteria['content_cross'] = torch.nn.L1Loss(reduction='mean')

        
    def _init_optimizers(self):
        params_autoenc = itertools.chain(*[self.networks[f'autoenc_{domain}'].parameters() for domain in self.domains])
        params_dis = itertools.chain(*[self.networks[f'dis_{domain}'].parameters() for domain in self.domains])
        self.optimizers = {
            'autoenc': Adam(params_autoenc, lr=self.conf['optimizer']['lr_autoenc']),
            'dis':     Adam(params_dis, lr=self.conf['optimizer']['lr_dis'])}

    
    def get_visuals(self):

        # Images derived from image_u (and thus perfectly aligned with it)
        visuals_set_u = {}
        visuals_set_u.update({k:v.detach() for k,v in self.input.items() if k == f'image_u'})
        visuals_set_u.update({k:v.detach() for k,v in self.output.items() if k in ['image_uu', 'image_uv', 'image_uvu']})

        # Images derived from image_v (and thus perfectly aligned with it)
        visuals_set_v = {}
        visuals_set_v.update({k:v.detach() for k,v in self.input.items() if k == f'image_v'})
        visuals_set_v.update({k:v.detach() for k,v in self.output.items() if k in ['image_vv', 'image_vu', 'image_vuv']})

        visuals = {'set_u': visuals_set_u, 'set_v': visuals_set_v}
        return visuals


    def training_step(self):
        
        # Update autoencoders
        self.set_requires_grad([f'autoenc_{domain}' for domain in self.domains], True)
        self.set_requires_grad([f'dis_{domain}' for domain in self.domains], False)
        self.optimizers['autoenc'].zero_grad(set_to_none=True)
        self.optimizers['dis'].zero_grad(set_to_none=True)
        self._compute_autoencoder_loss()        
        self.losses['total'].backward()
        self.optimizers['autoenc'].step()

        # Update discriminators
        self.set_requires_grad([f'autoenc_{domain}' for domain in self.domains], False)
        self.set_requires_grad([f'dis_{domain}' for domain in self.domains], True)
        self.optimizers['dis'].zero_grad(set_to_none=True)
        self._compute_discriminator_loss()        
        self.losses['gan_dis_total'].backward()
        self.optimizers['dis'].step() 
        
        
    def _compute_autoencoder_loss(self):

        u, v = self.domains[0], self.domains[1]
        image_u, image_v = self.input['image_u'], self.input['image_v']

        # --- 
        # Compute intermediate features and output images
        self.output = {}

        # Encode input images
        content_u, style_u = self.networks[f'autoenc_{u}'].encode(image_u)
        content_v, style_v = self.networks[f'autoenc_{v}'].encode(image_v)

        # Decode within-domain
        image_uu = self.networks[f'autoenc_{u}'].decode(content_u, style_u)
        image_vv = self.networks[f'autoenc_{v}'].decode(content_v, style_v)
        self.output.update({'image_uu': image_uu, 'image_vv': image_vv})

        # Decode cross-domain
        if self.paired_finetuning:
            image_uv = self.networks[f'autoenc_{v}'].decode(content_u, style_v)
            image_vu = self.networks[f'autoenc_{u}'].decode(content_v, style_u)
        else:
            style_v_rand = torch.randn(style_v.shape, device=self.device)
            style_u_rand = torch.randn(style_u.shape, device=self.device)
            image_uv = self.networks[f'autoenc_{v}'].decode(content_u, style_v_rand)
            image_vu = self.networks[f'autoenc_{u}'].decode(content_v, style_u_rand)
        self.output.update({'image_uv': image_uv, 'image_vu': image_vu})

        # Encode translated images into content and style code        
        content_uv, style_uv = self.networks[f'autoenc_{v}'].encode(image_uv)
        content_vu, style_vu = self.networks[f'autoenc_{u}'].encode(image_vu)        

        # Cycle decode
        if self.loss_weights['image_cycle'] > 0:
            image_uvu = self.networks[f'autoenc_{u}'].decode(content_uv, style_u)
            image_vuv = self.networks[f'autoenc_{v}'].decode(content_vu, style_v)
            self.output.update({'image_uvu': image_uvu, 'image_vuv': image_vuv})
        
        # ---
        # Compute losses
        losses = {}

        # GAN loss
        image_vu_gan = image_vu.clone()
        image_uv_gan = image_uv.clone()
        
        if self.body_conditioned_dis:
            image_vu_gan = torch.cat([image_vu_gan, self.input['body_v']], dim=1)
            image_uv_gan = torch.cat([image_uv_gan, self.input['body_u']], dim=1)
        pred = self.networks[f'dis_{u}'](image_vu_gan)
        loss_gan_autoenc_u = self.criteria['gan'](pred, is_real=True)
        pred = self.networks[f'dis_{v}'](image_uv_gan)        
        loss_gan_autoenc_v = self.criteria['gan'](pred, is_real=True)
        losses['gan_autoenc'] = loss_gan_autoenc_u + loss_gan_autoenc_v
        
        # Image recon loss
        loss_image_self_u = self.criteria['image_self'](image_uu, image_u)
        loss_image_self_v = self.criteria['image_self'](image_vv, image_v)
        losses['image_self'] = loss_image_self_u + loss_image_self_v
        
        # Content recon loss
        loss_content_self_u = self.criteria['content_self'](content_uv, content_u.detach())
        loss_content_self_v = self.criteria['content_self'](content_vu, content_v.detach())
        losses['content_self'] = loss_content_self_u + loss_content_self_v

        # Style recon loss
        loss_style_self_u = self.criteria['style_self'](style_vu, style_u_rand)
        loss_style_self_v = self.criteria['style_self'](style_uv, style_v_rand)
        losses['style_self'] = loss_style_self_u + loss_style_self_v

        # Style recon loss
        if self.paired_finetuning:
            loss_style_self_u = self.criteria['style_self'](style_vu, style_u.detach())
            loss_style_self_v = self.criteria['style_self'](style_uv, style_v.detach())
        else:
            loss_style_self_u = self.criteria['style_self'](style_vu, style_u_rand)
            loss_style_self_v = self.criteria['style_self'](style_uv, style_v_rand)
        losses['style_self'] = loss_style_self_u + loss_style_self_v


        # Total
        losses['total'] = self.loss_weights['gan'] * losses['gan_autoenc'] + \
                          self.loss_weights['image_self']   * losses['image_self']   + \
                          self.loss_weights['content_self'] * losses['content_self'] + \
                          self.loss_weights['style_self']   * losses['style_self']
        
        # Paired losses
        if self.paired_finetuning:
            # Image cross
            loss_image_cross_u = self.criteria['image_cross'](image_vu, image_u)
            loss_image_cross_v = self.criteria['image_cross'](image_uv, image_v)
            losses['image_cross'] = loss_image_cross_u + loss_image_cross_v
            # Content cross
            losses['content_cross'] = self.criteria['content_cross'](content_params_u, content_params_v)
            # Update total
            losses['total'] += self.loss_weights['image_cross'] * losses['image_cross'] + \
                               self.loss_weights['content_cross'] * losses['content_cross']

        # Optional losses
        #   Cycle consistency loss
        if self.loss_weights['image_cycle'] > 0:
            loss_image_cycle_u = self.criteria['image_cycle'](image_uvu, image_u)
            loss_image_cycle_v = self.criteria['image_cycle'](image_vuv, image_v)
            losses['image_cycle'] = loss_image_cycle_u + loss_image_cycle_v
            losses['total'] += self.loss_weights['image_cycle'] * losses['image_cycle']      

        self.losses = losses


    def _compute_discriminator_loss(self):

        u, v = self.domains[0], self.domains[1]

        image_u = self.input['image_u']
        image_v = self.input['image_v']
        image_vu = self.output['image_vu'].detach()
        image_uv = self.output['image_uv'].detach()

        if self.body_conditioned_dis:
            image_u = torch.cat([image_u, self.input['body_u']], dim=1)
            image_v = torch.cat([image_v, self.input['body_v']], dim=1)
            image_vu = torch.cat([image_vu, self.input['body_v']], dim=1)
            image_uv = torch.cat([image_uv, self.input['body_u']], dim=1)

        # Domain u
        pred_real = self.networks[f'dis_{u}'](image_u)
        pred_fake = self.networks[f'dis_{u}'](image_vu)
        loss_gan_dis_u = self.criteria['gan'](pred_real, is_real=True) + self.criteria['gan'](pred_fake, is_real=False)
        
        # Domain v
        pred_real = self.networks[f'dis_{v}'](image_v)
        pred_fake = self.networks[f'dis_{v}'](image_uv)
        loss_gan_dis_v = self.criteria['gan'](pred_real, is_real=True) + self.criteria['gan'](pred_fake, is_real=False)
        
        self.losses['gan_dis_total'] = loss_gan_dis_u + loss_gan_dis_v


class StochasticContentMUNIT(MUNIT):
    
    def _init_networks(self):

        # Common settings for both autoencs
        self.networks = {f'autoenc_{domain}': SCMUNITAutoEncoder(**self.conf['autoencoder']) for domain in self.domains}

        # Common settings for both discriminators
        if self.mode == 'train':
            discriminators = {f'dis_{domain}': MultiScalePatchDiscriminator(**self.conf['discriminator']) for domain in self.domains}
            self.networks.update(discriminators)

        self.networks = {k: net.to(self.device) for k, net in self.networks.items()}
        if self.mode == 'train':
            self.set_net_mode_train()
        elif self.mode in ['infer']:
            self.set_net_mode_eval()


    def _init_criteria(self):
        
        super()._init_criteria()

        # Optional losses
        if self.loss_weights['content_kl'] > 0:
            self.criteria['content_kl'] = GaussianKLLoss()
    
        
    def _compute_autoencoder_loss(self):

        u, v = self.domains[0], self.domains[1]
        image_u, image_v = self.input['image_u'], self.input['image_v']

        # --- 
        # Compute intermediate features and output images
        self.output = {}

        # Encode input images
        content_mean_u, content_logvar_u, style_u = self.networks[f'autoenc_{u}'].encode(image_u)
        content_mean_v, content_logvar_v, style_v = self.networks[f'autoenc_{v}'].encode(image_v)

        # Decode within-domain
        content_u = content_mean_u + torch.exp(0.5 * content_logvar_u) * torch.randn_like(content_mean_u)
        content_v = content_mean_v + torch.exp(0.5 * content_logvar_v) * torch.randn_like(content_mean_v)
        image_uu = self.networks[f'autoenc_{u}'].decode(content_u, style_u)
        image_vv = self.networks[f'autoenc_{v}'].decode(content_v, style_v)
        self.output.update({'image_uu': image_uu, 'image_vv': image_vv})

        # Decode cross-domain
        if self.paired_finetuning:
            content_u2 = content_mean_u + torch.exp(0.5 * content_logvar_u) * torch.randn_like(content_mean_u)
            content_v2 = content_mean_v + torch.exp(0.5 * content_logvar_v) * torch.randn_like(content_mean_v)
            image_uv = self.networks[f'autoenc_{v}'].decode(content_u2, style_v)
            image_vu = self.networks[f'autoenc_{u}'].decode(content_v2, style_u)
        else:
            style_v_rand = torch.randn(style_v.shape, device=self.device)
            style_u_rand = torch.randn(style_u.shape, device=self.device)
            image_uv = self.networks[f'autoenc_{v}'].decode(content_u, style_v_rand)
            image_vu = self.networks[f'autoenc_{u}'].decode(content_v, style_u_rand)
        self.output.update({'image_uv': image_uv, 'image_vu': image_vu})

        # Encode translated images into content and style code        
        content_mean_uv, content_logvar_uv, style_uv = self.networks[f'autoenc_{v}'].encode(image_uv)
        content_mean_vu, content_logvar_vu, style_vu = self.networks[f'autoenc_{u}'].encode(image_vu)        

        # Cycle decode
        if self.loss_weights['image_cycle'] > 0:
            content_uv = content_mean_uv + torch.exp(0.5 * content_logvar_uv) * torch.randn_like(content_mean_uv)
            content_vu = content_mean_vu + torch.exp(0.5 * content_logvar_vu) * torch.randn_like(content_mean_vu)
            image_uvu = self.networks[f'autoenc_{u}'].decode(content_uv, style_u)
            image_vuv = self.networks[f'autoenc_{v}'].decode(content_vu, style_v)
            self.output.update({'image_uvu': image_uvu, 'image_vuv': image_vuv})
        
        # ---
        # Compute losses
        losses = {}

        # GAN loss
        image_vu_gan = image_vu.clone()
        image_uv_gan = image_uv.clone()
        
        if self.body_conditioned_dis:
            image_vu_gan = torch.cat([image_vu_gan, self.input['body_v']], dim=1)
            image_uv_gan = torch.cat([image_uv_gan, self.input['body_u']], dim=1)
        pred = self.networks[f'dis_{u}'](image_vu_gan)
        loss_gan_autoenc_u = self.criteria['gan'](pred, is_real=True)
        pred = self.networks[f'dis_{v}'](image_uv_gan)        
        loss_gan_autoenc_v = self.criteria['gan'](pred, is_real=True)
        losses['gan_autoenc'] = loss_gan_autoenc_u + loss_gan_autoenc_v
        
        # Image recon loss
        loss_image_self_u = self.criteria['image_self'](image_uu, image_u)
        loss_image_self_v = self.criteria['image_self'](image_vv, image_v)
        losses['image_self'] = loss_image_self_u + loss_image_self_v
        
        # Content recon loss
        content_params_u = torch.cat([content_mean_u, content_logvar_u], dim=1)
        content_params_uv = torch.cat([content_mean_uv, content_logvar_uv], dim=1)
        content_params_v = torch.cat([content_mean_v, content_logvar_v], dim=1)
        content_params_vu = torch.cat([content_mean_vu, content_logvar_vu], dim=1)
        loss_content_self_u = self.criteria['content_self'](content_params_uv, content_params_u.detach())
        loss_content_self_v = self.criteria['content_self'](content_params_vu, content_params_v.detach())
        losses['content_self'] = loss_content_self_u + loss_content_self_v

        # Style recon loss
        loss_style_self_u = self.criteria['style_self'](style_vu, style_u_rand)
        loss_style_self_v = self.criteria['style_self'](style_uv, style_v_rand)
        losses['style_self'] = loss_style_self_u + loss_style_self_v

        # Style recon loss
        if self.paired_finetuning:
            loss_style_self_u = self.criteria['style_self'](style_vu, style_u.detach())
            loss_style_self_v = self.criteria['style_self'](style_uv, style_v.detach())
        else:
            loss_style_self_u = self.criteria['style_self'](style_vu, style_u_rand)
            loss_style_self_v = self.criteria['style_self'](style_uv, style_v_rand)
        losses['style_self'] = loss_style_self_u + loss_style_self_v


        # Total
        losses['total'] = self.loss_weights['gan'] * losses['gan_autoenc'] + \
                          self.loss_weights['image_self']   * losses['image_self']   + \
                          self.loss_weights['content_self'] * losses['content_self'] + \
                          self.loss_weights['style_self']   * losses['style_self']
        
        # Paired losses
        if self.paired_finetuning:
            # Image cross
            loss_image_cross_u = self.criteria['image_cross'](image_vu, image_u)
            loss_image_cross_v = self.criteria['image_cross'](image_uv, image_v)
            losses['image_cross'] = loss_image_cross_u + loss_image_cross_v
            # Content cross
            losses['content_cross'] = self.criteria['content_cross'](content_params_u, content_params_v)
            # Update total
            losses['total'] += self.loss_weights['image_cross'] * losses['image_cross'] + \
                               self.loss_weights['content_cross'] * losses['content_cross']

        # Optional losses
        #   Cycle consistency loss
        if self.loss_weights['image_cycle'] > 0:
            loss_image_cycle_u = self.criteria['image_cycle'](image_uvu, image_u)
            loss_image_cycle_v = self.criteria['image_cycle'](image_vuv, image_v)
            losses['image_cycle'] = loss_image_cycle_u + loss_image_cycle_v
            losses['total'] += self.loss_weights['image_cycle'] * losses['image_cycle']
        #   KL loss on content
        if self.loss_weights['content_kl'] > 0:
            loss_content_kl_u = self.criteria['content_kl'](content_mean_u, content_logvar_u)
            loss_content_kl_v = self.criteria['content_kl'](content_mean_v, content_logvar_v)
            losses['content_kl'] = loss_content_kl_u + loss_content_kl_v
            losses['total'] += self.loss_weights['content_kl'] * losses['content_kl']        

        self.losses = losses
