import sys

import torch
from tqdm import tqdm
import pywt, ptwt

sys.path.append("//wsl.localhost/Ubuntu/home/csrao/git-personal/llmr")  # Workstation
from llmr.fft import fft2c, ifft2c
from llmr.intensity import rescale_intensity
from llmr.conversion import torch2np_clean
from llmr.metrics import nmse
from llmr.spatial import pad_to_nearest_divisible_size, unpad



@torch.no_grad()
def l1_wavelet_ista(input, config):

    # Raw data processing
    kspace, mask, csm, max_eig = input['kspace'].to(torch.cfloat), input['mask'], input['csm'].to(torch.cfloat), input['max_eig']
    ground_truth = input['ground_truth'].abs().to(torch.float)

    # Recon        
    recon_nmse = []
    dc_step_size = 1 / max_eig
    image_estim = sense2d_forward_op_hermitian(kspace, csm, mask)
    orig_shape = image_estim.shape[1:]

    for _ in tqdm(range(config['num_iters'])):
        
        # Prox update
        image_estim = pad_to_nearest_divisible_size(image_estim[0], divisor=2, strict=False)
        wt_coeffs = dwt2(image_estim)
        wt_coeffs[0] = prox_l1_norm_complex(wt_coeffs[0], config['weight'] * dc_step_size)
        for level in range(1, len(wt_coeffs)):
            for i in range(3):
                wt_coeffs[level][i] = prox_l1_norm_complex(wt_coeffs[level][i], config['weight'] * dc_step_size)
        image_estim = idwt2(wt_coeffs)
        image_estim = unpad(image_estim, orig_shape).unsqueeze(0)

        # DC update
        image_estim = image_estim - dc_step_size * sense2d_forward_op_hermitian( sense2d_forward_op(image_estim, csm, mask) - kspace, csm, mask )
        
        # Tracking
        recon_nmse.append(nmse(torch2np_clean(ground_truth.abs()), torch2np_clean(image_estim.abs())))

    output = {'image_estim': image_estim, 'recon_nmse': recon_nmse}
    return output


@torch.no_grad()
def pnp_cosmo(input, config):

    domain_ids = {'t1w': 1, 't2w': 2}
    ref_domain, recon_domain = input['ref_domain'], input['recon_domain']
    ref_domain_id, recon_domain_id = domain_ids[ref_domain], domain_ids[recon_domain]

    # Raw data processing
    kspace, mask, csm, max_eig = input['kspace'], input['mask'], input['csm'], input['max_eig']
    phase_map = input['phase_map']
    recon_intensity_range, ref_intensity_range = input['recon_intensity_range'], input['ref_intensity_range']
    pad_divisor = config['pad_divisor'] if 'pad_divisor' in config.keys() else 4

    # CoSMo data processing
    cosmo = input['cosmo']
    ground_truth = input['ground_truth'].abs()
    image_gt = mri_to_cosmo_transform_chain(ground_truth, pad_divisor, recon_intensity_range)
    image_ref = mri_to_cosmo_transform_chain(input['image_ref'], pad_divisor, ref_intensity_range)

    content_gt_mean, content_gt_logvar = cosmo.networks[f'autoenc_{recon_domain_id}'].content_encoder(image_gt)
    content_ref_mean, content_ref_logvar = cosmo.networks[f'autoenc_{ref_domain_id}'].content_encoder(image_ref)

    noise = torch.randn_like(content_gt_mean)
    content_gt = content_gt_mean + torch.exp(0.5*content_gt_logvar) * noise        
    content_ref = content_ref_mean + torch.exp(0.5*content_ref_logvar) * noise
    style_gt = cosmo.networks[f'autoenc_{recon_domain_id}'].style_encoder(image_gt)
    
    if config['ideal_content']: content_estim = content_gt
    else:                       content_estim = content_ref
    
    # Recon
    style_nmse, content_nmse, recon_nmse = [], [], []
    dc_step_size = 1 / max_eig
    cr_step_size = config['cr_step_size']    
    image_estim = sense2d_forward_op_hermitian(kspace, csm, mask, phase_map)
    orig_shape = image_estim.shape[1:]

    for i in tqdm(range(config['num_iters'])):
        
        # CC update
        image_estim = mri_to_cosmo_transform_chain(image_estim, pad_divisor, recon_intensity_range)
        style_estim = cosmo.networks[f'autoenc_{recon_domain_id}'].style_encoder(image_estim)
        image_estim = cosmo.networks[f'autoenc_{recon_domain_id}'].decode(content_estim, style_estim)
        image_estim = cosmo_to_mri_transform_chain(image_estim, orig_shape, recon_intensity_range)

        # DC update
        image_estim = image_estim - dc_step_size * sense2d_forward_op_hermitian( sense2d_forward_op(image_estim, csm, mask, phase_map) - kspace, csm, mask, phase_map )

        # Content refinement
        if config['cr_enable']:
            style_image = mri_to_cosmo_transform_chain(image_estim, pad_divisor, recon_intensity_range)
            style = cosmo.networks[f'autoenc_{recon_domain_id}'].style_encoder(style_image)
            content_estim = update_content(content_estim, style, kspace, csm, mask, phase_map, cosmo, cr_step_size, orig_shape, domain_ids[recon_domain], recon_intensity_range)

        # Tracking
        style_nmse.append(nmse(torch2np_clean(style_gt), torch2np_clean(style_estim)))
        content_nmse.append(nmse(torch2np_clean(content_gt), torch2np_clean(content_estim)))
        recon_nmse.append(nmse(torch2np_clean(ground_truth.abs()), torch2np_clean(image_estim.abs())))
        
    output = {'image_estim': image_estim, 'style_nmse': style_nmse, 'content_nmse': content_nmse, 'recon_nmse': recon_nmse}
    return output


# ---
# Utils

def l1_norm(tensor): return torch.linalg.norm(tensor.flatten(), ord=1)

def l2_norm(tensor): return torch.linalg.norm(tensor.flatten(), ord=2)

def sense2d_forward_op(image, csm, mask, phase=None):
    if phase is not None: image = image * torch.exp(1j * phase)
    kspace = fft2c(image * csm, axes=[-2,-1]) * mask
    return kspace

def sense2d_forward_op_hermitian(kspace, csm, mask, phase=None):
    coil_images = ifft2c(kspace * mask, axes=[-2,-1])
    image = torch.sum(coil_images * csm.conj(), dim=1, keepdim=True)
    if phase is not None: image = image * torch.exp(-1j * phase)
    return image

def dwt2(image, wavelet=pywt.Wavelet('db4'), wt_level=4):
    wt_coeffs_real = ptwt.wavedec2(image.real, wavelet, level=wt_level, mode='periodic')
    wt_coeffs_imag = ptwt.wavedec2(image.imag, wavelet, level=wt_level, mode='periodic')
    wt_coeffs = [wt_coeffs_real[0] + 1j*wt_coeffs_imag[0]]
    for level in range(1, wt_level + 1):
        wt_coeffs.append([])
        for i in range(3):
            wt_coeffs[level].append(wt_coeffs_real[level][i] + 1j*wt_coeffs_imag[level][i])
    return wt_coeffs

def idwt2(wt_coeffs, wavelet=pywt.Wavelet('db4')):
    wt_coeffs_real, wt_coeffs_imag = [wt_coeffs[0].real], [wt_coeffs[0].imag]
    for level in range(1, len(wt_coeffs)):
        wt_coeffs_real.append([])
        wt_coeffs_imag.append([])
        for i in range(3):
            wt_coeffs_real[level].append(wt_coeffs[level][i].real)
            wt_coeffs_imag[level].append(wt_coeffs[level][i].imag)
        wt_coeffs_real[level] = tuple(wt_coeffs_real[level])
        wt_coeffs_imag[level] = tuple(wt_coeffs_imag[level])
    image_real = ptwt.waverec2(wt_coeffs_real, wavelet)
    image_imag = ptwt.waverec2(wt_coeffs_imag, wavelet)
    image = image_real + 1j*image_imag
    return image[0]  # 1st dim is batch, and is added by waverec2. Remove it.

def flatten_wavelet_repr(wt_coeffs):
    wt_coeffs_flat = [wt_coeffs[0].flatten()]
    for level in range(1, len(wt_coeffs)):
        for i in range(3):
            wt_coeffs_flat.append(wt_coeffs[level][i].flatten())
    wt_coeffs_flat = torch.cat(wt_coeffs_flat, axis=0)
    return wt_coeffs_flat

def prox_l1_norm_complex(tensor, alpha):
    # Based on: https://stats.stackexchange.com/questions/357339/soft-thresholding-for-the-lasso-with-complex-valued-data
    return torch.exp(1j*tensor.angle()) * torch.maximum(tensor.abs() - alpha, torch.zeros_like(tensor.abs()))

def cosmo_to_mri_transform_chain(image, orig_shape, orig_range):
    image = unpad(image, orig_shape)
    if isinstance(orig_range, float): orig_range = [0., orig_range]
    image = rescale_intensity(image, from_range=[-1,1], to_range=orig_range)
    image = image + 1j*torch.zeros_like(image)
    return image

def mri_to_cosmo_transform_chain(image, pad_divisor, orig_range):
    image = image.abs()
    if isinstance(orig_range, float): orig_range = [0., orig_range]
    image = rescale_intensity(image, from_range=orig_range, to_range=[-1,1], clip=True)
    image = pad_to_nearest_divisible_size(image, divisor=pad_divisor, strict=False)
    return image

def update_content(content_estim, style_estim, kspace, csm, mask, phase_map, cosmo, weight, orig_shape, recon_domain_id, recon_intensity_range):
    
    torch.set_grad_enabled(True)
    content_estim = content_estim.detach()
    content_estim.requires_grad = True

    # Compute loss
    image_from_content = cosmo.networks[f'autoenc_{recon_domain_id}'].decode(content_estim, style_estim.detach())
    image_from_content = cosmo_to_mri_transform_chain(image_from_content, orig_shape, recon_intensity_range)
    loss = l2_norm((sense2d_forward_op(image_from_content, csm, mask, phase_map) - kspace) * mask) ** 2
    
    # Compute grad
    content_grad = torch.autograd.grad(outputs=[loss], inputs=[content_estim])[0]
    torch.set_grad_enabled(False)
    
    # GD step
    content_estim = content_estim - weight * content_grad

    return content_estim
