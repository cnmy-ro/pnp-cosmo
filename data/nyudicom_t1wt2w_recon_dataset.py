from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import SimpleITK as sitk
import itk
from scipy import ndimage
import cv2

sys.path.append("//wsl.localhost/Ubuntu/home/csrao/git-personal/llmr")  # Workstation
from llmr.intensity import rescale_intensity
from llmr.conversion import np2sitk, sitk2np, np2itk, itk2np
from llmr.spatial import resample_sitk
from llmr.fft import fft2c


FINAL_PIXEL_SPACING = (0.630208337306976, 0.630208337306976)  # mm. (W,H) format. This is the most frequent x-y spacing for T2


class NYUDicomT1WT2WReconVolumeDataset(Dataset):

    def __init__(
            self,
            root, 
            fold, 
            recon_contrast='t2w',
            fetch_body=False,
            accel=4,
            center_frac=0.08,
            kspace_noise_factor=0,
            ):
        
        super().__init__()
        self.root = root
        self.fetch_body = fetch_body
        self.accel = accel
        self.center_frac = center_frac
        self.kspace_noise_factor = kspace_noise_factor
        self.recon_contrast = recon_contrast

        # Load pair info file
        pairs_info = pd.read_csv(root / Path('pairs_info_crossval.csv'))

        # Subjects list
        self.subjects_list = pairs_info[pairs_info[f'fold{fold}'].astype(str) == 'test']['subject'].values
        self.subjects_list = sorted(self.subjects_list)

        self.maskgen = Uniform1DMaskGen(seed=0, undersampling_axis=2, true_accel=True, accelerations=[accel], center_fractions=[center_frac])

    def __len__(self):
        return len(self.subjects_list)
    
    def __getitem__(self, idx):
        
        # Fetch slices
        volume_t1w, volume_t2w, spacing_t1w, spacing_t2w, subject_label = self._fetch_volumes(idx)

        # Normalize T1W and T2W
        t1w_values_range = (0, np.percentile(volume_t1w, 99.5))
        t2w_values_range = (0, np.percentile(volume_t2w, 99.5))
        volume_t1w = rescale_intensity(volume_t1w, t1w_values_range, to_range=(0, 1), clip=True)
        volume_t2w = rescale_intensity(volume_t2w, t2w_values_range, to_range=(0, 1), clip=True)

        # Simulate T2W kspace        
        mask = self.maskgen.generate_mask([1, volume_t2w.shape[1], volume_t2w.shape[2]])
        mask = np.repeat(mask, axis=0, repeats=volume_t2w.shape[0])
        if self.recon_contrast == 't2w':
            kspace = fft2c(volume_t2w, axes=(-2,-1)) * mask
            noise_sd_array = self.kspace_noise_factor * volume_t2w.max(axis=(-2,-1), keepdims=True)
        else:
            kspace = fft2c(volume_t1w, axes=(-2,-1)) * mask
            noise_sd_array = self.kspace_noise_factor * volume_t1w.max(axis=(-2,-1), keepdims=True)
        noise_real = noise_sd_array * np.random.randn(*kspace.shape)
        noise_imag = noise_sd_array * np.random.randn(*kspace.shape)
        kspace += noise_real + 1j * noise_imag

        # Derive body mask
        body = []
        for i in range(volume_t2w.shape[0]):
            body_slice = generate_body_mask(volume_t2w[i])
            body.append(body_slice)
        body = np.stack(body, axis=0)

        # To tensor
        volume_t1w = torch.tensor(volume_t1w, dtype=torch.float)
        volume_t2w = torch.tensor(volume_t2w, dtype=torch.float)
        kspace = torch.tensor(kspace, dtype=torch.cfloat)
        mask = torch.tensor(mask, dtype=torch.bool)
        body = torch.tensor(body, dtype=torch.float32)
        
        # Pack and Return
        if self.recon_contrast == 't2w': example = {'image_ref': volume_t1w, 'image_gt': volume_t2w}
        else:                            example = {'image_ref': volume_t2w, 'image_gt': volume_t1w}
        example.update({'kspace': kspace, 'mask': mask, 'body': body,
                   'subject_label': subject_label, 'spacing_t1w': spacing_t1w, 'spacing_t2w': spacing_t2w})
        
        return example

    def _fetch_volumes(self, idx):
        
        # ---
        # Load volumes
        subject_label = self.subjects_list[idx]

        t2w_path = self.root / Path(f"dicom/{subject_label}/t2")
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(t2w_path))
        reader.SetFileNames(dicom_names)
        volume_t2w = reader.Execute()        
        spacing_t2w = volume_t2w.GetSpacing()
        volume_t2w = sitk2np(volume_t2w).astype(np.float32)
        # Resample to the common spacing
        volume_t2w, spacing_t2w = self._resample_volume_to_t2w_spacing(volume_t2w, spacing_t2w)
    
        # If paired, register T1W and cache
        registered_cache_dir = self.root / Path(f"precomputed/registered_t1")
        image_path_t1w_precomputed = registered_cache_dir / Path(f"crossval/{subject_label}/t1.npy")
        if not image_path_t1w_precomputed.exists():
            # Load the unregistered volume
            t1w_path = self.root / Path(f"dicom/{subject_label}/t1")
            dicom_names = reader.GetGDCMSeriesFileNames(str(t1w_path))
            reader.SetFileNames(dicom_names)
            volume_t1w = reader.Execute()
            spacing_t1w = volume_t1w.GetSpacing()
            volume_t1w = sitk2np(volume_t1w).astype(np.float32)            
            # Register T1w volume to T2w volume
            volume_t1w, spacing_t1w = _register_t1w_volume_to_t2w_volume(volume_t1w, volume_t2w, spacing_t1w, spacing_t2w)
            # cache the registered image
            save_dir = image_path_t1w_precomputed.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(image_path_t1w_precomputed, volume_t1w)
        else:
            # Fetch the pre-registered T1w volume
            volume_t1w = np.load(image_path_t1w_precomputed)
            spacing_t1w = spacing_t2w

        return volume_t1w, volume_t2w, spacing_t1w, spacing_t2w, subject_label

    def _resample_volume_to_t2w_spacing(self, volume, curr_spacing):
        new_spacing = (FINAL_PIXEL_SPACING[0], FINAL_PIXEL_SPACING[1], curr_spacing[2])
        volume = _resample(volume, curr_spacing, new_spacing)
        return volume, new_spacing


def _resample(image, curr_spacing, new_spacing):
    image = np2sitk(image, curr_spacing)
    image = resample_sitk(image, new_spacing)
    image = sitk2np(image)
    return image

def _register_t1w_volume_to_t2w_volume(volume_t1w, volume_t2w, spacing_t1w, spacing_t2w):
    volume_t1w = np2itk(volume_t1w.copy().astype(np.float32), spacing_t1w)
    volume_t2w = np2itk(volume_t2w.copy().astype(np.float32), spacing_t2w)

    parameter_object = itk.ParameterObject.New()
    rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid', 4)
    rigid_parameter_map['FinalBSplineInterpolationOrder'] = ['3']
    parameter_object.AddParameterMap(rigid_parameter_map)    
    volume_t1w_reg, transform_params = itk.elastix_registration_method(
            volume_t2w, volume_t1w,
            parameter_object=parameter_object,
            log_to_console=False)
    spacing_t1w = spacing_t2w
    volume_t1w_reg = itk2np(volume_t1w_reg)
    return volume_t1w_reg, spacing_t1w

def smooth_contour_points(contour: np.ndarray, radius: int = 3, sigma: int = 10) -> np.ndarray:
    """
    Function that smooths contour points using the approach from 
    https://stackoverflow.com/a/37536310
    
    Simple explanation: Convolve 1D gaussian filter over the points to smoothen the curve
    """
    neighbourhood = 2 * radius + 1

    # Contour length is the total number of points + extra points
    # to ensure circularity.
    contour_length = len(contour) + 2 * radius
    # Last group of points.
    offset = (len(contour) - radius)

    x_filtered, y_filtered = [], []

    for idx in range(contour_length):
        x_filtered.append(contour[(offset + idx) \
                                          % len(contour)][0][0])

        y_filtered.append(contour[(offset + idx) \
                                          % len(contour)][0][1])

    # Gaussian blur from opencv is basically applying gaussian convolution
    # filter over these points.
    x_smooth = cv2.GaussianBlur(np.array(x_filtered), (radius, 1), sigma)
    y_smooth = cv2.GaussianBlur(np.array(y_filtered), (radius, 1), sigma)

    # Add smoothened point for
    smooth_contours = []
    for idx, (x, y) in enumerate(zip(x_smooth, y_smooth)):
        if idx < len(contour) + radius:
            smooth_contours.append(np.array([x, y]))

    return np.array(smooth_contours)

def generate_body_mask(image: np.ndarray) -> np.ndarray:
    """
    Adapted from `ganslate` code: 
    https://github.com/ganslate-team/ganslate/blob/a90d92eaf041331cd3397f788cb60884cb0e176b/ganslate/data/utils/body_mask.py#L46
    """

    thresh = 0.95 * image.min() + 0.05 * image.max()
    binarized_image = np.uint8(image >= thresh)

    body_mask = np.zeros(image.shape)

    # Returns a label map with a unique integer label for each
    # connected geometrical object in the given binary array.
    # Integer labels of components start from 1. Background is 0.
    connected_components, _ = ndimage.label(binarized_image)

    # Get counts for each component in the connected component analysis
    label_counts = [
        np.sum(connected_components == label) for label in range(1,
                                                                 connected_components.max() + 1)
    ]
    max_label = np.argmax(label_counts) + 1

    # Image with largest component binary mask
    binarized_image = connected_components == max_label
    binarized_image = np.uint8(binarized_image)

    # Find contours for each binary slice
    contours, _ = cv2.findContours(binarized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Get the largest contour based on its area
    largest_contour = max(contours, key=cv2.contourArea)

    # Smooth contour so that surface irregularities are removed better
    smoothed_contour = smooth_contour_points(largest_contour)

    # Project the points onto the body_mask image, everything
    # inside the points is set to 1.
    cv2.drawContours(body_mask, [smoothed_contour], -1, 1, -1)

    return body_mask

class Uniform1DMaskGen:

    def __init__(self, seed, undersampling_axis, accelerations, center_fractions, true_accel):
        self.set_rng(seed)
        self.accelerations = accelerations
        self.undersampling_axis = undersampling_axis
        self.center_fractions = center_fractions
        self.true_accel = true_accel
    
    def set_rng(self, seed):
        self.rng = np.random.default_rng(seed)

    def generate_mask(self, shape):
        mask = np.zeros(shape[self.undersampling_axis], dtype=np.int8)
        accel, center_frac = self._choose_acceleration()
        assert accel <= 1/center_frac, "Too high center frac."
        mask = self._sample_center_region(mask, center_frac)
        mask = self._sample_outer_region(mask, accel)
        mask = self._mask_1d_to_nd(mask, shape)
        return mask.astype(bool)
    
    def _choose_acceleration(self):
        idx = self.rng.choice(len(self.center_fractions))
        return self.accelerations[idx], self.center_fractions[idx]    

    def _sample_center_region(self, mask, center_frac):
        num_total_lines = mask.shape[0]
        num_center_lines = round(num_total_lines * center_frac)
        start = (num_total_lines - num_center_lines + 1) // 2
        mask[start : start + num_center_lines] = 1
        return mask

    def _sample_outer_region(self, mask, accel):
        num_total_lines = mask.shape[0]
        num_center_lines = int(mask.sum())
        num_outer_lines = round(num_total_lines / accel - num_center_lines)
        if self.true_accel: # For exactly the same accel as specified
            outer_region_idxs = np.argwhere(mask == 0)
            sampled_outer_line_idxs = self.rng.choice(outer_region_idxs, size=num_outer_lines, replace=False)
            mask[sampled_outer_line_idxs] = 1
        else:  # fastMRI-style random sampling. Accel reflects the specified value on average and not for each individual masks. 
            prob = num_outer_lines / (num_total_lines - num_center_lines)
            outer_mask = self.rng.uniform(size=num_total_lines) < prob
            mask = np.logical_or(mask, outer_mask)
        return mask
    
    def _mask_1d_to_nd(self, mask, shape):
        expanded_shape = np.ones([len(shape)], dtype=int)
        expanded_shape[self.undersampling_axis] = mask.shape[0]
        mask = np.reshape(mask, newshape=expanded_shape)
        repetitions = list(shape)
        repetitions[self.undersampling_axis] = 1
        return np.tile(mask, reps=repetitions)
