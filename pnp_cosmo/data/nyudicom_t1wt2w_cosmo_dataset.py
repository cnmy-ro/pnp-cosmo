from pathlib import Path
import sys

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
from llmr.spatial import resample_sitk, pad_to_nearest_divisible_size


FINAL_PIXEL_SPACING = (0.630208337306976, 0.630208337306976)  # mm. (W,H) format. This is the most frequent x-y spacing for T2


class NYUDicomT1WT2WCoSMoDataset(Dataset):

    def __init__(
            self,
            root, 
            fold,
            fetch_body=False,
            size_divisor=32, 
            paired=False
            ):
    
        super().__init__()
        self.root = root
        self.fetch_body = fetch_body
        self.size_divisor = size_divisor
        self.paired = paired

        # Load pair info file
        pairs_info = pd.read_csv(root / Path('crossval_splits.csv'))

        # Subjects list
        self.subjects_list = pairs_info[pairs_info[f'fold{fold}'].astype(str) == 'train']['subject'].values

    def __len__(self):
        return len(self.subjects_list)
    
    def __getitem__(self, idx):
        
        # Fetch slices
        image_t1w, image_t2w, spacing_t1w, spacing_t2w, t1w_values_range, t2w_values_range = self._fetch_slices(idx)

        # Resample to standard spacing. In paired case, the elastix registration step resamples T1W image internally.
        image_t2w, _ = self._resample_slice_to_t2w_spacing(image_t2w, spacing_t2w)
        if not self.paired:
            image_t1w, _ = self._resample_slice_to_t2w_spacing(image_t1w, spacing_t1w)

        # Generate body mask
        if self.fetch_body:
            body_t1w = generate_body_mask(image_t1w)
            if self.paired: body_t2w = body_t1w
            else:           body_t2w = generate_body_mask(image_t2w)

        # Pad
        image_t1w = pad_to_nearest_divisible_size(image_t1w, divisor=self.size_divisor, strict=False, pad_mode='reflect')
        image_t2w = pad_to_nearest_divisible_size(image_t2w, divisor=self.size_divisor, strict=False, pad_mode='reflect')
        if self.fetch_body:
            body_t1w = pad_to_nearest_divisible_size(body_t1w, divisor=self.size_divisor, strict=False, pad_mode='reflect')
            body_t2w = pad_to_nearest_divisible_size(body_t2w, divisor=self.size_divisor, strict=False, pad_mode='reflect')

        # Normalize
        image_t1w = rescale_intensity(image_t1w, from_range=t1w_values_range, to_range=(-1, 1), clip=True)
        image_t2w = rescale_intensity(image_t2w, from_range=t2w_values_range, to_range=(-1, 1), clip=True)

        # To tensor
        image_t1w = torch.tensor(image_t1w, dtype=torch.float).unsqueeze(0)
        image_t2w = torch.tensor(image_t2w, dtype=torch.float).unsqueeze(0)
        if  self.fetch_body:
            body_t1w = torch.tensor(body_t1w, dtype=torch.float).unsqueeze(0)
            body_t2w = torch.tensor(body_t2w, dtype=torch.float).unsqueeze(0)

        # Return
        example = {'image_u': image_t1w, 'image_v': image_t2w}
        if self.fetch_body:
            example.update({'body_u': body_t1w, 'body_v': body_t2w})
        return example
    
    def _fetch_slices(self, idx):
        
        # ---
        # Load volumes
        idx_b = idx
        t2w_path = self.root / Path(f"dicom/{self.subjects_list[idx_b]}/t2")
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(t2w_path))
        reader.SetFileNames(dicom_names)
        volume_t2w = reader.Execute()        
        spacing_t2w = volume_t2w.GetSpacing()
        volume_t2w = sitk2np(volume_t2w)
        
        # If paired, register T1W and cache
        if self.paired:  
            idx_a = idx
            registered_cache_dir = self.root / Path(f"precomputed/registered_t1")
            image_path_t1w_precomputed = registered_cache_dir / Path(f"crossval/{self.subjects_list[idx_a]}/t1.npy")
            if not image_path_t1w_precomputed.exists():
                # Load the unregistered volume
                t1w_path = self.root / Path(f"dicom/{self.subjects_list[idx_a]}/t1")
                dicom_names = reader.GetGDCMSeriesFileNames(str(t1w_path))
                reader.SetFileNames(dicom_names)
                volume_t1w = reader.Execute()
                spacing_t1w = volume_t1w.GetSpacing()
                volume_t1w = sitk2np(volume_t1w)
                # First resample the T2W volume to the common spacing
                volume_t2w, spacing_t2w = self._resample_volume_to_t2w_spacing(volume_t2w, spacing_t2w)
                # Then, register T1w volume to T2w volume
                volume_t1w, spacing_t1w = _register_t1w_volume_to_t2w_volume(volume_t1w, volume_t2w, spacing_t1w, spacing_t2w)
                # cache the registered image
                save_dir = image_path_t1w_precomputed.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                np.save(image_path_t1w_precomputed, volume_t1w)
            else:
                # Fetch the pre-registered T1w volume
                volume_t1w = np.load(image_path_t1w_precomputed)
                spacing_t1w = spacing_t2w
        
        else:
            idx_a = np.random.randint(len(self.subjects_list))
            t1w_path = self.root / Path(f"dicom/{self.subjects_list[idx_a]}/t1")
            dicom_names = reader.GetGDCMSeriesFileNames(str(t1w_path))
            reader.SetFileNames(dicom_names)
            volume_t1w = reader.Execute()
            spacing_t1w = volume_t1w.GetSpacing()
            volume_t1w = sitk2np(volume_t1w)
            
        # ---
        # Get intensity ranges to do normalization later
        t2w_values_range = (0, np.percentile(volume_t2w,99.5))
        t1w_values_range = (0, np.percentile(volume_t1w,99.5))
    
        # ---
        # Select slices        
        slice_idx_t2w = np.random.randint(volume_t2w.shape[0] - 4)
        if self.paired: slice_idx_t1w = slice_idx_t2w
        else:           slice_idx_t1w = np.random.randint(volume_t1w.shape[0] - 4)
        image_t2w = volume_t2w[slice_idx_t2w]
        image_t1w = volume_t1w[slice_idx_t1w]

        return image_t1w, image_t2w, spacing_t1w, spacing_t2w, t1w_values_range, t2w_values_range
    
    def _resample_slice_to_t2w_spacing(self, image, curr_spacing):
        new_spacing = FINAL_PIXEL_SPACING
        image = _resample(image, curr_spacing, new_spacing)
        return image, new_spacing
    
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
    Function adapted from `ganslate`: 
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