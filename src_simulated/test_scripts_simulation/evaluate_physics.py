# compute AES
# make 2 figures
# 2 subjects
# How to go down

# ------------------------------------------------------------
# Author: Ajay Sharma
# Purpose: Low-Field MRI → Super-Resolution Reconstruction (SRR)
# Description:
#   Loads LF .3d data, performs preprocessing, applies trained SRR model,
#   and visualizes results (denoising + super-resolution).
# ------------------------------------------------------------

import os
import sys
import glob
import time
import math
import json
import logging
import argparse
from datetime import datetime
import re

# ------------------------------------------------------------
# Third-Party Scientific Computing
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.ndimage import (
    zoom,
    gaussian_filter,
    center_of_mass,
    rotate,
    binary_dilation,
    binary_erosion,
    binary_fill_holes
)

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ------------------------------------------------------------
# Medical Imaging
# ------------------------------------------------------------
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import pydicom

# ------------------------------------------------------------
# Image Processing
# ------------------------------------------------------------
import cv2
from skimage.morphology import ball
from keras.preprocessing.image import img_to_array, load_img

# ------------------------------------------------------------
# Machine Learning / Deep Learning
# ------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, LeakyReLU,
    Activation, Concatenate, Add, UpSampling2D, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# NumPy Random Utilities
# ------------------------------------------------------------
from numpy import asarray, load, zeros, ones
from numpy.random import randint
from random import random as rand_func

# ------------------------------------------------------------
# Project Path Setup
# ------------------------------------------------------------
sys.path.insert(0, './')
sys.path.insert(0, './data_read_code')

# ------------------------------------------------------------
# Project-Specific Imports
# ------------------------------------------------------------
from src_simulated.test_scripts_simulation.evaluate_niv_lf_test import evaluate_model, predict_volume
from src_niv.metrics import psnr, ssim, mse, composite_loss
from src_niv.utils import visualize_pair

from src_simulated.cyclegan_models.config_lf import config_lf
from src_simulated.cyclegan_models.losses_cyclegan import *
from src_simulated.cyclegan_models.models import *
from src_simulated.cyclegan_models.train_utils_params_parallel import *


def resample_volume_numpy(im, current_spacing=(2.0, 2.0, 5.0), new_spacing=(1.0, 1.0, 2.0), order=3):
    """
    Resample a 3D numpy volume to the desired voxel spacing.
    
    Args:
        im (np.ndarray): 3D MRI volume (Z, Y, X)
        current_spacing (tuple): Current voxel spacing in mm (z, y, x)
        new_spacing (tuple): Desired voxel spacing in mm (z, y, x)
        order (int): Interpolation order (0=nearest, 1=linear, 3=cubic)
    
    Returns:
        np.ndarray: Resampled 3D volume
    """
    zoom_factors = [current_spacing[i] / new_spacing[i] for i in range(3)]
    print(f"Zoom factors (z, y, x): {zoom_factors}")
    
    resampled_im = zoom(im, zoom_factors, order=order)
    print(f"Original shape: {im.shape} → New shape: {resampled_im.shape}")
    
    return resampled_im

def load_lf_3d_file(lf_path):
    
    # Assuming .3d is a simple binary float32 file with known dimensions
    # You may need to adapt this to your .3d file format

    sub_folder = os.path.basename(lf_path)
    data_folder = os.path.dirname(lf_path)
    sample_data = kea3d(data_folder=data_folder, sub_folder=sub_folder)
    kspace = sample_data.kspace_gauss_filter
    im = np.abs(np.fft.fftshift(np.fft.fftn((np.fft.fftshift(kspace)))))

    acqu_path = lf_path + '/acqu.par'
    image_path_LF = lf_path + '/data.3d'
    ImageScanParams = keaProc.readPar(acqu_path)
        
    # self.LF_ref_kSpace = keaProc.readKSpace(image_path_LF)
    # LF_acq = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(self.LF_ref_kSpace)))
    
    LF_ref_kSpace = kspace
    LF_acq = im
    
    LF_ref_im = np.abs(LF_acq)
    fov_LF_ref_acq = ImageScanParams.get('FOV')
    matrix_LF_ref_acq = LF_acq.shape
    res_LF_ref_acq = np.divide(fov_LF_ref_acq, matrix_LF_ref_acq)
    print(Fore.CYAN + 'Matrix size of acquired Low Field image: ', matrix_LF_ref_acq, Style.RESET_ALL)
    print(Fore.CYAN + 'FOV of acquired LF: ', fov_LF_ref_acq, Style.RESET_ALL)
    print(Fore.CYAN + 'Resolution of acquired LF: ', res_LF_ref_acq, Style.RESET_ALL)

    num_slices = LF_acq.shape[2]

    # fig, axes = plt.subplots(2, 8, figsize=(20, 8))
    # # fig.suptitle(f'All Axial Slices for {name}\n{subject}\n{Visit_id}\n3DTSE/{subf}', fontsize=16)
    # axes = axes.flatten()

    # for i in range(16):
    #     if i < num_slices:
    #         slice_img = np.flipud(np.abs(LF_acq[:, :, i]).T)
    #         axes[i].imshow(slice_img, cmap='gray')
    #         axes[i].set_title(f'Slice {i + 1}')
    #         axes[i].axis('off')
    #     else:
    #         axes[i].axis('off')

    # plt.tight_layout()
    # # plt.savefig(f'Figures/{subject}/{fig_name}')
    # plt.show()
    # plt.close()

    return im

# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize_volume(vol, method='minmax'):
    if method=='minmax':
        vol_min, vol_max = vol.min(), vol.max()
        if vol_max - vol_min > 0:
            vol = (vol - vol_min) / (vol_max - vol_min)
        else:
            vol = np.zeros_like(vol)
    elif method=='zscore':
        mean, std = vol.mean(), vol.std()
        if std>0:
            vol = (vol - mean) / std
        else:
            vol = np.zeros_like(vol)
    return vol

def normalize_dataset(X, y, method='minmax'):
    X_norm = np.array([normalize_volume(vol, method) for vol in X])
    y_norm = np.array([normalize_volume(vol, method) for vol in y])
    return X_norm, y_norm

# ============================================================
# 🔹 1. Intensity-Based Centering
# ============================================================
def circshift_center_intensity(image):
    """
    Circularly shift a 3D MRI volume to align its intensity centroid
    with the geometric center (reduces wrap-around artifacts).
    """
    shifted = np.copy(image)
    com = np.array(center_of_mass(shifted))
    geom_center = np.array([s / 2 for s in shifted.shape])
    shift = np.round(geom_center - com).astype(int)

    for axis in range(3):
        shifted = np.roll(shifted, shift[axis], axis=axis)

    return shifted, com, geom_center, shift

# ============================================================
# 🔹 2. Fractional Shift + Rotation
# ============================================================

def circshift_3d(image, up_fraction=0.0, left_fraction=0.0, rotation_deg=0.0, axes=(1, 0)):
    """
    Apply circular (wrap-around) shift and optional rotation to a 3D MRI volume.

    Args:
        image (np.ndarray): 3D volume (H, W, D)
        up_fraction (float): Fraction of height to shift vertically.
                             Positive = up, Negative = down.
        left_fraction (float): Fraction of width to shift horizontally.
                               Positive = left, Negative = right.
        rotation_deg (float): Rotation in degrees (applied in-plane).
        axes (tuple): Axes for rotation (default=(1,0) means rotate in x–y plane).

    Returns:
        np.ndarray: Shifted and rotated 3D image.
    """
    shifted = np.copy(image)
    H, W, D = shifted.shape

    # Compute pixel shifts
    shift_y = int(H * up_fraction)
    shift_x = int(W * left_fraction)

    # Apply circular shifts
    if shift_y != 0:
        shifted = np.roll(shifted, -shift_y, axis=0)
    if shift_x != 0:
        shifted = np.roll(shifted, -shift_x, axis=1)

    # Apply rotation if requested
    if rotation_deg != 0:
        shifted = rotate(shifted, rotation_deg, axes=axes, reshape=False, order=1, mode='wrap')

    return shifted

# ============================================================
# 🔹 3. Simple Upward Shift Only
# ============================================================
def circshift_up(image, fraction=0.3):
    """
    Move image upward by a given fraction of its height.
    """
    shifted = np.copy(image)
    shift_y = int(image.shape[0] * fraction)
    shifted = np.roll(shifted, -shift_y, axis=0)
    return shifted

# ============================================================
# 🔹 4. Visualization Utility
# ============================================================
def visualize_volume(volume, title="Volume", rows=3, cmap='gray'):
    """
    Visualize all slices of a 3D volume in multiple rows.
    """
    num_slices = volume.shape[2]
    cols = int(np.ceil(num_slices / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    axes = axes.flatten()

    for i in range(num_slices):
        axes[i].imshow(volume[:, :, i], cmap=cmap)
        axes[i].set_title(f'{title} - Slice {i+1}', fontsize=8)
        axes[i].axis('off')

    for j in range(num_slices, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def pad_or_crop_volume_to_shape(volume, target_shape=(144, 144, 40)):
    """
    Pads or crops a 3D/4D volume symmetrically to match the target shape.
    
    - Pads with zeros if smaller.
    - Crops centrally if larger.

    Args:
        volume (np.ndarray): 3D (H, W, D) or 4D (B, H, W, D) volume.
        target_shape (tuple): Desired shape (H, W, D).

    Returns:
        np.ndarray: Volume with exact target shape.
        list: [(before_h, after_h), (before_w, after_w), (before_d, after_d)] padding info.
    """
    if volume.ndim == 4:
        _, h, w, d = volume.shape
    elif volume.ndim == 3:
        h, w, d = volume.shape
    else:
        raise ValueError(f"Unsupported input shape: {volume.shape}")

    target_h, target_w, target_d = target_shape

    def compute_pad_or_crop(size, target):
        if size == target:
            return (0, 0, None)
        elif size < target:  # pad
            diff = target - size
            pad_before = diff // 2
            pad_after = diff - pad_before
            return (pad_before, pad_after, None)
        else:  # crop
            crop_before = (size - target) // 2
            crop_after = crop_before + target
            return (0, 0, (crop_before, crop_after))

    pad_h = compute_pad_or_crop(h, target_h)
    pad_w = compute_pad_or_crop(w, target_w)
    pad_d = compute_pad_or_crop(d, target_d)

    # Apply cropping if needed
    if pad_h[2] is not None or pad_w[2] is not None or pad_d[2] is not None:
        h_slice = slice(pad_h[2][0], pad_h[2][1]) if pad_h[2] else slice(None)
        w_slice = slice(pad_w[2][0], pad_w[2][1]) if pad_w[2] else slice(None)
        d_slice = slice(pad_d[2][0], pad_d[2][1]) if pad_d[2] else slice(None)
        if volume.ndim == 4:
            volume = volume[:, h_slice, w_slice, d_slice]
        else:
            volume = volume[h_slice, w_slice, d_slice]

    # Apply padding if needed
    pad_h = (pad_h[0], pad_h[1])
    pad_w = (pad_w[0], pad_w[1])
    pad_d = (pad_d[0], pad_d[1])
    pad_info = [pad_h, pad_w, pad_d]

    if volume.ndim == 4:
        padded = np.pad(volume, ((0, 0), pad_h, pad_w, pad_d), mode='constant')
    else:
        padded = np.pad(volume, (pad_h, pad_w, pad_d), mode='constant')

    return padded, pad_info

def rot90_3d(volume, k=1, axes=(0, 1)):
    """
    Apply 90° rotation (k times) to all slices in a 3D volume.

    Args:
        volume (np.ndarray): 3D array (H, W, D)
        k (int): Number of 90° rotations. (1=90°, 2=180°, 3=270°)
        axes (tuple): Axes to rotate along. Default (0, 1) = in-plane rotation.

    Returns:
        np.ndarray: Rotated 3D volume.
    """
    rotated = np.zeros_like(volume)
    for i in range(volume.shape[2]):
        rotated[:, :, i] = np.rot90(volume[:, :, i], k=k, axes=axes)
    return rotated

def extract_head_mask(volume, threshold=0.1, min_size=5000, dilation_iter=2, erosion_iter=1):
    """
    Extract the head region from a low-field MRI volume using morphological operations.
    
    Steps:
        1. Threshold the image to get a rough foreground mask.
        2. Fill small holes.
        3. Keep only the largest connected component (the head).
        4. Apply morphological closing (dilation + erosion).
        5. Return the binary head mask and cleaned image.
    
    Args:
        volume (np.ndarray): 3D MRI volume (H, W, D)
        threshold (float): Threshold for binarization (0-1 normalized intensity)
        min_size (int): Minimum voxel count to retain a connected component
        dilation_iter (int): Number of dilation iterations
        erosion_iter (int): Number of erosion iterations

    Returns:
        np.ndarray: Cleaned volume (only head region)
        np.ndarray: Binary head mask
    """

    # Normalize if not already in 0–1
    if volume.max() > 1:
        volume = volume / np.max(volume)

    # Step 1: Threshold to create binary mask
    mask = volume > threshold

    # Step 2: Fill holes
    mask = binary_fill_holes(mask)

    # Step 3: Keep only the largest connected component
    labeled, num = label(mask)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        largest_label = sizes[1:].argmax() + 1  # skip background (label 0)
        mask = labeled == largest_label

    # Step 4: Morphological cleanup
    struct = ball(2)
    for _ in range(dilation_iter):
        mask = binary_dilation(mask, structure=struct)
    for _ in range(erosion_iter):
        mask = binary_erosion(mask, structure=struct)

    # Step 5: Remove small isolated components if any remain
    mask = remove_small_objects(mask, min_size=min_size)

    # Apply mask to original image
    cleaned = volume * mask

    return cleaned, mask

# # Compute Average Edge Strength (AES)
def compute_aes(image):
    # Compute gradients using Sobel filter
    from scipy.ndimage import sobel
    dx = sobel(image, axis=0)
    dy = sobel(image, axis=1)
    if image.ndim == 3: 
        dz = sobel(image, axis=2)
    else:
        dz = 0
            
    # Compute gradient magnitude
    grad_mag = np.sqrt(dx**2 + dy**2 + dz**2)
    # Average Edge Strength
    aes_value = np.mean(grad_mag)
    return aes_value

# Global flag to ensure deletion happens only once
_pngs_deleted = False

def visualize_slice(pred2, name='', output_dir='outputs_59228/trail1'):
    """
    Visualize selected slices from a 3D volume, compute AES per slice,
    and save a high-resolution multi-slice figure.
    """
    global _pngs_deleted

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # # 🔹 Delete all existing PNG files only on first run
    # if not _pngs_deleted:
    #     old_pngs = glob.glob(os.path.join(output_dir, "*.png"))
    #     for f in old_pngs:
    #         try:
    #             os.remove(f)
    #         except Exception as e:
    #             print(f"⚠️ Could not delete {f}: {e}")
    #     if old_pngs:
    #         print(f"🧹 Deleted {len(old_pngs)} existing PNG files in {output_dir}")
    #     else:
    #         print(f"🧹 No old PNGs found in {output_dir}")
    #     _pngs_deleted = True  # Mark cleanup as done

    # Define slice range
    start_slice, end_slice = 24, 27
    num_slices = end_slice - start_slice + 1

    # 🔹 Use a larger figure size for publication-quality output
    fig, axes = plt.subplots(1, num_slices, figsize=(5 * num_slices, 10))

    for i, slice_idx in enumerate(range(start_slice, end_slice + 1)):
        im = pred2[:, :, slice_idx]
        aes = compute_aes(im)

        axes[i].imshow(im, cmap='gray')
        axes[i].set_title(f"AES: {aes:.4f}", fontsize=14)
        axes[i].axis('off')

    plt.tight_layout()

    # 🔹 High-DPI save paths
    save_path_high = os.path.join(output_dir, f"{name}_all_slices_highres.png")

    # # Save standard figure
    # plt.savefig(save_path_std, bbox_inches='tight', pad_inches=0.1, dpi=200)

    # Save high-resolution figure (publication-ready)
    plt.savefig(save_path_high, bbox_inches='tight', pad_inches=0.05, dpi=600)

    plt.show()

    print(f"🖼️  Saved high-resolution figure: {save_path_high}")

def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    x = np.squeeze(x)
    if x.dtype.kind not in ("u", "i", "f"):
        x = x.astype(np.float32)
    return x

def visualize_comparison(
    im,
    pred1,
    pred2,
    name='comparison',
    output_dir='outputs_59228/trial1',
    affine=None,
    slice_range=(24, 27),
    expose_individual_png_paths=True,
    view_orthoslices=False
):
    """
    Display corresponding slices from original (im), pred1 (denoiser), and pred2 (srr)
    in a single figure: rows = versions, columns = slices.
    Saves both standard and high-resolution images.
    Also saves volumes as NIfTI (.nii.gz).
    Optionally opens orthogonal slicer windows for each saved NIfTI (loaded from disk).
    """

    os.makedirs(output_dir, exist_ok=True)

    # Convert inputs to numpy
    im_np    = _to_numpy(im)
    pred1_np = _to_numpy(pred1)
    pred2_np = _to_numpy(pred2)

    # Validate volumes
    for vol_name, vol in [('im', im_np), ('pred1', pred1_np), ('pred2', pred2_np)]:
        if vol is None:
            raise ValueError(f"{vol_name} is None.")
        if vol.ndim < 3:
            raise ValueError(f"{vol_name} must be at least 3D (H, W, D). Got shape {vol.shape}.")

    # Determine slice indices safely
    start_slice, end_slice = slice_range
    max_depth = min(im_np.shape[-1], pred1_np.shape[-1], pred2_np.shape[-1])
    if end_slice >= max_depth:
        end_slice = max_depth - 1
    if start_slice > end_slice:
        start_slice = max(0, end_slice - 3)
    slice_indices = range(start_slice, end_slice + 1)
    num_slices = len(slice_indices)

    # Helper to get a 2D slice
    def get_slice(volume, idx):
        v = np.squeeze(volume)
        if v.ndim == 2:
            return v
        if v.ndim != 3:
            v = np.squeeze(v)
            if v.ndim != 3:
                raise ValueError(f"Volume shape after squeeze is not 3D: {v.shape}")
        if idx < 0 or idx >= v.shape[-1]:
            raise IndexError(f"Slice index {idx} out of range for depth {v.shape[-1]}")
        return v[:, :, idx]

    # Create figure (3 rows × num_slices columns)
    fig, axes = plt.subplots(3, num_slices, figsize=(5 * num_slices, 12))
    if num_slices == 1:
        axes = np.atleast_2d(axes)

    titles = ['Original (im)', 'Prediction 1 (denoiser)', 'Prediction 2 (srr)']
    volumes = [im_np, pred1_np, pred2_np]

    # Plot each version per slice
    for col, idx in enumerate(slice_indices):
        for row, (vol, label) in enumerate(zip(volumes, titles)):
            ax = axes[row, col]
            sl = get_slice(vol, idx)
            ax.imshow(sl, cmap='gray')
            if col == 0:
                ax.set_ylabel(label, fontsize=14)
            if row == 0:
                ax.set_title(f"Slice {idx}", fontsize=14)
            ax.axis('off')

    plt.tight_layout()

    # Save figures (standard and high-res)
    save_path_std = os.path.join(output_dir, f"{name}_comparison.png")
    save_path_high = os.path.join(output_dir, f"{name}_comparison_highres.png")
    # plt.savefig(save_path_std, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.savefig(save_path_high, bbox_inches='tight', pad_inches=0.05, dpi=600)
    plt.show()

    # Save as NIfTI .nii.gz
    if affine is None:
        affine = np.eye(4, dtype=np.float32)

    nii_original = os.path.join(output_dir, f"{name}_original.nii.gz")
    nii_denoiser = os.path.join(output_dir, f"{name}_denoiser.nii.gz")
    nii_srr      = os.path.join(output_dir, f"{name}_srr.nii.gz")

    nib.save(nib.Nifti1Image(im_np,    affine), nii_original)
    nib.save(nib.Nifti1Image(pred1_np, affine), nii_denoiser)
    nib.save(nib.Nifti1Image(pred2_np, affine), nii_srr)

    # Optional: expose per-version high-res "all-slices" PNG paths (placeholders)
    png_high_paths = {}
    if expose_individual_png_paths:
        png_high_paths = {
            "original": os.path.join(output_dir, f"{name}_original_all_slices_highres.png"),
            "denoiser": os.path.join(output_dir, f"{name}_denoiser_all_slices_highres.png"),
            "srr":      os.path.join(output_dir, f"{name}_srr_all_slices_highres.png"),
        }

    # print(f"✅ Saved standard figure: {save_path_std}")
    print(f"🖼️ Saved high-resolution figure: {save_path_high}")
    print(f"🧠 Saved NIfTI volumes: {nii_original}, {nii_denoiser}, {nii_srr}")

    # View orthogonal slices from each saved NIfTI (loaded from disk)
    if view_orthoslices:
        try:
            # Load back from disk and display orthoslicers
            for label, path in [("Original", nii_original),
                                ("Denoiser", nii_denoiser),
                                ("SRR", nii_srr)]:
                img = nib.load(path)
                data = img.get_fdata()
                slicer = OrthoSlicer3D(data)
                # Try to add a title (not guaranteed depending on backend)
                try:
                    slicer.fig.suptitle(f"OrthoSlicer: {label} ({name})")
                except Exception:
                    pass
                slicer.show()
        except Exception as e:
            print(f"⚠️ OrthoSlicer display failed: {e}")

    return {
        "fig_paths": {"std": save_path_std, "high": save_path_high},
        "nii_paths": {"original": nii_original, "denoiser": nii_denoiser, "srr": nii_srr},
        "png_high_paths": png_high_paths
    }

def unsharp_mask(volume, sigma=1.0, amount=1.0):
    """
    Apply unsharp masking to a 3D volume.
    Args:
        volume (np.ndarray): 3D input volume.
        sigma (float): Gaussian blur sigma.
        amount (float): Strength of sharpening.
    Returns:
        np.ndarray: Sharpened volume.
    """
    blurred = gaussian_filter(volume, sigma=sigma)
    mask = volume - blurred
    sharpened = volume + amount * mask
    return np.clip(sharpened, 0, 1)  # assuming normalized input

# ------------------------------------------------------------
# Load and resample HF volume
# ------------------------------------------------------------

trial1 = False
trail2 = True
unsharp_mask_ = False

if trial1:
    model_name = 'residual_srr_unet_l1_l2_ssim_mse_ssim_edge'
    folder_path = "niv_results/outputs_src_simulated/Output_patch_noise"
else:
    model_name = 'residual_srr_unet_l1_l2_ssim_l2_ssim_edge'
    folder_path = "niv_results/outputs_src_simulated/Output_patch_noise"
    print("Using model:", model_name)
    print("Using folder:", folder_path)

if unsharp_mask_ and trial1:   
    output_dir='src_simulated/outputs/outputs_355281/trail11_unsharp_masked'
elif unsharp_mask_ and not trial1:
    output_dir='src_simulated/outputs/outputs_355281/trail22_unsharp_masked'
elif not unsharp_mask_ and trial1:
    output_dir='src_simulated/outputs/outputs_355281/trail11_pred1_only'
else:
    output_dir='src_simulated/outputs/outputs_355281/trail22_pred1_only'

# ------------------------------------------------------------
# Load and resample LF volume
# ------------------------------------------------------------
class DomainAGenerator:
    def __init__(self, path, batch_size=1, target_h=128, target_w=128, target_d=35, 
                 target_spacing=(1,1,2), field_strength=0.05, rotate=False, visit=1, shuffle=True):
        """
        Generator for Domain A (LF MRI)
        Returns 3D volume + context per volume
        """
        self.path = path
        self.files = [os.path.join(path,f) for f in os.listdir(path) if f.endswith((".nii",".nii.gz"))]
        self.batch_size = batch_size
        self.target_h = target_h
        self.target_w = target_w
        self.target_d = target_d
        self.target_spacing = target_spacing
        self.field_strength = field_strength
        self.rotate = rotate
        self.visit = visit
        self.apply_brain_extraction = False
        #no morphology
        self.shuffle = shuffle

        # Context scaler for volume-wise context
        self.scaler = StandardScaler()
        dummy_ctx = self._create_context(len(self.files))
        self.scaler.fit(dummy_ctx)

        self.on_epoch_end()
    
    def __len__(self):
        return len(self.files) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.files)

    def _parse_value(self, val, default):
        """
        Convert string values like '20e3', '50.0', '2.068d' to float safely.
        """
        try:
            if isinstance(val, str):
                val = val.replace('d', '')  # remove trailing 'd'
                val = val.replace('"', '')  # remove quotes
                return float(eval(val))     # handles '20e3'
            return float(val)
        except:
            return float(default)


    def _create_context(self, base_params_path, fpath=None, default_context=True, N=1):
        """
        Create context vector (N, 5) for each volume.
        N: number of slices or samples per volume
        """

        if default_context:
            TE = np.random.uniform(80, 120)
            TR = np.random.uniform(2000, 3000)
            bandwidth = np.random.uniform(150, 250)
            rxGain = np.random.uniform(20, 40)
            etLength = np.random.uniform(8, 16)
            dwellTime = np.random.uniform(5, 10)
            SNR = (self.field_strength * 10) + np.random.normal(0, 2)

        else:
            if fpath is None:
                raise ValueError("fpath must be provided when default_context=False")

            # -----------------------------
            # Convert .nii/.nii.gz → .json
            # -----------------------------
            if fpath.endswith('.nii.gz'):
                json_name = os.path.basename(fpath[:-7] + '.json')
            elif fpath.endswith('.nii'):
                json_name = os.path.basename(fpath[:-4] + '.json')
            else:
                raise ValueError("Unsupported file extension")

            json_path = os.path.join(base_params_path, json_name)

            if not os.path.exists(json_path):
                print(f"[WARNING] JSON not found: {json_path}, using defaults")
                return self._create_context(base_params_path, fpath, default_context=True, N=N)

            # -----------------------------
            # Load JSON
            # -----------------------------
            with open(json_path, 'r') as jf:
                params = json.load(jf)

            params = params.get("ImageScanParameters", {})
            # print(f"\n[INFO] Loaded params from {json_path}")

            # Extract parameters
            TE = self._parse_value(params.get('echoTime', 100), 100)
            TR = self._parse_value(params.get('repTime', 2500), 2500)
            bandwidth = self._parse_value(params.get('bandwidth', 200), 200)
            etLength = self._parse_value(params.get('etLength', 12), 12)

            # Visit from filename
            visit_match = re.search(r'V(\d+)', fpath)
            if visit_match:
                self.visit = int(visit_match.group(1))

        # ---------------------
        # Create context vector repeated N times
        # ---------------------
        single_row = np.array([TE, TR, bandwidth, etLength, self.field_strength], dtype=np.float32)
        context = np.tile(single_row, (N, 1))  # shape: (N, 5)

        return context
    
    def _extract_brain_volume(self, vol):
        """
        Perform slice-wise brain extraction using Otsu + morphology.

        Additionally supports building a *combined* mask across slices (union mask),
        and applying the same mask to the entire volume to reduce slice-to-slice flicker.

        Args:
            vol: (H, W, D) volume (your docstring says normalized [-1,1], but this works either way)

        Returns:
            brain_vol: (H, W, D) uint8-like range as produced by OpenCV masking (0..255)
        """
        import cv2

        print("[INFO] Performing slice-wise brain extraction...")

        num_slices = vol.shape[2]
        H, W = vol.shape[0], vol.shape[1]

        # --- settings (safe defaults if attributes not present) ---
        combine_masks = bool(getattr(self, "combine_slice_masks", True))
        min_slices = int(getattr(self, "mask_min_slices", 1))  # require presence in >=K slices
        kernel_size = int(getattr(self, "mask_kernel_size", 16))
        dilate_iters = int(getattr(self, "mask_dilate_iters", 1))
        close_iters = int(getattr(self, "mask_close_iters", 1))

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        norm_slices = []
        masks = []

        # 1) Build per-slice normalized images + masks
        for i in range(num_slices):
            slice_data = vol[:, :, i]

            if slice_data is None or slice_data.size == 0:
                norm_slices.append(np.zeros((H, W), dtype=np.uint8))
                masks.append(np.zeros((H, W), dtype=np.uint8))
                continue

            try:
                norm = cv2.normalize(
                    np.abs(slice_data),
                    None, 0, 255,
                    cv2.NORM_MINMAX,
                    cv2.CV_8U
                )
                norm_slices.append(norm)

                # --- Otsu threshold ---
                _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # --- Find contours ---
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                mask = np.zeros_like(norm, dtype=np.uint8)
                if contours:
                    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
                        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

                    # --- Morphological refinement ---
                    mask = cv2.dilate(mask, kernel, iterations=dilate_iters)
                    for _ in range(max(1, close_iters)):
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                masks.append(mask)

            except Exception as e:
                print(f"[WARN] Slice {i} failed: {e}")
                norm_slices.append(np.zeros((H, W), dtype=np.uint8))
                masks.append(np.zeros((H, W), dtype=np.uint8))

        norm_vol = np.stack(norm_slices, axis=2)   # (H,W,D) uint8
        mask_vol = np.stack(masks, axis=2)         # (H,W,D) uint8 0/255

        # 2) Optionally build combined mask over depth
        if combine_masks:
            # presence count across slices
            present = (mask_vol > 0).astype(np.uint8)    # (H,W,D) 0/1
            count = np.sum(present, axis=2)              # (H,W)

            k = max(1, min(int(min_slices), num_slices))
            combined_2d = (count >= k).astype(np.uint8) * 255  # (H,W) 0/255

            # optional cleanup to smooth combined mask boundary
            combined_2d = cv2.morphologyEx(combined_2d, cv2.MORPH_CLOSE, kernel)

            mask_to_apply = np.repeat(combined_2d[:, :, None], num_slices, axis=2)
        else:
            mask_to_apply = mask_vol

        # 3) Apply mask to all slices
        brain_vol = np.where(mask_to_apply > 0, norm_vol, 0).astype(np.uint8)

        return brain_vol

    def _load_volume(self, fpath):
        nii = nib.load(fpath)
        vol = nii.get_fdata().astype(np.float32)  # Fix negatives
        # current_spacing = nii.header.get_zooms()[:3]
        # visualize_slices(vol)
        current_spacing = (1.0,1.0,2.0)
        print(f"[INFO] Loaded {os.path.basename(fpath)} with shape {vol.shape}")
        # print(f"[INFO] Loading spacing {current_spacing}")

        # Resample to target spacing
        zoom_factors = (
            current_spacing[0] / self.target_spacing[0],
            current_spacing[1] / self.target_spacing[1],
            current_spacing[2] / self.target_spacing[2]
        )
        vol = zoom(vol, zoom_factors, order=1)
        vol = np.ascontiguousarray(vol)

        # Accept only target in-plane resolution
        h, w, d = vol.shape
        if h != self.target_h or w != self.target_w:
            print(f"[SKIP] {os.path.basename(fpath)} wrong in-plane size {vol.shape}")
            return None

        # Crop or pad depth
        if d != self.target_d:
            ds = max((self.target_d - d) // 2, 0)
            de = ds + min(d, self.target_d)
            d0 = max((d - self.target_d) // 2, 0)
            out = np.zeros((h, w, self.target_d), dtype=vol.dtype)
            out[:, :, ds:de] = vol[:, :, d0:d0 + (de - ds)]
            vol = out

        # ===============================
        # ✅ INSERT BRAIN EXTRACTION HERE
        # ===============================
        if self.apply_brain_extraction:
            vol = self._extract_brain_volume(vol)

        # Normalize [-1,1]
        # vol = (vol / np.max(vol) - 0.5) * 2 if np.max(vol) > 0 else vol
        # visualize_slices(vol)

        return vol  # Do not add channel

    def __iter__(self):
        for fpath in self.files:
            vol = self._load_volume(fpath)
            if vol is None:
                continue
            
            # print(f"[LOAD] {os.path.basename(fpath)}: final shape {vol.shape}")
            # Apply rotation if needed
            if self.rotate:
                # Rotate 90 degrees k times along the in-plane axes (0,1)
                # You can change k to random 0-3 for random rotation
                vol = np.rot90(vol, k=1, axes=(0, 1))
            base_params_path = 'niv_raw_data/Nipah_IRF_data/LFMRI_DATA_IRF_ALL_PARAMS'
            ctx = self._create_context(base_params_path, fpath=fpath, default_context=False)
            ctx = self.scaler.transform(ctx)
            save_file = os.path.basename(fpath)
            print(f"[GENERATOR] Yielding {save_file}")
            yield vol, ctx[0], save_file

# Example paths (use your config_lf paths)
path_A = config_lf.path_lf_t1w  # LF T1w data directory
genA = DomainAGenerator(path_A)

# Fetch one batch from Domain A
for volA, ctxA, save_file in genA:
    print("Domain A volume shape:", volA.shape)
    print("Domain A context:", ctxA.shape)
    print("Domain A context values:", ctxA)
    print("Domain A save file:", save_file)
    break

# Path for the second stage model (SRR or enhancement)
model_name = 'residual_srr_unet_l2_edge_gram_matrix_loss_l2_ssim_edge'

# folder_path = "niv_results/outputs_src_simulated/Output_patch_noise"
folder_path = "niv_results/outputs_src_simulated/enhancement"

# Path for saving the generated volumes from domain adaptation step
output_dir_lf ='niv_results/Evaluator_data/VolA'
output_dir_denoise ='niv_results/Evaluator_data/CycleGAN'
output_dir_enhance ='niv_results/Evaluator_data/Enhancement'
output_dir = output_dir_denoise

for volA, ctxA, save_file in genA:

    print("Domain A volume shape:", volA.shape)
    print("Domain A context:", ctxA.shape)
    print("Domain A context values:", ctxA)
    # im = im.astype(np.float32)
    volA = np.expand_dims(volA, axis=0)  # (1, H, W, D)
    print("Final LF input shape:", volA.shape)

    if volA.max() > 1:
        volA = (volA / np.max(volA) - 0.5) * 2

    # print min and max of volA after normalization
    print(f"Generated volume range after normalization: min={volA.min()}, max={volA.max()}")

    # get domain adopted and perform further steps
    results, pred1, pred2, model1, model2 = evaluate_model(
        folder_path=folder_path,
        model_name=model_name,
        X_test=volA,
        y_test=volA,
        patch_size=(64, 64, 32),
        overlap=0.5,
        visualize_slices=[15]
    )

    print("Evaluation Results:after Stage 2 Refinement")

    volA = np.squeeze(volA, axis=0)

    visualize_comparison(volA, pred1, pred2, name='trial_comparison', output_dir=output_dir)