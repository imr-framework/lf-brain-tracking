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
import logging
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
from scipy.ndimage import (
    zoom,
    gaussian_filter,
    center_of_mass,
    rotate,
    binary_dilation,
    binary_erosion,
    binary_fill_holes
)

from skimage.morphology import ball
import cv2
import pydicom
import tensorflow as tf
from tensorflow.keras.models import load_model

# ------------------------------------------------------------
# Project-specific imports
# ------------------------------------------------------------
sys.path.insert(0, './')
sys.path.insert(0, './data_read_code')

from read_kea3d import kea3d
import LFsim.keaDataProcessing as keaProc
from LFsim.utils_sim import *
from src_simulated.evaluate_niv_lf_test import evaluate_model, predict_volume
from src_niv.metrics import psnr, ssim, mse, composite_loss
from src_niv.utils import visualize_pair

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

lf_path = 'niv_raw_data/Nipah_IRF_data/data_niv/data_sim_check/datad4/35528D56_3'
im = load_lf_3d_file(lf_path)

print(f"Loaded LF volume: {im.shape}, range=({im.min():.4f}, {im.max():.4f})")

visualize_volume(im, title="original LF")

im = resample_volume_numpy(im,
                            current_spacing=(2.0, 2.0, 5.33333333),
                            new_spacing=(1.0, 1.0, 2.0),
                            order=3)

print(f"Resampled LF volume: {im.shape}, range=({im.min():.4f}, {im.max():.4f})")
visualize_volume(im, title="Resampled LF")

im = rot90_3d(im, k=1, axes=(0, 1))
print(f"Rotated LF volume: {im.shape}, range=({im.min():.4f}, {im.max():.4f})")

# ============================================================
# 🔹 5. Example Workflow
# ============================================================
# --- C. Up 20%, Left 10%, Rotate 15° CCW ---

im = circshift_3d(im, up_fraction=0.20, left_fraction=0.0, rotation_deg=0)
visualize_volume(im, title="Shifted + Rotated")
print("Original shape:", im.shape)
subject = '35528_1'
slice_indices = [24,25,26,27,28]  # Slices to display

# # Morphological analysis: threshold, fill holes, dilation, erosion, masking
# cleaned_im, mask = extract_head_mask(im_shift_rot, threshold=0.091, dilation_iter=1, erosion_iter=1)
# print(f"After morphological cleanup: shape={cleaned_im.shape}, mask sum={mask.sum()}")

# visualize_volume(cleaned_im, title="Morphologically Cleaned")

im = normalize_volume(im, method='minmax')

# visualize_volume(im, title="Normalized LF")
# im_fixed, pad_info = pad_or_crop_volume_to_shape(im_shift_rot, target_shape=(144, 144, 40))
# print("Processed shape:", im_fixed.shape)
# print("Pad/Crop info:", pad_info)
# visualize_volume(im_fixed, title="paded")

# ------------------------------------------------------------
# Add batch axis and evaluate
# ------------------------------------------------------------

im = np.expand_dims(im, axis=0)  # (1, H, W, D)
print("Final LF input shape:", im.shape)

im = im.astype(np.float32)

# apply total variation filtering to reduce noise
import numpy as np
from skimage.restoration import denoise_tv_chambolle

def tv_denoise_lf_mri(volume, weight=0.03, n_iter=200, normalize=True):
    """
    3D Total Variation denoising for very low-field MRI.

    Parameters
    ----------
    volume : np.ndarray
        Input LF-MRI volume (D, H, W)
    weight : float
        TV regularization strength (default 0.03 for LF-MRI)
    n_iter : int
        Number of iterations
    normalize : bool
        Normalize volume to [0, 1] before TV filtering

    Returns
    -------
    np.ndarray
        TV-denoised volume (same shape as input)
    """
    vol = volume.astype(np.float32)

    # Normalize intensity (important for TV stability)
    if normalize:
        vmin, vmax = vol.min(), vol.max()
        vol = (vol - vmin) / (vmax - vmin + 1e-8)

    # 3D TV denoising
    tv_vol = denoise_tv_chambolle(
        vol,
        weight=weight,
        max_num_iter=n_iter,
        channel_axis=None
    )

    # Restore original intensity range
    if normalize:
        tv_vol = tv_vol * (vmax - vmin) + vmin

    return tv_vol

# apply gaussian smoothing to mimic LF blurring
# im = np.array([tv_denoise_lf_mri(im[i], weight=0.03, n_iter=200) for i in range(len(im))])

import numpy as np
from scipy.ndimage import convolve

def lowpass_2x2x2(volume):
    
    """
    Apply a 2x2x2 low-pass (averaging) filter to a 3D volume.

    Parameters
    ----------
    volume : np.ndarray
        Input volume (D, H, W)

    Returns
    -------
    np.ndarray
        Smoothed volume
    """

    kernel = np.ones((2, 2, 2), dtype=np.float32) / 8.0
    volume = volume.astype(np.float32)

    filtered = convolve(volume, kernel, mode='reflect')

    return filtered

# Applying low-pass filtering
print("Applying post-processing low-pass filtering...")
# apply total variation filtering on pred1 to reduce noise
# pred2 = lowpass_2x2x2(im)
# im = np.array([gaussian_filter(im[i], sigma=1) for i in range(len(im))])
print("✅ Test data pre-processed.")

results, pred1, pred2, model1, model2 = evaluate_model(
    folder_path=folder_path,
    model_name=model_name,
    X_test=im,
    y_test=im,
    patch_size=(64, 64, 32),
    overlap=0.5,
    visualize_slices=[15]
    )

print("Evaluation Results:after Stage 2 Refinement")

im = np.squeeze(im, axis=0)

if unsharp_mask_:
    pred1 = unsharp_mask(pred1, sigma=1.0, amount=1.0)
    pred2 = unsharp_mask(pred2, sigma=1.0, amount=1.0)

visualize_comparison(im, pred1, pred2, name='trial_comparison', output_dir=output_dir)

# ------------------------------------------------------------
# 3D Total Variation Denoising Function
# ------------------------------------------------------------

# print("Applying post-processing denoising...")
# # apply total variation filtering on pred1 to reduce noise
# pred1 = tv_denoise_lf_mri(pred1, weight=0.03, n_iter=200)
# visualize_slice(pred1, 'SRR_Output', output_dir=output_dir)

# Applying low-pass filtering
print("Applying post-processing low-pass filtering...")
# apply total variation filtering on pred1 to reduce noise
pred2 = lowpass_2x2x2(pred2)
visualize_slice(pred2, 'SRR_Output', output_dir=output_dir)

visualize_slice(im, 'Original', output_dir=output_dir)

# Working on the High-field Data

# data_folder = "Data/data_sim_check/35528simulated_LF/train_test"
# subjects = ["35528"]
# test_days = [5]

# def load_subject_day_data(subject, day):
#     path = os.path.join(data_folder, f"{subject}_day{day}_train_data.npy")
#     if not os.path.exists(path):
#         return None
#     data = np.load(path, allow_pickle=True).item()
#     return data["x_train"].astype(np.float32), data["y_train"].astype(np.float32)

# X_test, y_test = [], []
# for subj in subjects:
#     for day in test_days:
#         d = load_subject_day_data(subj, day)
#         if d is not None:
#             X_test.append(d[0])
#             y_test.append(d[1])

# X_test, y_test = np.array(X_test), np.array(y_test)
# print(f"📦 Loaded {len(X_test)} test volumes.")
# X_test, y_test = normalize_dataset(X_test, y_test, method='minmax')

# #gaussian smoothing to reduce noise X_test
# from scipy.ndimage import gaussian_filter
# X_test = np.array([gaussian_filter(vol, sigma=1) for vol in X_test])
# # X_test = np.array([gaussian_filter(vol, sigma=1) for vol in X_test])
# print("✅ Test data normalized.")

# # ------------------------------------------------------------
# # 🔹 Evaluate each model sequentially
# # ------------------------------------------------------------

# results, pred1, pred2, model1, model2 = evaluate_model(
#     folder_path=folder_path,
#     model_name=model_name,
#     X_test=X_test[1:2],       # evaluate on subset for speed
#     y_test=y_test[1:2],
#     patch_size=(64, 64, 32),
#     overlap=0.5,
#     visualize_slices=[15]
# )

# visualize_comparison(X_test[1:2], pred1, pred2, name='trial_HF_comparison', output_dir=output_dir)