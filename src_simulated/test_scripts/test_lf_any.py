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

def visualize_all_pngs(
    im,
    pred1,
    pred2,
    name='comparison',
    output_dir='outputs_59228/trial1',
    expose_individual_png_paths=True,
    view_orthoslices=True
):
    """
    Display all slices from original (im), pred1, pred2 in a single figure:
    rows = versions (im, pred1, pred2), columns = all slices.
    Saves a high-resolution figure and optionally opens orthogonal slicer windows.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Convert inputs to numpy
    def _to_numpy(x):
        return np.array(x) if not isinstance(x, np.ndarray) else x

    im_np    = _to_numpy(im)
    pred1_np = _to_numpy(pred1)
    pred2_np = _to_numpy(pred2)

    # Validate volumes
    for vol_name, vol in [('im', im_np), ('pred1', pred1_np), ('pred2', pred2_np)]:
        if vol is None:
            raise ValueError(f"{vol_name} is None.")
        if vol.ndim < 3:
            raise ValueError(f"{vol_name} must be at least 3D. Got shape {vol.shape}.")

    # All slices
    num_slices = im_np.shape[-1]
    slice_indices = range(num_slices)

    titles = ['Original (im)', 'Prediction 1 (denoiser)', 'Prediction 2 (srr)']
    volumes = [im_np, pred1_np, pred2_np]

    # Create figure (3 rows × num_slices columns)
    fig, axes = plt.subplots(3, num_slices, figsize=(2*num_slices, 6))
    if num_slices == 1:
        axes = np.atleast_2d(axes)

    # Plot each version per slice
    for col, idx in enumerate(slice_indices):
        for row, (vol, label) in enumerate(zip(volumes, titles)):
            ax = axes[row, col]
            ax.imshow(vol[:, :, idx], cmap='gray')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(label, fontsize=12)
            if row == 0:
                ax.set_title(f"Slice {idx}", fontsize=10)

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    save_path = os.path.join(output_dir, f"{name}_all_slices.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=600)
    plt.show()
    print(f"🖼️ Saved high-resolution figure: {save_path}")

    # Display each volume in OrthoSlicer3D
    if view_orthoslices:
        for vol, label in zip(volumes, titles):
            print(f"🔍 Viewing {label} in OrthoSlicer3D")
            slicer = OrthoSlicer3D(vol)
            slicer.show()

    return {
        "fig_path": save_path,
        "volumes": titles
    }



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
unsharp_mask_ = True

if trial1:
    model_name = 'residual_srr_unet_l1_l2_ssim_mse_ssim_edge'
    folder_path = "src_simulated/outputs/Output_patch"
else:
    model_name = 'residual_srr_unet_l1_l2_ssim_l2_ssim_edge'
    folder_path = "src_simulated/outputs/Output_patch_noise"
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
# better ones: 26184 (3), 30366 (2), 35528 (d56), 59228(d42)

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

visualize = True
# -----------------------------
# File dialog
# -----------------------------
root = tk.Tk()
root.withdraw()  # hide the main window

initial_dir = "Data/Nipah_IRF_data/LFMRI_DATA_IRF_NIFTI_best_Corrected"
filename = filedialog.askopenfilename(
    title="Select a NIfTI file",
    filetypes=[("NIfTI files", "*.nii *.nii.gz")],
    initialdir=initial_dir
)

if not filename:
    raise ValueError("No file selected!")

print("Selected file:", filename)

# -----------------------------
# Extract folder name from selected file path
# -----------------------------
# Example: filename = "Data/Nipah_IRF_data/LFMRI_DATA_IRF_best/35528/IRF_071E_2_C1_20240710&35528_D_minus27&V01&3DTSE&4.nii.gz"
folder_name = Path(filename).parent.name  # gets '35528'
print("Folder name extracted:", folder_name)
# -----------------------------

# -----------------------------
# Set output directory and output filename
# -----------------------------
out_dir = f"src_simulated/outputs_lfmri/{folder_name}"
os.makedirs(out_dir, exist_ok=True)

out_name = Path(filename).name  # just the file name
# Remove .nii.gz or .nii extension for PNG output name
stem = Path(filename).stem
if stem.endswith('.nii'):
    stem = stem[:-4]
out_name_png = stem + ".png"
out_path = os.path.join(out_dir, out_name)

print("Output directory:", out_dir)
print("Output file path:", out_path)


os.makedirs(out_dir, exist_ok=True)
out_name = Path(filename).stem + ".png"
out_png = os.path.join(out_dir, out_name)

# -----------------------------
# Load LF NIfTI volume
# -----------------------------

nifti = nib.load(filename)
im_orig = nifti.get_fdata()
ny, nx, nz = im_orig.shape

print(f"Loaded LF volume: {im_orig.shape}, range=({im_orig.min():.4f}, {im_orig.max():.4f})")
print(f"Voxel size: {nifti.header.get_zooms()} mm")
# Visualize original LF volume
if visualize:
    visualize_volume(im_orig, title="Original LF Volume")

# -----------------------------
# Load and preprocess LF volume
# -----------------------------
im = im_orig.copy()

current_spacing = nifti.header.get_zooms() # Example current spacing in mm (z, y, x)
# Resample to 1x1x2 mm³ voxel size

im = resample_volume_numpy(im,
                            current_spacing=current_spacing,
                            new_spacing=(1.0, 1.0, 2.0),
                            order=3)

print(f"Resampled LF volume: {im.shape}, range=({im.min():.4f}, {im.max():.4f})")

if visualize:
    visualize_volume(im, title="Resampled LF")

im = rot90_3d(im, k=1, axes=(0, 1))
print(f"Rotated LF volume: {im.shape}, range=({im.min():.4f}, {im.max():.4f})")

if visualize:
    visualize_volume(im, title="Rotated LF")

subject = '35528_1'
slice_indices = [24,25,26,27,28]  # Slices to display

im = normalize_volume(im, method='minmax')

if visualize:
    visualize_volume(im, title="Normalized LF")

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

visualize_all_pngs(im, pred1, pred2, name=stem, output_dir=out_dir)

# visualize_comparison(im, pred1, pred2, name='trial_comparison', output_dir=output_dir)

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