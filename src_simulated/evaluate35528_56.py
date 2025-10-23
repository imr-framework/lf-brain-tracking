# compute AES
# make 2 figures
# 2 subjects
# How to go down

# ------------------------------------------------
# Example Usage
# ------------------------------------------------
import sys
sys.path.insert(0, './')
sys.path.insert(0, './data_read_code')
from src_simulated.evaluate_niv_lf_test import evaluate_model, predict_volume
import argparse
import sys
import logging
from datetime import datetime
import time
import pydicom
from glob import glob
import LFsim.keaDataProcessing as keaProc
from LFsim.utils_sim import *
import os
from read_kea3d import kea3d

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from src_niv.metrics import psnr, ssim, mse, composite_loss
from src_niv.utils import visualize_pair
from scipy.ndimage import zoom

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

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, rotate

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
import numpy as np
from scipy.ndimage import rotate

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

import numpy as np

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

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
from skimage.morphology import ball

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


import numpy as np
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion, label
from skimage.morphology import ball, remove_small_objects

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

# ------------------------------------------------------------
# Load and resample LF volume
# ------------------------------------------------------------
model_name = 'residual_srr_unet_l1_l2_ssim_mse_ssim_edge'
folder_path = "Output_patch"
data_folder = "Data/data_sim_check/3T_1simulated_LF/train_test"
subjects = ["26184"]
test_days = [3]

lf_path = 'Data/data_sim_check/35528D56_3'
im = load_lf_3d_file(lf_path)
print(f"Loaded LF volume: {im.shape}, range=({im.min():.4f}, {im.max():.4f})")
# visualize_volume(im, title="original LF")

im = resample_volume_numpy(im,
                            current_spacing=(2.0, 2.0, 5.33333333),
                            new_spacing=(1.0, 1.0, 2.0),
                            order=3)

print(f"Resampled LF volume: {im.shape}, range=({im.min():.4f}, {im.max():.4f})")
# visualize_volume(im, title="Resampled LF")

im = rot90_3d(im, k=1, axes=(0, 1))
print(f"Rotated LF volume: {im.shape}, range=({im.min():.4f}, {im.max():.4f})")
# ============================================================
# 🔹 5. Example Workflow
# ============================================================
# --- C. Up 20%, Left 10%, Rotate 15° CCW ---

im = circshift_3d(im, up_fraction=0.20, left_fraction=0.0, rotation_deg=0)
visualize_volume(im, title="Shifted + Rotated")
print("Original shape:", im.shape)


# # Morphological analysis: threshold, fill holes, dilation, erosion, masking
# cleaned_im, mask = extract_head_mask(im_shift_rot, threshold=0.091, dilation_iter=1, erosion_iter=1)
# print(f"After morphological cleanup: shape={cleaned_im.shape}, mask sum={mask.sum()}")

# visualize_volume(cleaned_im, title="Morphologically Cleaned")

im = normalize_volume(im, method='minmax')
# Rotate 90° counterclockwise (default)
visualize_volume(im, title="Normalized LF")


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

results, pred1, pred2, model1, model2 = evaluate_model(
    folder_path=folder_path,
    model_name=model_name,
    X_test=im,
    y_test=im,
    patch_size=(64, 64, 32),
    overlap=0.5,
    visualize_slices=[15]
)

visualize_volume(pred1, title="Shifted + Rotated")
visualize_volume(pred2, title="Shifted + Rotated")

print("Evaluation Results:after Stage 2 Refinement")

im = np.squeeze(im, axis=0)

# Visualize slices 26 to 32 in large format
start_slice, end_slice = 23, 30
fig, axes = plt.subplots(1, end_slice - start_slice + 1, figsize=(3 * (end_slice - start_slice + 1), 6))
for i, slice_idx in enumerate(range(start_slice, end_slice + 1)):
    axes[i].imshow(im[:, :, slice_idx], cmap='gray')
    axes[i].set_title(f'Slice {slice_idx}', fontsize=14)
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# # Apply morphological operation to im
# cleaned_im, mask = extract_head_mask(im, threshold=0.1, dilation_iter=1, erosion_iter=1)
# print(f"After morphological cleanup: shape={cleaned_im.shape}, mask sum={mask.sum()}")

# visualize_volume(cleaned_im, title="Morphologically Cleaned")

# im = cleaned_im
for i in range(6):
    # ---- Stage 2 Refinement ----
    pred2 = predict_volume(model2, im, patch_size=(64,64,32), overlap=0.5)

    # Visualize slices 26 to 32 in large format
    start_slice, end_slice = 23, 30
    fig, axes = plt.subplots(1, end_slice - start_slice + 1, figsize=(3 * (end_slice - start_slice + 1), 6))
    for i, slice_idx in enumerate(range(start_slice, end_slice + 1)):
        axes[i].imshow(pred2[:, :, slice_idx], cmap='gray')
        axes[i].set_title(f'Slice {slice_idx}', fontsize=14)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    im = pred2
    aes = compute_aes(im)
    print(f"AES after iteration: {aes:.4f}")

# # Apply morphological operation to im
# cleaned_im, mask = extract_head_mask(im, threshold=0.1, dilation_iter=1, erosion_iter=1)
# print(f"After morphological cleanup: shape={cleaned_im.shape}, mask sum={mask.sum()}")

# visualize_volume(cleaned_im, title="Morphologically Cleaned")
# im = cleaned_im

# # ---- Stage 2 Refinement ----
# pred2 = predict_volume(model2, im, patch_size=(64,64,32), overlap=0.5)
# # visualize_volume(pred2, title="Shifted + Rotated")

# # Visualize slices 26 to 32 in large format
# start_slice, end_slice = 23, 30
# fig, axes = plt.subplots(1, end_slice - start_slice + 1, figsize=(3 * (end_slice - start_slice + 1), 6))
# for i, slice_idx in enumerate(range(start_slice, end_slice + 1)):
#     axes[i].imshow(pred2[:, :, slice_idx], cmap='gray')
#     axes[i].set_title(f'Slice {slice_idx}', fontsize=14)
#     axes[i].axis('off')
# plt.tight_layout()
# plt.show()

# im = pred2
# aes = compute_aes(im)
# print(f"AES after iteration: {aes:.4f}")