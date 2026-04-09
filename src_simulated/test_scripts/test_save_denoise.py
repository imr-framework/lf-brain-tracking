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
"""
cycleGAN model
Based on the code by Jason Brownlee from his blogs on https://machinelearningmastery.com/
I am adapting his code to various applications but original credit goes to Jason.
The model uses instance normalization layer:
Normalize the activations of the previous layer at each step,
i.e. applies a transformation that maintains the mean activation
close to 0 and the activation standard deviation close to 1.
Standardizes values on each output feature map rather than across features in a batch.
Download instance normalization code from here: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py
Or install keras_contrib using guidelines here: https://github.com/keras-team/keras-contrib
"""

import sys
sys.path.insert(0, './')

import os
import random
from random import random as rand_func
from numpy import asarray, load, zeros, ones
from numpy.random import randint
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
import json

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, LeakyReLU,
    Activation, Concatenate, Add, UpSampling2D, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import re
# Image loading and preprocessing
from keras.preprocessing.image import img_to_array, load_img

# Nifti / medical image processing
import nibabel as nib
from scipy.ndimage import zoom

# Load config = CycleGANConfig() class from config.py
from src_simulated.cyclegan_models.config_lf import config_lf
# import losses from losses.py
from src_simulated.cyclegan_models.losses_cyclegan import *

# models.py load all models
from src_simulated.cyclegan_models.models import *
from src_simulated.cyclegan_models.train_utils_params_parallel import *

EPOCHS = config_lf.EPOCHS

# test mode
TEST = config_lf.TEST
SLICES_TEST = config_lf.SLICES_TEST

#parameters for descriminator
DISC_LOSS = config_lf.DISC_LOSS
DISC_LEARNING_RATE = config_lf.DISC_LEARNING_RATE
DISC_BETA_1 = config_lf.DISC_BETA_1
DISC_LOSS_WEIGHTS = config_lf.DISC_LOSS_WEIGHTS

#parameters for generator
GEN_LOSS_1 = config_lf.GEN_LOSS_1
GEN_LOSS_2 = config_lf.GEN_LOSS_2
GEN_LOSS_3 = config_lf.GEN_LOSS_3
GEN_LOSS_4 = config_lf.GEN_LOSS_4
GEN_LEARNING_RATE = config_lf.GEN_LEARNING_RATE
GEN_BETA_1 = config_lf.GEN_BETA_1
GEN_LOSS_WEIGHTS = config_lf.GEN_LOSS_WEIGHTS

#train parameters
INITIAL_LR = config_lf.INITIAL_LR
N_ITER = config_lf.N_ITER
N_ITER_DECAY = config_lf.N_ITER_DECAY

# Output directories
OUTPUT_DIR = config_lf.OUTPUT_DIR
# Visualization parameters
VISUALIZE = config_lf.VISUALIZE  # Whether to visualize test examples during training
num_slices = config_lf.N_SLICES

# Read dataset

data_folder = "Data/data_sim_check/35528simulated_LF/train_test"
subjects = ["26184", "30366", "35528","34507", "35547", "59228", "59877","59233"]
train_day = [1,2,3,4,5]

def crop_or_pad_depth(vol, target_D):
    """
    Only crop/pad depth dimension.
    """
    h, w, d = vol.shape
    out = np.zeros((h, w, target_D), dtype=vol.dtype)

    ds = max((target_D - d) // 2, 0)
    de = ds + min(d, target_D)

    d0 = max((d - target_D) // 2, 0)

    out[:, :, ds:de] = vol[:, :, d0:d0 + (de - ds)]
    return out

def normalize_volume(vol):
    """
    Normalize an MRI volume to [-1, 1] for CycleGAN.
    Args:
        vol: numpy array, shape (H,W,D,C) or (H,W,D) or (N,H,W,D,C)
    Returns:
        normalized volume in [-1,1]
    """
    vol = np.abs(vol)           # remove negative artifacts
    max_val = np.max(vol)
    if max_val > 0:
        vol = vol / max_val     # scale to [0,1]
    vol = (vol - 0.5) * 2       # scale to [-1,1]
    return vol

def crop_or_pad_depth(vol, target_d=35):
    """Crop or pad depth to target size."""
    h, w, d = vol.shape
    if d > target_d:
        start = (d - target_d) // 2
        vol = vol[:, :, start:start+target_d]
    elif d < target_d:
        pad_before = (target_d - d) // 2
        pad_after = target_d - d - pad_before
        vol = np.pad(vol, ((0,0),(0,0),(pad_before, pad_after)), mode='constant')
    return vol

def load_nii_volumes(path, target_spacing=(1,1,2),TARGET_H = 128, TARGET_W = 128, TARGET_D = 35, add_channel=False):

    """
    Load NIfTI volumes, resample by voxel spacing only,
    ensure H=W=128 or discard, crop/pad D→35, normalize to [-1,1].
    """

    volumes = []

    for fname in os.listdir(path):
        if not fname.endswith((".nii", ".nii.gz")):
            continue

        fpath = os.path.join(path, fname)
        nii = nib.load(fpath)
        vol = nii.get_fdata().astype(np.float32)

        #load header and resolution as current_spacing and print current spacing and target spacing
        header = nii.header
        current_spacing = header.get_zooms()[:3]  # get voxel size
        print(f"[INFO] Current spacing for: {current_spacing}")
        print(f"[INFO] Target spacing: {target_spacing}")

        # 0. Fix LF-MRI negative values
        vol = np.abs(vol)

        # -------------------------------------------------------
        # 1. RESAMPLE ONLY USING SPACING (NO SHAPE-BASED RESIZE)
        # -------------------------------------------------------
        zoom_factors = (
            current_spacing[0] / target_spacing[0],
            current_spacing[1] / target_spacing[1],
            current_spacing[2] / target_spacing[2]
        )

        vol_iso = zoom(vol, zoom_factors, order=1)
        vol_iso = np.ascontiguousarray(vol_iso)

        h, w, d = vol_iso.shape
        print(f"[INFO] Volume {fname} resampled → {vol_iso.shape}")

        # -------------------------------------------------------
        # 2. ACCEPT ONLY 128×128 IN-PLANE RESOLUTION
        # -------------------------------------------------------
        if h != TARGET_H or w != TARGET_W:
            print(f"[SKIP] {fname} skipped — incorrect size: {vol_iso.shape}")
            continue

        # -------------------------------------------------------
        # 3. FIX DEPTH TO 35 WITHOUT DISTORTION
        # -------------------------------------------------------
        vol_fixed = crop_or_pad_depth(vol_iso, TARGET_D)

        # -------------------------------------------------------
        # 4. Normalize to [-1,1] (CycleGAN requirement)
        # -------------------------------------------------------
        #convert to grayscale
        vol_fixed = normalize_volume(vol_fixed)

        # -------------------------------------------------------
        # 5. Optional channel dimension
        # -------------------------------------------------------
        if add_channel:
            vol_fixed = vol_fixed[..., None]   # (H,W,D,1)

        print(f"[LOAD] {fname}: final shape {vol_fixed.shape}")
        volumes.append(vol_fixed)

    if not volumes:
        raise ValueError("No valid volumes loaded. Check dimensions and input path.")

    return np.stack(volumes, axis=0)

# example test

def visualize_volume_samples(gen_small, gen_large, num_samples=5):
    """
    Visualize central slices of 3D volumes from two generators (small and large domain).
    
    Args:
        gen_small: generator for small domain (yields vol, ctx)
        gen_large: generator for large domain (yields vol, ctx)
        num_samples: number of samples to visualize from each generator
    """
    # Fetch volumes (ignore context)
    vols_small = [next(gen_small)[0] for _ in range(num_samples)]
    vols_large = [next(gen_large)[0] for _ in range(num_samples)]

    # Create subplots
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    for i in range(num_samples):
        # Small domain
        vol = vols_small[i]
        z = vol.shape[2] // 2  # central slice
        axes[0, i].imshow(vol[:, :, z], cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Small {i+1}")

        # Large domain
        vol = vols_large[i]
        z = vol.shape[2] // 2
        axes[1, i].imshow(vol[:, :, z], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Large {i+1}")

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import math

def visualize_single_volume(generator, slice_axis=2):
    """
    Display all slices of a single 3D volume from a generator.
    
    Args:
        generator: Python generator yielding (vol, ctx)
        slice_axis: axis along which to slice the volume (0=H, 1=W, 2=D)
    """
    vol, _ = next(generator)  # get one volume, ignore context

    # Determine number of slices along chosen axis
    num_slices = vol.shape[slice_axis]
    cols = 8  # number of columns in grid
    rows = math.ceil(num_slices / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    axes = axes.flatten()

    for i in range(num_slices):
        if slice_axis == 0:
            img = vol[i, :, :]
        elif slice_axis == 1:
            img = vol[:, i, :]
        else:
            img = vol[:, :, i]

        axes[i].imshow(img.T, cmap='gray', origin='lower')
        axes[i].axis('off')
        axes[i].set_title(f'Slice {i}')

    # Turn off any unused subplots
    for j in range(num_slices, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# -----------------------------
# DATA LOADING
# -----------------------------
def load_subject_day_data(subject, day, folder=data_folder):
    file_path = os.path.join(folder, f"{subject}_day{day}_train_data.npy")
    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True).item()
        X = data["x_train"].astype(np.float32)[np.newaxis, ...]
        y = data["y_train"].astype(np.float32)[np.newaxis, ...]
        return X, y
    else:
        raise FileNotFoundError(f"{file_path} does not exist!")

def load_data_for_days(subjects, days):
    X_list, y_list = [], []
    for subject in subjects:
        for day in days:
            try:
                X_sub, y_sub = load_subject_day_data(subject, day)
                X_list.append(X_sub)
                y_list.append(y_sub)
            except FileNotFoundError:
                print(f"[WARNING] Missing data for subject {subject}, day {day}")
    if X_list:
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y
    else:
        return None, None
    
# write a function to visualize slices from the random volume
def visualize_slices(volume, n_cols=5):
    """
    Visualize slices from a 3D volume.
    """
    n_slices = volume.shape[2]
    n_rows = (n_slices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_slices):
        axes[i].imshow(volume[:, :, i], cmap="gray")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

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

def resample_volume(volume, current_spacing=(1,1,2), target_spacing=(1,1,1), order=3):
    """
    Resample a 3D volume to new voxel spacing.

    Parameters:
    volume : np.ndarray
        3D volume (D, H, W)
    current_spacing : tuple
        Original voxel spacing (z, y, x)
    target_spacing : tuple
        Desired voxel spacing (z, y, x)
    order : int
        Interpolation order:
        0 = nearest (labels)
        1 = linear
        3 = cubic (recommended for MRI)

    Returns:
    resampled_volume : np.ndarray
    """

    zoom_factors = (
        current_spacing[0] / target_spacing[0],
        current_spacing[1] / target_spacing[1],
        current_spacing[2] / target_spacing[2],
    )

    resampled_volume = zoom(volume, zoom_factors, order=order)
    return resampled_volume

def center_crop_volume(vol, size=128):
    H, W, D = vol.shape
    start_h = (H - size) // 2
    start_w = (W - size) // 2
    return vol[start_h:start_h+size, start_w:start_w+size, :]

# dataset path
# Check generators;
import os, random
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler


def visualize_comparison(
    im,
    pred1,
    pred2,
    name='comparison',
    output_dir='outputs_59228/trial1',
    affine=None,
    slice_range=(17, 21),
    expose_individual_png_paths=False,
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
    # plt.savefig(save_path_high, bbox_inches='tight', pad_inches=0.05, dpi=600)
    plt.show()

    # Save as NIfTI .nii.gz
    if affine is None:
        affine = np.eye(4, dtype=np.float32)

    # nii_original = os.path.join(output_dir, f"{name}_original.nii.gz")
    nii_denoiser = os.path.join(output_dir, name)
    # nii_srr      = os.path.join(output_dir, f"{name}_srr.nii.gz")

    # nib.save(nib.Nifti1Image(im_np,    affine), nii_original)
    nib.save(nib.Nifti1Image(pred1_np, affine), nii_denoiser)
    # nib.save(nib.Nifti1Image(pred2_np, affine), nii_srr)

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
    # print(f"🧠 Saved NIfTI volumes: {nii_original}, {nii_denoiser}, {nii_srr}")

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
    
    return nii_denoiser
    # return {
    #     "fig_paths": {"std": save_path_std, "high": save_path_high},
    #     "nii_paths": {"original": nii_original, "denoiser": nii_denoiser, "srr": nii_srr},
    #     "png_high_paths": png_high_paths
    # }


class DomainAGenerator:
    def __init__(self, path, batch_size=1, target_h=128, target_w=128, target_d=40, 
                 target_spacing=(1,1,2), field_strength=0.05, rotate=True, visit=1, shuffle=True):
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
    
    # NORMALIZATION
    # -----------------------------
    def _normalize_volume(self, vol, method='minmax'):
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

    def _load_volume(self, fpath):
        nii = nib.load(fpath)
        
        vol = np.abs(nii.get_fdata().astype(np.float32))  # Fix negatives
        current_spacing = nii.header.get_zooms()[:3]

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
        
        # Normalize using min-max normalization to [0, 1]
        vol = self._normalize_volume(vol, method='minmax')

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
            base_params_path = 'niv_raw_data/Nipah_IRF_data/data_niv/LFMRI_DATA_IRF_ALL_PARAMS_1'
            ctx = self._create_context(base_params_path, fpath=fpath, default_context=False)
            ctx = self.scaler.transform(ctx)
            save_file = os.path.basename(fpath)
            print(f"[GENERATOR] Yielding {save_file}")
            yield vol, ctx[0], save_file

# Example paths (use your config_lf paths)
path_A = config_lf.path_lf_test
genA = DomainAGenerator(path_A)

# Fetch one batch from Domain A
for volA, ctxA, save_file in genA:
    print("Domain A volume shape:", volA.shape)
    print("Domain A context:", ctxA.shape)
    print("Domain A context values:", ctxA)
    print("Domain A save file:", save_file)
    break

# im = np.expand_dims(im, axis=0)  # (1, H, W, D)
# print("Final LF input shape:", im.shape)

trial1 = False
trial2 = True
# unsharp_mask_ = True

model_name = 'residual_srr_unet_l1_l2_ssim_l2_ssim_edge'
folder_path = "niv_results/outputs_src_simulated/Output_patch_noise"
output_dir_denoise ='niv_raw_data/Nipah_IRF_data/data_niv/Low_field_data_DA/LFMRI_DATA_T1w_denoise'
output_dir_enhance ='niv_raw_data/Nipah_IRF_data/data_niv/Low_field_data_DA/LFMRI_DATA_T1w_enhance'
output_dir = output_dir_denoise

for volA, ctxA, save_file in genA:

    print("Domain A volume shape:", volA.shape)
    print("Domain A context:", ctxA.shape)
    print("Domain A context values:", ctxA)
    # im = im.astype(np.float32)
    volA = np.expand_dims(volA, axis=0)  # (1, H, W, D)
    print("Final LF input shape:", volA.shape)
    
    # Domain adaptation and get results of full volume prediction of low-field MRI Then denoise and enhancement.

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
    # Convert inputs to numpy
    volA = np.squeeze(volA, axis=0)

    im_np    = _to_numpy(volA)
    pred1_np = _to_numpy(pred1)
    pred2_np = _to_numpy(pred2)

    # nii_original = os.path.join(output_dir, f"{name}_original.nii.gz")
    nii_denoiser = os.path.join(output_dir, save_file)
    # nii_srr      = os.path.join(output_dir, f"{name}_srr.nii.gz")
    # if nii_denoiser path not present create it
    if not os.path.exists(output_dir_denoise):
        os.makedirs(output_dir_denoise)

    # pred1_np is saved and displayed  rotated 90 degrees back to original orientation (undoing the rotation in the generator)
    pred1_np = np.rot90(pred1_np, k=-1, axes=(0, 1))
    
    affine=None
    # Save as NIfTI .nii.gz
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
        # nib.save(nib.Nifti1Image(im_np,    affine), nii_original)
    nib.save(nib.Nifti1Image(pred1_np, affine), nii_denoiser)

    visualize_comparison(volA, pred1, pred2, name=save_file, output_dir=output_dir_denoise)