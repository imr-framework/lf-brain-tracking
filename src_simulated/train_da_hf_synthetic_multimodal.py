# ------------------------------------------------------------
# Author: Ajay Sharma
# Purpose: Low-Field MRI → Super-Resolution Reconstruction (SRR)
# Description:
#   Loads LF .3d data, performs preprocessing, applies trained SRR model,
#   and visualizes results (denoising + super-resolution).
# ------------------------------------------------------------

# ------------------------------------------------------------
# Standard Library Imports
# ------------------------------------------------------------
import sys
sys.path.insert(0, './')
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

output_path = "niv_results/Output_patch_noise"
os.makedirs(output_path, exist_ok=True)

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


from src_simulated.train_scripts.niv_srr_simulated_training_patch import (
    load_data_for_days,
    normalize_dataset,
    srr_batch_generator,
    build_or_load_model,
    compile_model,
    train_model
)
from src_niv.models.ResUNet import residual_srr_unet
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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.keras.metrics import MeanSquaredError
from src_simulated.losses import *

model_type = residual_srr_unet
model_name = "residual_srr_unet"

# ---------------------------------
# Machine Learning / Deep Learning
# ---------------------------------

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
from src_simulated.cyclegan_models.losses_cyclegan import *
from src_simulated.cyclegan_models.models import *
from src_simulated.cyclegan_models.train_utils_params_parallel import *


from src_simulated.cyclegan_models.config_lf import config_lf
from src_simulated.config import config
# ============================================================
# CONFIGURATION WRAPPER
# ============================================================
class Config:
    """Wrapper for config_lf to avoid repetitive calls."""
    
    # General
    EPOCHS = config_lf.EPOCHS
    TEST = config_lf.TEST
    SLICES_TEST = config_lf.SLICES_TEST

    # Discriminator
    DISC_LOSS = config_lf.DISC_LOSS
    DISC_LR = config_lf.DISC_LEARNING_RATE
    DISC_BETA_1 = config_lf.DISC_BETA_1
    DISC_LOSS_WEIGHTS = config_lf.DISC_LOSS_WEIGHTS

    # Generator
    GEN_LOSS = [
        config_lf.GEN_LOSS_1,
        config_lf.GEN_LOSS_2,
        config_lf.GEN_LOSS_3,
        config_lf.GEN_LOSS_4,
    ]
    GEN_LR = config_lf.GEN_LEARNING_RATE
    GEN_BETA_1 = config_lf.GEN_BETA_1
    GEN_LOSS_WEIGHTS = config_lf.GEN_LOSS_WEIGHTS

    # Training
    INITIAL_LR = config_lf.INITIAL_LR
    N_ITER = config_lf.N_ITER
    N_ITER_DECAY = config_lf.N_ITER_DECAY

    # Output
    OUTPUT_DIR = config_lf.OUTPUT_DIR
    VISUALIZE = config_lf.VISUALIZE
    N_SLICES = config_lf.N_SLICES


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def _to_numpy(x):
    """Convert tensor/array to numpy."""
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return np.asarray(x)


def _get_slice(volume, idx):
    """Extract a 2D slice safely from 3D volume."""
    v = np.squeeze(volume)
    if v.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {v.shape}")
    return v[:, :, idx]


def _prepare_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def visualize_volume_samples(gen_small, gen_large, num_samples=5):
    """Compare central slices from two generators."""
    
    vols_small = [next(gen_small)[0] for _ in range(num_samples)]
    vols_large = [next(gen_large)[0] for _ in range(num_samples)]

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    for i in range(num_samples):
        for row, vol in enumerate([vols_small[i], vols_large[i]]):
            z = vol.shape[2] // 2
            axes[row, i].imshow(vol[:, :, z], cmap='gray')
            axes[row, i].axis('off')
            axes[row, i].set_title(
                f"{'Small' if row == 0 else 'Large'} {i+1}"
            )

    plt.tight_layout()
    plt.show()


def visualize_single_volume(generator, axis=2):
    """Display all slices of one volume."""
    
    vol, _ = next(generator)
    num_slices = vol.shape[axis]

    cols = 8
    rows = math.ceil(num_slices / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    axes = axes.flatten()

    for i in range(num_slices):
        if axis == 0:
            img = vol[i]
        elif axis == 1:
            img = vol[:, i, :]
        else:
            img = vol[:, :, i]

        axes[i].imshow(img.T, cmap='gray', origin='lower')
        axes[i].set_title(f"Slice {i}")
        axes[i].axis('off')

    for j in range(num_slices, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_comparison(
    im,
    pred1,
    pred2=None,
    name="comparison",
    output_dir="outputs",
    affine=None,
    slice_range=(10, 25),
    save_nifti=True,
    show_ortho=False
):
    """
    Compare original vs predictions across slices.
    """

    _prepare_output_dir(output_dir)

    im = _to_numpy(im)
    pred1 = _to_numpy(pred1)
    pred2 = _to_numpy(pred2) if pred2 is not None else None

    volumes = [im, pred1] + ([pred2] if pred2 is not None else [])
    titles = ["Original", "Denoised"] + (["SRR"] if pred2 is not None else [])

    # Slice handling
    start, end = slice_range
    max_depth = min(v.shape[-1] for v in volumes if v is not None)
    end = min(end, max_depth - 1)

    slice_ids = list(range(start, end + 1))

    fig, axes = plt.subplots(len(volumes), len(slice_ids),
                             figsize=(4*len(slice_ids), 4*len(volumes)))

    if len(volumes) == 1:
        axes = np.expand_dims(axes, 0)

    for col, idx in enumerate(slice_ids):
        for row, (vol, title) in enumerate(zip(volumes, titles)):
            ax = axes[row, col]
            ax.imshow(_get_slice(vol, idx), cmap='gray')
            if col == 0:
                ax.set_ylabel(title)
            if row == 0:
                ax.set_title(f"Slice {idx}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Save NIfTI
    if save_nifti:
        if affine is None:
            affine = np.eye(4)

        paths = {}
        paths["denoised"] = os.path.join(output_dir, f"{name}")
        nib.save(nib.Nifti1Image(pred1, affine), paths["denoised"])

        # if pred2 is not None:
        #     paths["srr"] = os.path.join(output_dir, f"{name}_srr.nii.gz")
        #     nib.save(nib.Nifti1Image(pred2, affine), paths["srr"])

        print(f"✅ Saved NIfTI: {paths}")

    # Optional Ortho view
    if show_ortho:
        for label, vol in zip(titles, volumes):
            if vol is None:
                continue
            try:
                OrthoSlicer3D(vol).show()
            except Exception as e:
                print(f"⚠️ Ortho view failed for {label}: {e}")

    return paths if save_nifti else None

    
# write a function to visualize slices from the random volume
def visualize_slices(volume, n_cols=5):
    """
    Visualize slices from a 3D volume.
    """
    if volume.ndim == 4:
        volume = np.squeeze(volume)
    n_slices = volume.shape[2]
    n_rows = (n_slices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_slices):
        axes[i].imshow(volume[:, :, i], cmap="gray")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

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
        vol = (vol / np.max(vol) - 0.5) * 2 if np.max(vol) > 0 else vol
        # vol = self._normalize_volume(vol, method='zscore')

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

def generate_volume_from_generator(g_model, vol_3d, context_vec, batch_size=8):
    """
    vol_3d: (H, W, D) in [-1,1]
    context_vec: (C,) or (1,C)
    returns: fake_vol_3d (H, W, D) in [-1,1]
    """
    if context_vec.ndim == 1:
        context_vec = context_vec[None, :]  # (1,C)

    H, W, D = vol_3d.shape
    # (D,H,W,1)
    x = np.transpose(vol_3d, (2, 0, 1)).astype(np.float32)[..., None]

    out_slices = []
    for s in range(0, D, batch_size):
        xb = x[s:s+batch_size]                           # (B,H,W,1)
        cb = np.repeat(context_vec, xb.shape[0], axis=0) # (B,C)
        yb = g_model.predict([xb, cb], verbose=0)         # (B,H,W,1)
        out_slices.append(yb)

    y = np.concatenate(out_slices, axis=0)  # (D,H,W,1)
    y = y[..., 0]                            # (D,H,W)
    fake_vol = np.transpose(y, (1, 2, 0))    # (H,W,D)
    return fake_vol

def evaluate_one_subject_volume(g_model, X_vol, c_vol, out_path=None, nii_affine=None, batch_size=8):
    """
    Pulls ONE subject from `data_gen`, generates the full translated volume, and optionally saves.
    Assumes data_gen yields (X_vol, c_vol), where X_vol is (1,H,W,D) or (H,W,D),
    and c_vol is (1,C) or (C,).
    """
    if X_vol.ndim == 4:
        X_vol = X_vol[0]
    if c_vol.ndim == 2:
        c_vol = c_vol[0]

    fake_vol = generate_volume_from_generator(g_model, X_vol, c_vol, batch_size=batch_size)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if nii_affine is None:
            nii_affine = np.eye(4)
        nib.save(nib.Nifti1Image(fake_vol.astype(np.float32), affine=nii_affine), out_path)
        print(f"> Saved generated volume: {out_path}")

    return X_vol, fake_vol, c_vol

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

def center_crop_volume(vol, size=128):
    H, W, D = vol.shape
    start_h = (H - size) // 2
    start_w = (W - size) // 2
    return vol[start_h:start_h+size, start_w:start_w+size, :]

# Load Generator of the Domain B (HF MRI) trained with CycleGAN for domain adaptation
class DomainBGenerator:
    def __init__(self, path, substring=None, target_spacing=(1,1,2),
                 target_h=140, target_w=140, target_d=35,crop_size=128,
                 rotate=True, add_channel=False, test=False):
        """
        Generator for Domain B volumes and context vectors.
        """
        self.path = path
        self.substring = substring
        self.target_spacing = target_spacing
        self.target_h = target_h
        self.target_w = target_w
        self.target_d = target_d
        self.crop_size = crop_size
        self.rotate = rotate
        self.add_channel = add_channel
        self.test = test

        # -----------------------------
        # Find valid NIfTI files
        # -----------------------------
        self.files = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames.sort()
            filenames.sort()
            for fname in filenames:
                if fname.endswith((".nii", ".nii.gz")) and (substring is None or substring in fname):
                    self.files.append(os.path.join(dirpath, fname))

        self.files.sort(key=lambda x: (os.path.dirname(x), os.path.basename(x)))

        if len(self.files) == 0:
            raise ValueError(f"No files found in {path} with substring={substring}")

        # Limit for test mode
        if self.test:
            self.files = self.files[:5]

        # Fit context scaler on dummy context
        self.scaler = StandardScaler()
        dummy_ctx = self._create_context(len(self.files))
        self.scaler.fit(dummy_ctx)

    def _create_context(self, N):

        params = {
            "TE": 20000,
            "TR": 3000,
            "bandwidth": 200,
            "etLength": 12,
            "field_strength": 3.0
        }

        context = np.stack(
            [np.full(N, v) for v in params.values()],
            axis=1
        )

        return context.astype(np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load one volume (with preprocessing) and return (volume, context)
        """
        fpath = self.files[idx]
        nii = nib.load(fpath)
        vol = np.abs(nii.get_fdata().astype(np.float32))  # remove negatives
        header = nii.header
        current_spacing = header.get_zooms()[:3]

        # Resample
        zoom_factors = (
            current_spacing[0] / self.target_spacing[0],
            current_spacing[1] / self.target_spacing[1],
            current_spacing[2] / self.target_spacing[2]
        )
        vol = zoom(vol, zoom_factors, order=1)
        vol = np.ascontiguousarray(vol)

        h, w, d = vol.shape
        # Skip if in-plane resolution wrong
        if h != self.target_h or w != self.target_w:
            raise ValueError(f"Skipped {os.path.basename(fpath)}: in-plane size {vol.shape}")

        # Crop/pad depth
        vol = crop_or_pad_depth(vol, self.target_d)

        # Normalize
        vol = normalize_volume(vol)

        # Rotate if needed
        if self.rotate:
            vol = np.rot90(vol, k=3, axes=(0,1))


        # ---- Apply crop BEFORE adding channel ----
        H, W, D = vol.shape
        if self.crop_size is not None and (H > self.crop_size or W > self.crop_size):
            vol = center_crop_volume(vol, self.crop_size)
        
        if self.add_channel:
            vol = vol[..., None]  # H x W x D x 1

        # Keep vol only 0 to 30 slices in a volume  depth dimension
        vol = vol[..., :30]

        # Create context for this single volume
        context = self._create_context(1)[0]  # shape (9,)

        return vol, context

    def generator(self):
        """
        Python generator yielding (volume, context) indefinitely
        """
        while True:
            idxs = np.random.permutation(len(self.files))
            for idx in idxs:
                try:
                    vol, ctx = self.__getitem__(idx)
                    yield vol, ctx
                except ValueError:
                    continue


# ...existing code...
def ensure_minus1_to_1(vol):
    """Force volume into [-1,1] (safe if already in range)."""
    vol = np.asarray(vol, dtype=np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    # If it already looks like [-1,1], keep it
    if vmin >= -1.05 and vmax <= 1.05:
        return np.clip(vol, -1.0, 1.0)

    # If it's [0,1]-ish, map to [-1,1]
    if vmin >= -0.05 and vmax <= 1.05:
        return (np.clip(vol, 0.0, 1.0) - 0.5) * 2.0

    # Otherwise do min-max to [-1,1]
    if vmax > vmin:
        vol01 = (vol - vmin) / (vmax - vmin)
        return (vol01 - 0.5) * 2.0
    return np.zeros_like(vol, dtype=np.float32)

# ... existing code ...

def build_synthetic_lf_dataset_from_hf(genB, g_model_BtoA, n_vols=50, batch_size_slices=4):
    """
    Creates:
      x_fake_lf: (N,H,W,D) generated by g_model_BtoA from HF volumes
      y_hf:      (N,H,W,D) the original HF volumes (same as your y_train concept)
      ctx:       (N,C) contexts used
    """
    x_fake_lf, y_hf, ctxs = [], [], []
    for k, (volB, ctxB) in enumerate(genB):
        if k >= n_vols:
            break

        # ensure input to CycleGAN is 3D and normalized
        volB = ensure_minus1_to_1(volB)  # (H,W,D)

        # generate full translated volume (H,W,D)
        _, fake_vol, _ = evaluate_one_subject_volume(
            g_model_BtoA, volB, ctxB,
            out_path=None,
            batch_size=1
        )

        fake_vol = ensure_minus1_to_1(fake_vol)

        # store paired volumes for SRR: X=LF(fake), Y=HF(real)
        x_fake_lf.append(fake_vol.astype(np.float32))
        y_hf.append(volB.astype(np.float32))
        ctxs.append(np.asarray(ctxB, dtype=np.float32))

        print(f"[{k+1}/{n_vols}] built pair: X_fake {fake_vol.shape} Y_real {volB.shape}")

    return np.stack(x_fake_lf, axis=0), np.stack(y_hf, axis=0), np.stack(ctxs, axis=0)

# -----------------------------
# MODEL BUILD / COMPILE / TRAIN
# -----------------------------
def build_or_load_model(model_type, checkpoint_path, input_shape=(64,64,32,1)):
    if os.path.exists(checkpoint_path):
        print(f"✅ Loading model from checkpoint: {checkpoint_path}")
        model = load_model(checkpoint_path, compile=False)
    else:
        print("⚠️ Checkpoint not found. Building new model...")
        model = model_type(config.input_shape)
    return model

# Define custom PSNR and SSIM metrics
def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def compile_model(model, lr=0.001, loss_type='l1_l2_ssim'):
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    mse_metric = MeanSquaredError(name='mse')

    # --- Select Loss ---
    if loss_type == 'l1':
        loss_fn = l1_loss
    elif loss_type == 'l2':
        loss_fn = l2_loss
    elif loss_type == 'ssim':
        loss_fn = ssim_loss
    elif loss_type == 'l1_l2_ssim':
        loss_fn = l1_l2_ssim_loss
    elif loss_type == 'l2_ssim':
        loss_fn = l2_ssim_loss
    elif loss_type == 'l1_ssim':
        loss_fn = l1_ssim_loss
    elif loss_type == 'l2_ssim_edge':
        loss_fn = l2_ssim_edge_loss
    elif loss_type == 'l1_l2_ssim_edge':
        loss_fn = l1_l2_ssim_edge_loss
    elif loss_type == 'l2_edge_gram_matrix_loss':
        loss_fn = l2_edge_gram_matrix_loss
    else:
        raise ValueError("❌ Invalid loss_type. Choose from: ['l1', 'l2', 'ssim', 'l1_ssim', 'l2_ssim', 'l1_l2_ssim', 'mse_ssim_edge'].")

    # --- compile ---
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[mse_metric,
                 psnr_metric,
                 ssim_metric]
    )

    print(f"✅ Model compiled with {loss_type.upper()} loss and lr={lr}")
    return model

# -----------------------------
# PADDING UTILITIES
# -----------------------------
def pad_volume_to_multiple(vol, multiple=8):
    h, w, d = vol.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_d = (multiple - d % multiple) % multiple

    pad_before = (pad_h//2, pad_w//2, pad_d//2)
    pad_after = (pad_h - pad_before[0], pad_w - pad_before[1], pad_d - pad_before[2])

    padded_vol = np.pad(vol, ((pad_before[0], pad_after[0]),
                              (pad_before[1], pad_after[1]),
                              (pad_before[2], pad_after[2])),
                        mode='constant', constant_values=0)
    pad_widths = (pad_before, pad_after)
    return padded_vol, pad_widths

def unpad_volume(vol, pad_widths):
    (h_before, w_before, d_before), (h_after, w_after, d_after) = pad_widths
    return vol[h_before:vol.shape[0]-h_after,
               w_before:vol.shape[1]-w_after,
               d_before:vol.shape[2]-d_after]

# -----------------------------
# ROTATION AUGMENTATION
# -----------------------------
def rotation_3d(volume, angle, axes=(0,1)):
    return rotate(volume, angle, axes=axes, reshape=False, mode='reflect')

def srr_batch_generator(
        lf_volumes, hf_volumes,
        batch_size=32,
        patch_xy=64, patch_z=32,
        patches_per_volume=8,
        augment=True,
        augmentations_per_patch=3,
        angles=(0, 5, 10, 15, 20, 45, 60, 90),
        jitter=5,
        intensity_aug=False,
        noise_std=0.01,
        gaussian_blur=False
    ):
    """
    3D SRR Patch Generator for patch-based training.

    Parameters:
    -----------
    lf_volumes, hf_volumes : np.ndarray
        Arrays of shape (N, H, W, D).
    batch_size : int
        Number of total samples per batch.
    patches_per_volume : int
        Random patches extracted per volume.
    augmentations_per_patch : int
        Number of augmentations per patch.
    """
    n_samples = lf_volumes.shape[0]
    volume_order = np.random.permutation(n_samples)
    vol_idx = 0

    while True:
        x_batch, y_batch = [], []

        while len(x_batch) < batch_size:
            idx = volume_order[vol_idx]
            lf = lf_volumes[idx]
            hf = hf_volumes[idx]

            vol_idx = (vol_idx + 1) % n_samples
            if vol_idx == 0:
                volume_order = np.random.permutation(n_samples)

            lf, _ = pad_volume_to_multiple(lf, multiple=8)
            hf, _ = pad_volume_to_multiple(hf, multiple=8)
            H, W, D = lf.shape

            for _ in range(patches_per_volume):
                # Extract random patch
                z_start = np.random.randint(0, max(1, D - patch_z))
                y_start = np.random.randint(0, max(1, H - patch_xy))
                x_start = np.random.randint(0, max(1, W - patch_xy))

                lf_patch = lf[y_start:y_start + patch_xy,
                              x_start:x_start + patch_xy,
                              z_start:z_start + patch_z]
                hf_patch = hf[y_start:y_start + patch_xy,
                              x_start:x_start + patch_xy,
                              z_start:z_start + patch_z]

                #  Assuming hf_patch is a NumPy 3D array (e.g., shape (H, W, D))
                # hf_patch[hf_patch < 0.01] = 0
                # # continue patche generation with patches_per_volume-1 with zeros more than 60% in hf_patch
                # if np.sum(hf_patch == 0) / hf_patch.size > 0.7:
                # 
                # Add original patch

                x_batch.append(np.expand_dims(lf_patch, axis=-1))
                y_batch.append(np.expand_dims(hf_patch, axis=-1))

                # Add multiple augmentations per patch
                if augment:
                    for _ in range(augmentations_per_patch):
                        aug_lf, aug_hf = augment_patch_pair(
                            lf_patch.copy(), hf_patch.copy(),
                            angles, jitter, intensity_aug, noise_std, gaussian_blur
                        )
                        x_batch.append(np.expand_dims(aug_lf, axis=-1))
                        y_batch.append(np.expand_dims(aug_hf, axis=-1))

        # Stack and trim
        x_batch = np.stack(x_batch[:batch_size], axis=0)
        y_batch = np.stack(y_batch[:batch_size], axis=0)
        # print(f"Generated batch: X {x_batch.shape} Y {y_batch.shape}")
        yield x_batch.astype(np.float32), y_batch.astype(np.float32)


# Helper for augmentations
def augment_patch_pair(lf_patch, hf_patch, angles, jitter, intensity_aug, noise_std, gaussian_blur):
    base_angle = random.choice(angles)
    sign = random.choice([-1, 1])
    angle = sign * (base_angle + random.uniform(0, jitter))
    axes = random.choice([(0, 1), (0, 2), (1, 2)])

    aug_lf = rotation_3d(lf_patch, angle, axes)
    aug_hf = rotation_3d(hf_patch, angle, axes)

    if gaussian_blur:
        lf_sigma_lb = 1
        lf_sigma_ub = 1.25
        aug_lf = gaussian_filter(aug_lf, sigma=random.uniform(lf_sigma_lb, lf_sigma_ub))

    # visualize_pair(aug_lf, aug_hf, slice_indices = [10,12,14,16,18,20])
    # visualize_pair(lf_patch, aug_lf, slice_indices = [10,12,14,16,18,20])


    # if random.random() < 0.5:
    #     flip_axis = random.choice([0, 1, 2])
    #     aug_lf = np.flip(aug_lf, axis=flip_axis)
    #     aug_hf = np.flip(aug_hf, axis=flip_axis)

    if intensity_aug:
        scale = 1.0 + 0.2 * (np.random.rand() - 0.5)
        aug_lf *= scale
        aug_hf *= scale

        mean_lf = aug_lf.mean()
        mean_hf = aug_hf.mean()
        contrast_factor = 1.0 + 0.2 * (np.random.rand() - 0.5)
        aug_lf = (aug_lf - mean_lf) * contrast_factor + mean_lf
        aug_hf = (aug_hf - mean_hf) * contrast_factor + mean_hf

        aug_lf += np.random.normal(0, noise_std, aug_lf.shape)
        aug_hf += np.random.normal(0, noise_std, aug_hf.shape)

        aug_lf = np.clip(aug_lf, 0, np.max(aug_lf))
        aug_hf = np.clip(aug_hf, 0, np.max(aug_hf))

    return aug_lf, aug_hf

# ...existing code...

def run_training(lf_train, hf_train, lf_val, hf_val,
                 output_path, model_type=residual_srr_unet, model_name='ResUNet',
                 loss_type='l1_l2_ssim', patch_xy=64, patch_z=32,
                 batch_size=32, steps_per_epoch=48, epochs=500,
                 model=None, initial_epoch=0):
    """
    If `model` is provided, continues training it from initial_epoch.
    Otherwise builds/loads from checkpoint_path as before.
    """
    if model is None:
        model = build_or_load_model(model_type, checkpoint_path=config.checkpoint_path)
        model = compile_model(model, lr=0.001, loss_type=loss_type)

    # Generators
    train_gen = srr_batch_generator(
        lf_volumes=lf_train,
        hf_volumes=hf_train,
        batch_size=batch_size,
        patch_xy=patch_xy,
        patch_z=patch_z,
        augment=config.train_augment
    )

    val_gen = srr_batch_generator(
        lf_volumes=lf_val,
        hf_volumes=hf_val,
        batch_size=1,
        patch_xy=patch_xy,
        patch_z=patch_z,
        augment=config.val_augment
    )

    validation_steps = max(1, len(lf_val) // 1)

    # Train (directly here so we can pass initial_epoch with minimal disruption)
    csv_log_path = os.path.join(output_path, os.path.splitext(os.path.basename(config.checkpoint_path))[0] + "_training_log.csv")

    checkpoint_cb = ModelCheckpoint(
        config.checkpoint_path,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True
    )

    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=30,
        mode='min',
        verbose=1,
        min_lr=1e-8
    )

    early_stop_cb = EarlyStopping(
        monitor='val_ssim',
        patience=60,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    csv_logger_cb = CSVLogger(csv_log_path, append=(initial_epoch > 0))

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb, csv_logger_cb]
    )

    final_model_path = os.path.join(output_path, f"{model_name}_final.keras")
    model.save(final_model_path)
    print(f"✅ Final model saved at: {final_model_path}")

    return model, history

def pick_random_b2a_checkpoint(model_path_da, seed=None):
    ckpts = sorted(glob.glob(os.path.join(model_path_da, "g_BtoA_*.keras")))
    if not ckpts:
        raise FileNotFoundError(f"No g_BtoA_*.keras found in: {model_path_da}")
    rng = np.random.default_rng(seed)
    return ckpts[int(rng.integers(0, len(ckpts)))]


# ...existing code...

def pick_random_checkpoint_from_combinations(combinations, seed=None, must_exist=True):
    """
    Simplest picker:
      combinations = [
        ("path/to/folder1", "g_BtoA_000300.keras"),
        ("path/to/folder2", "g_BtoA_000250.keras"),
        ...
      ]
    Returns:
      full_ckpt_path
    """
    if not combinations:
        raise ValueError("combinations list is empty")

    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(combinations)))

    folder, fname = combinations[idx]
    ckpt_path = os.path.join(folder, fname)

    if must_exist and not os.path.exists(ckpt_path):
        # try a few more random picks before failing
        for _ in range(min(10, len(combinations) - 1)):
            idx = int(rng.integers(0, len(combinations)))
            folder, fname = combinations[idx]
            ckpt_path = os.path.join(folder, fname)
            if os.path.exists(ckpt_path):
                return ckpt_path
        raise FileNotFoundError(f"Picked checkpoint does not exist (and retries failed): {ckpt_path}")

    return ckpt_path

# ...existing code...
# ...existing code...
from sklearn.model_selection import train_test_split

def load_all_hf_from_genB(genB, n_vols=None):
    """Load HF volumes deterministically via __getitem__ so we can split without leakage."""
    n_total = len(genB)
    n = n_total if n_vols is None else min(int(n_vols), n_total)

    vols, ctxs = [], []
    for i in range(n):
        vol, ctx = genB[i]  # deterministic
        vols.append(ensure_minus1_to_1(vol).astype(np.float32))
        ctxs.append(np.asarray(ctx, dtype=np.float32))

    return np.stack(vols, axis=0), np.stack(ctxs, axis=0)

def build_synthetic_from_arrays(hf_vols, hf_ctxs, g_model_BtoA, batch_size_slices=1):
    """Generate LF(fake) from fixed HF arrays."""
    x_fake = []
    for k in range(hf_vols.shape[0]):
        volB = ensure_minus1_to_1(hf_vols[k])
        ctxB = hf_ctxs[k]
        _, fake_vol, _ = evaluate_one_subject_volume(
            g_model_BtoA, volB, ctxB,
            out_path=None,
            batch_size=batch_size_slices
        )
        x_fake.append(ensure_minus1_to_1(fake_vol).astype(np.float32))
    return np.stack(x_fake, axis=0), hf_vols.astype(np.float32), hf_ctxs.astype(np.float32)

if __name__ == "__main__":

    # Load data
    # Example paths (use your config_lf paths)
    path_A = config_lf.path_lf_t1w  # LF T1w data directory
    genA = DomainAGenerator(path_A)

    # Fetch one batch from Domain A
    for volA, ctxA, save_file in genA:
        print("Domain A volume shape:", volA.shape)
        print("Domain A context:", ctxA.shape)
        print("Domain A context values:", ctxA)
        print("Domain A save file:", save_file)
        # visualize_slices(volA)
        break

    # Path to Domain B NIfTI files
    path_B = "niv_raw_data/Nipah_IRF_data/data_niv/IRF_3T_NIFTI"
    substring_B = "T2_n100"

    genB = DomainBGenerator(
        path=path_B,
        substring=substring_B,
        target_spacing=(1,1,2),
        target_h=140,
        target_w=140,
        target_d=35,
        rotate=True,
        add_channel=False,
        test=False
    )

    # Example: get first volume and its context
    for volB, ctxB in genB:
        print("Domain B volume shape:", volB.shape)
        print("Domain B context:", ctxB.shape)
        print("Domain B context values:", ctxB)
        # min and max
        print("Domain B context min:", volB.min())
        print("Domain B context max:", volB.max())
        # visualize slices
        # visualize_slices(volB)
        break

    # Model paths and names
    model_name = 'residual_srr_unet_l2_ssim_edge'
    folder_path = "niv_results/outputs_src_simulated_context/enhancement"

    output_dir = folder_path
    # Resume from latest checkpoints if available

    model_path_da = "niv_results/outputs_src_simulated_context/cyclegan_lfmri20t2w_2_lfsimulated_context_1000_da_all"
    model_files = {
        'g_A2B': os.path.join(model_path_da, 'g_AtoB_000300.keras'),
        'g_B2A': os.path.join(model_path_da, 'g_BtoA_000300.keras'),
        'd_A': os.path.join(model_path_da, 'd_A_000300.keras'),
        'd_B': os.path.join(model_path_da, 'd_B_000300.keras')
    }

    # make index to randomly select on of files and generate synthetic volume to train for diversity in training data; this is for models to generate synthetic data

    if all(os.path.exists(f) for f in model_files.values()):
        print(">>> Resuming from latest checkpoints...")
        g_model_AtoB = load_model(model_files['g_A2B'], compile=False)
        g_model_BtoA = load_model(model_files['g_B2A'], compile=False)
        d_model_A = load_model(model_files['d_A'], compile=False)
        d_model_B = load_model(model_files['d_B'], compile=False)
        d_model_A.compile(loss=DISC_LOSS, optimizer=Adam(learning_rate=DISC_LEARNING_RATE, beta_1=DISC_BETA_1), loss_weights=DISC_LOSS_WEIGHTS)
        d_model_B.compile(loss=DISC_LOSS, optimizer=Adam(learning_rate=DISC_LEARNING_RATE, beta_1=DISC_BETA_1), loss_weights=DISC_LOSS_WEIGHTS)


    for volB, ctxB in genB:

        print("Domain B volume shape:", volB.shape)
        print("Domain B context:", ctxB.shape)
        print("Domain B context values:", ctxB)
        # im = im.astype(np.float32)
        volB = np.expand_dims(volB, axis=0)  # (1, H, W, D)
        print("Final LF input shape:", volB.shape)

        # break
        # Domain adaptation and get results of full volume prediction of low-field MRI then denoise and enhancement.
        # generates one full volume from genA translated by g_model_AtoB

        # Print min and max range of volB before feeding into generator
        print(f"Input volume range before generator: min={volB.min()}, max={volB.max()}")
        real_vol, fake_vol, ctx = evaluate_one_subject_volume(
            g_model_BtoA, volB, ctxB,
            out_path=os.path.join(output_dir, "AtoB_generated_subject.nii.gz"),
            batch_size=1
        )

        print(f"Generated volume shape: {fake_vol.shape}")
        # add axis to fake_vol to make it (1,H,W,D) for evaluation
        fake_vol_1 = np.expand_dims(fake_vol, axis=0)
        print(f"Generated volume shape after adding batch axis: {fake_vol_1.shape}")
        # visualize_slices(fake_vol_1)
        #check in max and min of fake_vol_1 and if max > 1 then normalize in range 0 to 1

        if fake_vol_1.max() > 1:
            fake_vol_1 = (fake_vol_1 / np.max(fake_vol_1) - 0.5) * 2
        
        # print min and max of fake_vol_1 after normalization
        print(f"Generated volume range after normalization: min={fake_vol_1.min()}, max={fake_vol_1.max()}")

        # visualize_comparison(
        #     im=real_vol,
        #     pred1=fake_vol,
        #     name="domain_adaptation_comparison",
        #     output_dir=output_dir,
        #     slice_range=(10, 25),
        #     save_nifti=True,
        #     show_ortho=False
        # )
        break

    # ...existing code...

    print("🚀 Starting refinement (second-pass) training ...")
    tf.keras.backend.clear_session()

    config.model_name = f"{model_name}"
    config.checkpoint_path = os.path.join(config.output_path, f"{config.model_name}_checkpoint.keras")

    # ---- settings ----
    epochs_total = 500
    refresh_every = 50
    n_total_vols = 50
    n_train = 35
    n_val = 15
    batch_size_slices = 1

    # Load HF volumes ONCE and split (prevents train/val overlap)
    hf_all, ctx_all = load_all_hf_from_genB(genB, n_vols=n_total_vols)
    idx = np.arange(hf_all.shape[0])
    idx_train, idx_val = train_test_split(idx, test_size=n_val, random_state=42, shuffle=True)

    hf_train, ctx_train = hf_all[idx_train], ctx_all[idx_train]
    hf_val, ctx_val = hf_all[idx_val], ctx_all[idx_val]

    # Build/load SRR model once
    srr_model = build_or_load_model(residual_srr_unet, checkpoint_path=config.checkpoint_path)
    srr_model = compile_model(srr_model, lr=0.001, loss_type=config.loss_type_denoise)

    b2a_combinations = [
        # domain adoption modelstrained;
        ("niv_results/outputs_src_simulated_context/cyclegan_lfmri20t2w_2_lfsimulated_context_1000_da_all", "g_BtoA_000100.keras"),
        ("niv_results/outputs_src_simulated_context/cyclegan_lfmri20t2w_2_lfsimulated_context_1000_da_all", "g_BtoA_000200.keras"),
        ("niv_results/outputs_src_simulated_context/cyclegan_lfmri20t2w_2_lfsimulated_context_1000_da_all", "g_BtoA_000300.keras"),
        ("niv_results/outputs_src_simulated_context/cyclegan_lfmri20t2w_2_lfsimulated_context_1000_da_all", "g_BtoA_000400.keras"),


        # ("niv_results/outputs_src_simulated_context/cyclegan_lfmri20t2w_2_lfsimulated_context_2000_all", "g_BtoA_000700.keras"),
        # ("niv_results/outputs_src_simulated_context/cyclegan_lfmri20t2w_2_lfsimulated_context_2000_all", "g_BtoA_000500.keras"),
        # ("niv_results/outputs_src_simulated_context/cyclegan_lfmri20t2w_2_lfsimulated_context_2000_1", "g_BtoA_000600.keras"),
        # ("niv_results/outputs_src_simulated_context/cyclegan_lfmri20t2w_2_lfsimulated_context_1000_t2w", "g_BtoA_000400.keras")
    ]

    if len(b2a_combinations) == 1:
        b2a_ckpt = pick_random_checkpoint_from_combinations(b2a_combinations, seed=None)
        g_model_BtoA = load_model(b2a_ckpt, compile=False)

        X_train, y_train, _ = build_synthetic_from_arrays(hf_train, ctx_train, g_model_BtoA, batch_size_slices=batch_size_slices)
        X_val, y_val, _     = build_synthetic_from_arrays(hf_val,   ctx_val,   g_model_BtoA, batch_size_slices=batch_size_slices)

        visualize_slices(X_train[0])
        print(f"Training on {X_train.shape[0]} synthetic volumes (from {hf_train.shape[0]} real HF volumes)")
        print(f"Validation on {X_val.shape[0]} synthetic volumes (from {hf_val.shape[0]} real HF volumes)")

        srr_model, history = run_training(
            lf_train=X_train, hf_train=y_train,
            lf_val=X_val,     hf_val=y_val,
            output_path='niv_results/outputs_src_simulated_context/enhancement',
            model_type=residual_srr_unet,
            model_name=model_name,
            loss_type=config.loss_type_denoise,
            patch_xy=64, patch_z=32,
            batch_size=2,
            steps_per_epoch=5,
            epochs=epochs_total,
            model=srr_model,
            initial_epoch=0
        )
    else:
        current_epoch = 0
        while current_epoch < epochs_total:
            b2a_ckpt = pick_random_checkpoint_from_combinations(b2a_combinations, seed=None)
            g_model_BtoA = load_model(b2a_ckpt, compile=False)

            X_train, y_train, _ = build_synthetic_from_arrays(hf_train, ctx_train, g_model_BtoA, batch_size_slices=batch_size_slices)
            X_val, y_val, _     = build_synthetic_from_arrays(hf_val,   ctx_val,   g_model_BtoA, batch_size_slices=batch_size_slices)

            # visualize_slices(X_train[0])

            print(f"Training on {X_train.shape[0]} synthetic volumes (from {hf_train.shape[0]} real HF volumes)")
            print(f"Validation on {X_val.shape[0]} synthetic volumes (from {hf_val.shape[0]} real HF volumes)")
            # print entire shape of X_train and y_train
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

            next_epoch = min(current_epoch + refresh_every, epochs_total)

            srr_model, history = run_training(
                lf_train=X_train, hf_train=y_train,
                lf_val=X_val,     hf_val=y_val,
                output_path='niv_results/outputs_src_simulated_context/enhancement',
                model_type=residual_srr_unet,
                model_name=model_name,
                loss_type=config.loss_type_denoise,
                patch_xy=64, patch_z=32,
                batch_size=32,
                steps_per_epoch=35,
                epochs=next_epoch,
                model=srr_model,
                initial_epoch=current_epoch
            )
            current_epoch = next_epoch
