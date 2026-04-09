# ------------------------------------------------------------
# Author: Ajay Sharma
# Purpose: Low-Field MRI → Super-Resolution Reconstruction (SRR)
# Description:
#   Loads LF .3d data, performs preprocessing, applies trained SRR model,
#   and visualizes results (denoising + super-resolution).
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Standard Library Imports
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
from src_simulated.evaluate_niv_lf_test import evaluate_model, predict_volume
from src_niv.metrics import psnr, ssim, mse, composite_loss
from src_niv.utils import visualize_pair

from src_simulated.cyclegan_models.config_lf import config_lf
from src_simulated.cyclegan_models.losses_cyclegan import *
from src_simulated.cyclegan_models.models import *
from src_simulated.cyclegan_models.train_utils_params_parallel import *

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
    slice_range=(10, 20),
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
        paths["denoised"] = os.path.join(output_dir, f"{name}_denoised.nii.gz")
        nib.save(nib.Nifti1Image(pred1, affine), paths["denoised"])

        if pred2 is not None:
            paths["srr"] = os.path.join(output_dir, f"{name}_srr.nii.gz")
            nib.save(nib.Nifti1Image(pred2, affine), paths["srr"])

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
    
    # break
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