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
from src_simulated.test_scripts_simulation.evaluate_niv_lf_test import evaluate_model, predict_volume
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
    fig.suptitle(name, fontsize=18)  # Set the title at the top of the figure

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
    # plt.show()

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
        self.apply_brain_extraction = True  # Optional: set to False if you want to skip brain extraction

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
        
        # Brain extraction
        if self.apply_brain_extraction:
            vol = self._extract_brain_volume(vol)
            
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

# ------------------------------------------------
# Sliding Window Inference
# ------------------------------------------------
def predict_volume(model, lf_volume, patch_size=(64,64,32), overlap=0.5):
    """
    Sliding-window 3D prediction on LF volume.
    Returns predicted enhanced volume of same shape.
    """
    H, W, D = lf_volume.shape
    px, py, pz = patch_size

    # Strides
    sx = int(px * (1 - overlap))
    sy = int(py * (1 - overlap))
    sz = int(pz * (1 - overlap))

    pad_x = (px - H % px) if H % px != 0 else 0
    pad_y = (py - W % py) if W % py != 0 else 0
    pad_z = (pz - D % pz) if D % pz != 0 else 0

    lf_padded = np.pad(lf_volume, ((0,pad_x),(0,pad_y),(0,pad_z)), mode='reflect')
    H_pad, W_pad, D_pad = lf_padded.shape

    pred_volume = np.zeros_like(lf_padded, dtype=np.float32)
    count_volume = np.zeros_like(lf_padded, dtype=np.float32)

    for x in range(0, H_pad - px + 1, sx):
        for y in range(0, W_pad - py + 1, sy):
            for z in range(0, D_pad - pz + 1, sz):
                patch = lf_padded[x:x+px, y:y+py, z:z+pz]
                patch_input = np.expand_dims(patch, axis=(0,-1))
                pred_patch = model.predict(patch_input, verbose=0)
                pred_patch = np.squeeze(pred_patch)
                pred_volume[x:x+px, y:y+py, z:z+pz] += pred_patch
                count_volume[x:x+px, y:y+py, z:z+pz] += 1.0

    pred_volume /= np.maximum(count_volume, 1e-8)
    return pred_volume[:H, :W, :D]

def evaluate_model(folder_path, model_name, X_test, y_test,
                   patch_size=(64, 64, 32), overlap=0.5,
                   visualize_slices=[17]):
    """
    Evaluate a two-stage (base + retrained) SRR or denoising model pipeline.

    Args:
        folder_path (str): Path containing the model checkpoints.
        model_name (str): Base model name (without suffixes).
        X_test, y_test: Test LF and HF volumes (numpy arrays).
        patch_size (tuple): Patch size for 3D inference.
        overlap (float): Overlap ratio for sliding window inference.
        visualize_slices (list): Slice indices for visualization.

    Example:
        evaluate_model("models/", "LFHF_SRR_Model", X_test, y_test)
        -> loads:
            models/LFHF_SRR_Model_checkpoint.keras
            models/LFHF_SRR_Model_retrained_checkpoint.keras
    """

    # ------------------------------------------------------------
    # 🔹 Define model paths
    # ------------------------------------------------------------
    model_path_train = os.path.join(folder_path, f"{model_name}_checkpoint.keras")
    # model_path_retrained = os.path.join(folder_path, f"{model_name}_retrained_final.keras")

    if not os.path.exists(model_path_train):
        raise FileNotFoundError(f"Base model not found: {model_path_train}")
    # if not os.path.exists(model_path_retrained):
    #     raise FileNotFoundError(f"Retrained model not found: {model_path_retrained}")

    print(f"🔹 Base model: {model_path_train}")
    # print(f"🔹 Retrained model: {model_path_retrained}")

    # ------------------------------------------------------------
    # 🔹 Load models
    # ------------------------------------------------------------
    model1 = load_model(model_path_train,
                        custom_objects={'psnr': psnr, 'ssim': ssim,
                                        'mse': mse, 'composite_loss': composite_loss},
                        compile=False)
    # model2 = load_model(model_path_retrained,
    #                     custom_objects={'psnr': psnr, 'ssim': ssim,
    #                                     'mse': mse, 'composite_loss': composite_loss},
    #                     compile=False)
    print("✅ Both models loaded successfully.")

    results = []

    # ------------------------------------------------------------
    # 🔹 Evaluate each subject
    # ------------------------------------------------------------
    for i in range(len(X_test)):
        print(f"\n🧠 Evaluating subject {i+1}/{len(X_test)} ...")
        lf = X_test[i]
        hf = y_test[i]

        # ---- Stage 1 Prediction ----
        pred1 = predict_volume(model1, lf, patch_size=patch_size, overlap=overlap)

        # # ---- Stage 2 Refinement ----
        # pred2 = predict_volume(model2, pred1, patch_size=patch_size, overlap=overlap)

        # ---- Compute Metrics ----
        psnr1 = psnr(tf.convert_to_tensor(hf[np.newaxis, ..., np.newaxis]),
                     tf.convert_to_tensor(pred1[np.newaxis, ..., np.newaxis])).numpy()
        ssim1 = ssim(tf.convert_to_tensor(hf[np.newaxis, ..., np.newaxis]),
                     tf.convert_to_tensor(pred1[np.newaxis, ..., np.newaxis])).numpy()

        # psnr2 = psnr(tf.convert_to_tensor(hf[np.newaxis, ..., np.newaxis]),
        #              tf.convert_to_tensor(pred2[np.newaxis, ..., np.newaxis])).numpy()
        # ssim2 = ssim(tf.convert_to_tensor(hf[np.newaxis, ..., np.newaxis]),
        #              tf.convert_to_tensor(pred2[np.newaxis, ..., np.newaxis])).numpy()

        print(f"📈 Stage1 → PSNR: {psnr1:.3f}, SSIM: {ssim1:.4f}")
        # print(f"📈 Stage2 → PSNR: {psnr2:.3f}, SSIM: {ssim2:.4f}")

        # # ---- Visualization ----
        # visualize_and_save_results(model_name, lf, hf, pred1, pred2,
        #                   slices=visualize_slices, title_prefix=f"Subject_day5 {i}")

        # ---- Collect results ----
        results.append({
            'subject': i,
            'psnr_stage1': psnr1,
            'ssim_stage1': ssim1,
            # 'psnr_stage2': psnr2,
            # 'ssim_stage2': ssim2
        })

    print("\n✅ Evaluation complete.")
    return results, pred1, model1

# ...existing code...

def visualize_volume_all_slices(
    vol,
    name="volume",
    output_dir="outputs",
    axis=2,
    cols=8,
    cmap="gray",
    save_png=False,
    dpi=200,
    vmin=None,
    vmax=None
):
    """
    Display all slices of a single 3D volume in a grid.

    Args:
        vol: 3D volume (H,W,D) or (1,H,W,D) or (H,W,D,1) or torch/tf tensor.
        name: figure title (and filename if save_png=True)
        output_dir: where to save PNG if enabled
        axis: slice axis (0,1,2). Default 2 => vol[:,:,k]
        cols: number of columns in the grid
        cmap: matplotlib colormap
        save_png: if True, saves a PNG grid to output_dir
        dpi: PNG dpi
        vmin/vmax: fixed display range for imshow (useful for consistent contrast)
    """
    _prepare_output_dir(output_dir)

    v = _to_numpy(vol)
    v = np.squeeze(v)

    if v.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeeze, got shape {v.shape}")

    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

    num_slices = v.shape[axis]
    rows = int(np.ceil(num_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    fig.suptitle(f"{name} (axis={axis}, slices={num_slices})", fontsize=16)

    # normalize axes to 2D array
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes[:, None]

    axes_flat = axes.ravel()

    for i in range(rows * cols):
        ax = axes_flat[i]
        ax.axis("off")

        if i >= num_slices:
            continue

        if axis == 0:
            img = v[i, :, :]
        elif axis == 1:
            img = v[:, i, :]
        else:
            img = v[:, :, i]

        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{i}", fontsize=8)

    plt.tight_layout()
    plt.show()

    if save_png:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{name}_all_slices_axis{axis}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved PNG: {out_path}")

    return fig

# Example paths (use your config_lf paths)
path_A = config_lf.path_lf_t2w
genA = DomainAGenerator(path_A)

# Fetch one batch from Domain A
for volA, ctxA, save_file in genA:
    print("Domain A volume shape:", volA.shape)
    print("Domain A context:", ctxA.shape)
    print("Domain A context values:", ctxA)
    print("Domain A save file:", save_file)
    break

# Path for the second stage model (SRR or enhancement)
model_name = 'residual_srr_unet_l2_ssim_edge'
# folder_path = "niv_results/outputs_src_simulated/Output_patch_noise"
folder_path = "niv_results/outputs_src_simulated_context/enhancement"

# path for the domain adaptation CycleGAN model
model_path_da = "niv_results/outputs_src_simulated_context/cyclegan_lfmri20t2w_2_lfsimulated_context_1000_t2w_2"

# Path for saving the generated volumes from domain adaptation step
# Path for saving the generated volumes from domain adaptation step
output_dir_lf ='niv_raw_data/Nipah_IRF_data/data_niv/Evaluator_data/VolA'
output_dir_denoise ='niv_raw_data/Nipah_IRF_data/data_niv/Evaluator_data/CycleGAN'
output_dir_enhance ='niv_raw_data/Nipah_IRF_data/data_niv/Evaluator_data/Enhancement'
output_dir = output_dir_denoise

# Resume from latest checkpoints if available

model_files = {
    'g_A2B': os.path.join(model_path_da, 'g_AtoB_000800.keras'),  # 400 for other one; 700; 800, (900 better) but eyes problem for second
    'g_B2A': os.path.join(model_path_da, 'g_BtoA_000500.keras'),
    'd_A': os.path.join(model_path_da, 'd_A_000500.keras'),
    'd_B': os.path.join(model_path_da, 'd_B_000500.keras')
}

if all(os.path.exists(f) for f in model_files.values()):
    print(">>> Resuming from latest checkpoints...")
    g_model_AtoB = load_model(model_files['g_A2B'], compile=False)
    g_model_BtoA = load_model(model_files['g_B2A'], compile=False)
    d_model_A = load_model(model_files['d_A'], compile=False)
    d_model_B = load_model(model_files['d_B'], compile=False)
    d_model_A.compile(loss=DISC_LOSS, optimizer=Adam(learning_rate=DISC_LEARNING_RATE, beta_1=DISC_BETA_1), loss_weights=DISC_LOSS_WEIGHTS)
    d_model_B.compile(loss=DISC_LOSS, optimizer=Adam(learning_rate=DISC_LEARNING_RATE, beta_1=DISC_BETA_1), loss_weights=DISC_LOSS_WEIGHTS)

for volA, ctxA, save_file in genA:

    print("Domain A volume shape:", volA.shape)
    print("Domain A context:", ctxA.shape)
    print("Domain A context values:", ctxA)
    # im = im.astype(np.float32) 
    volA = np.expand_dims(volA, axis=0)  # (1, H, W, D)
    print("Final LF input shape:", volA.shape)
    
    # break
    # Domain adaptation and get results of full volume prediction of low-field MRI then denoise and enhancement.
    # generates one full volume from genA translated by g_model_AtoB

    # Print min and max range of volA before feeding into generator
    print(f"Input volume range before generator: min={volA.min()}, max={volA.max()}")
    real_vol, fake_vol, ctx = evaluate_one_subject_volume(
        g_model_AtoB, volA, ctxA,
        out_path=os.path.join(output_dir, "AtoB_generated_subject.nii.gz"),
        batch_size=1
    )

    # add o asix to fake_vol to make it (1,H,W,D) for evaluation
    fake_vol_1 = np.expand_dims(fake_vol, axis=0)
    #check in max and min of fake_vol_1 and if max > 1 then normalize in range 0 to 1

    if fake_vol_1.max() > 1:
        fake_vol_1 = (fake_vol_1 / np.max(fake_vol_1) - 0.5) * 2
    
    # print min and max of fake_vol_1 after normalization
    print(f"Generated volume range after normalization: min={fake_vol_1.min()}, max={fake_vol_1.max()}")

    # get domain adopted and perform further steps
    results, pred1, model1 = evaluate_model(
        folder_path=folder_path,
        model_name=model_name,
        X_test=fake_vol_1,
        y_test=fake_vol_1,
        patch_size=(64, 64, 32),
        overlap=0.5,
        visualize_slices=[15]
    )

    print("Evaluation Results:after Stage 2 Refinement")
    
    volA = np.squeeze(volA)
    visualize_comparison(volA, fake_vol, pred1, name=save_file, output_dir=output_dir_denoise)
    visualize_comparison(fake_vol, volA, pred1, name=save_file, output_dir=output_dir_lf)
    visualize_comparison(volA, pred1, fake_vol, name=save_file, output_dir=output_dir_enhance)
    # print("Evaluation Results:after Stage 2 Refinement")
    # visualize_volume_all_slices(volA, name="volA", axis=2, cols=6)
    # visualize_volume_all_slices(fake_vol, name="pred1", axis=2, cols=6, vmin=-1, vmax=1)
    # visualize_volume_all_slices(pred1, name="pred1", axis=2, cols=6, vmin=-1, vmax=1)
    