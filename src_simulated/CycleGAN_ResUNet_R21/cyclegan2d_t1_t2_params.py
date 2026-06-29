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

# ----------------------------------------------
# 3D Volume Generators with Context (T1/T2)
# ----------------------------------------------
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

class VolumeGeneratorWithContext:
    """
    Infinite generator for 3D MRI volumes with context vector.
    """

    def __init__(self, path,
                 substring=None,
                 target_spacing=(1,1,2),
                 target_h=140,
                 target_w=140,
                 target_d=35,
                 crop_size=128,
                 rotate=True,
                 add_channel=True,
                 test=False,
                 context_type='T1'):

        self.path = path
        self.substring = substring
        self.target_spacing = target_spacing
        self.target_h = target_h
        self.target_w = target_w
        self.target_d = target_d
        self.crop_size = crop_size
        self.rotate = rotate
        self.add_channel = add_channel
        self.context_type = context_type

        # Collect files
        self.files = []
        for dirpath, _, filenames in os.walk(path):
            for fname in sorted(filenames):
                if fname.endswith((".nii", ".nii.gz")) and \
                   (substring is None or substring in fname):
                    self.files.append(os.path.join(dirpath, fname))

        if test:
            self.files = self.files[:5]

        if not self.files:
            raise ValueError(f"No files found in {path}")

        print(f"[INFO] Found {len(self.files)} volumes")

    # -------------------------------------------------------
    # Context vector
    # -------------------------------------------------------
    def _default_context(self):
        """
        Realistic 3T MRI acquisition parameters
        [TE, TR, FlipAngle, Bandwidth, ETL, FieldStrength]
        """

        if self.context_type == 'T1':
            return np.array([
                2.5,     # TE (ms)
                1900,    # TR (ms)
                9,       # Flip angle
                240,     # Bandwidth
                1,       # ETL
                3.0      # Field strength
            ], dtype=np.float32)

        elif self.context_type == 'T2':
            return np.array([
                90,      # TE (ms)
                5000,    # TR (ms)
                120,     # Flip angle
                240,     # Bandwidth
                16,      # ETL
                3.0      # Field strength
            ], dtype=np.float32)

        else:
            raise ValueError("context_type must be 'T1' or 'T2'")

    # -------------------------------------------------------
    # Depth fix (pad or crop)
    # -------------------------------------------------------
    def _fix_depth(self, vol):
        h, w, d = vol.shape
        out = np.zeros((h, w, self.target_d), dtype=vol.dtype)

        ds = max((self.target_d - d) // 2, 0)
        de = ds + min(d, self.target_d)

        d0 = max((d - self.target_d) // 2, 0)
        out[:, :, ds:de] = vol[:, :, d0:d0 + (de - ds)]

        return out

    # -------------------------------------------------------
    # Normalize [-1,1]
    # -------------------------------------------------------
    def _normalize(self, vol):
        vol = vol / (np.max(vol) + 1e-8)
        return (vol - 0.5) * 2

    # -------------------------------------------------------
    # Load single volume
    # -------------------------------------------------------
    def _load_volume(self, fpath):
        nii = nib.load(fpath)
        vol = np.abs(nii.get_fdata().astype(np.float32))

        fname = os.path.basename(fpath)

        # --- Resample ---
        spacing = nii.header.get_zooms()[:3]
        zoom_factors = (
            spacing[0] / self.target_spacing[0],
            spacing[1] / self.target_spacing[1],
            spacing[2] / self.target_spacing[2]
        )

        vol = zoom(vol, zoom_factors, order=1)
        vol = np.ascontiguousarray(vol)

        h, w, d = vol.shape
        print(f"[INFO] {fname} → {vol.shape}")

        # --- Size check ---
        if h != self.target_h or w != self.target_w:
            print(f"[SKIP] {fname} (size mismatch)")
            return None

        # --- Fix depth ---
        vol = self._fix_depth(vol)

        # --- Center crop ---
        sh = (h - self.crop_size) // 2
        sw = (w - self.crop_size) // 2
        vol = vol[sh:sh+self.crop_size, sw:sw+self.crop_size, :]

        # --- Normalize ---
        vol = self._normalize(vol)

        # --- Rotate ---
        if self.rotate:
            vol = np.rot90(vol, axes=(0,1))

        # --- Channel ---
        if self.add_channel:
            vol = vol[..., np.newaxis]

        return vol

    # -------------------------------------------------------
    # Infinite generator
    # -------------------------------------------------------
    def generator(self):
        context = self._default_context()

        while True:
            for fpath in np.random.permutation(self.files):
                vol = self._load_volume(fpath)
                if vol is None:
                    continue

                yield vol, context

    # -------------------------------------------------------
    # Load all volumes (for evaluation)
    # -------------------------------------------------------
    def load_all(self):
        volumes = []
        contexts = []

        context = self._default_context()

        for fpath in self.files:
            vol = self._load_volume(fpath)
            if vol is None:
                continue

            volumes.append(vol)
            contexts.append(context)

        if not volumes:
            raise ValueError("No valid volumes loaded.")

        return np.stack(volumes), np.stack(contexts)

# example test

# Example: Call the VolumeGeneratorWithContext and print shapes

# Set up paths (replace with your actual data path)
test_path = "niv_raw_data/Nipah_IRF_data/data_niv/IRF_3T_t1-t2"
substring_filter = "T1_n100__00001"

# Instantiate for T1 and T2 context types
vgc_T1 = VolumeGeneratorWithContext(
    path=test_path,
    substring=substring_filter,
    target_spacing=(1,1,2),
    target_h=140,
    target_w=140,
    target_d=35,
    crop_size=128,
    rotate=True,
    add_channel=True,
    test=True,
    context_type='T1'
)
substring_filter = "T2_n100"
vgc_T2 = VolumeGeneratorWithContext(
    path=test_path,
    substring=substring_filter,
    target_spacing=(1,1,2),
    target_h=140,
    target_w=140,
    target_d=35,
    crop_size=128,
    rotate=True,
    add_channel=True,
    test=True,
    context_type='T2'
)

# Get one sample from each generator and print shapes
vol_T1, ctx_T1 = next(vgc_T1.generator())
print("VolumeGeneratorWithContext (T1) volume shape:", vol_T1.shape)
print("VolumeGeneratorWithContext (T1) context shape:", ctx_T1.shape)
print("Context values (T1):", ctx_T1)

vol_T2, ctx_T2 = next(vgc_T2.generator())
print("VolumeGeneratorWithContext (T2) volume shape:", vol_T2.shape)
print("VolumeGeneratorWithContext (T2) context shape:", ctx_T2.shape)
print("Context values (T2):", ctx_T2)

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

class DomainAGenerator:
    def __init__(self, path, batch_size=1, target_h=128, target_w=128, target_d=35, 
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
        
        # Normalize [-1,1]
        vol = (vol / np.max(vol) - 0.5) * 2 if np.max(vol) > 0 else vol

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
            yield vol, ctx[0]

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler

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

        # Keep vol only 0 to 25 slices in a volume  depth dimension
        vol = vol[..., :25]

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

# Example paths (use your config_lf paths)
path_A = config_lf.path_lf
genA = DomainAGenerator(path_A)

# Fetch one batch from Domain A
for volA, ctxA in genA:
    print("Domain A volume shape:", volA.shape)
    print("Domain A context:", ctxA.shape)
    print("Domain A context values:", ctxA)
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
vol, ctx = genB[0]
print("Volume shape:", vol.shape)  # H x W x D
print("Context shape:", ctx.shape) # (9,)
print("Context values:", ctx)

# Example: using generator for training
train_gen = genB.generator()
# vol1, ctx1 = next(train_gen)
# vol2, ctx2 = next(train_gen)
# print("Batch example:", vol1.shape, ctx1.shape)
# print("Batch example:", vol2.shape, ctx2.shape)

steps_per_epoch = min(len(genA.files), len(genB.files))  # number of volumes
print("Steps per epoch:", steps_per_epoch)
print("Total volumes in Domain A:", len(genA.files))
print("Total volumes in Domain B:", len(genB.files))

image_shape = [128,128,1]
print(image_shape)

# # print min and max of dataset sample points

# print("Dataset A min:", np.min(dataset[0][0]), "max:", np.max(dataset[0][0]))
# print("Dataset B min:", np.min(dataset[1][0]), "max:", np.max(dataset[1][0]))
# Steps per epoch

# -----------------------------
def infinite_volume_context_generator(domain_generator):
    """
    Infinite generator yielding (volume, context) from a domain generator.
    Reshuffles after each epoch.
    """
    while True:
        # Use random permutation for shuffling
        idxs = np.random.permutation(len(domain_generator.files))
        for idx in idxs:
            try:
                # DomainAGenerator: __iter__ yields (vol, ctx)
                # DomainBGenerator: use generator() method
                if hasattr(domain_generator, '__iter__'):
                    # For DomainAGenerator
                    for vol, ctx in domain_generator:
                        yield vol, ctx
                    break  # after one full epoch, reshuffle
                elif hasattr(domain_generator, 'generator'):
                    # For DomainBGenerator
                    for vol, ctx in domain_generator.generator():
                        yield vol, ctx
                    break
            except ValueError:
                continue  # skip invalid volumes

# Decide which is small/large
size_A = len(genA.files)  # 15
size_B = len(genB.files)  # 50

if size_A <= size_B:
    gen_small = infinite_volume_context_generator(genA)
    gen_large = infinite_volume_context_generator(genB)
    small_domain = 'A'
else:
    gen_small = infinite_volume_context_generator(genB)
    gen_large = infinite_volume_context_generator(genA)
    small_domain = 'B'

# # visualize_volume_samples(gen_small, gen_large, num_samples=5)
# if VISUALIZE:
#     visualize_single_volume(gen_small, slice_axis=2)
#     visualize_single_volume(gen_large, slice_axis=2)
#     visualize_single_volume(gen_small, slice_axis=2)
#     visualize_single_volume(gen_large, slice_axis=2)
#     visualize_single_volume(gen_small, slice_axis=2)
#     visualize_single_volume(gen_large, slice_axis=2)
#     visualize_single_volume(gen_small, slice_axis=2)
#     visualize_single_volume(gen_large, slice_axis=2)
#     visualize_single_volume(gen_large, slice_axis=2)
#     visualize_single_volume(gen_small, slice_axis=2)
#     visualize_single_volume(gen_large, slice_axis=2)
#     visualize_single_volume(gen_small, slice_axis=2)
#     visualize_single_volume(gen_large, slice_axis=2)
#     visualize_single_volume(gen_small, slice_axis=2)
#     visualize_single_volume(gen_large, slice_axis=2)

g_model_AtoB = define_generator(image_shape, context_dim=5)
g_model_BtoA = define_generator(image_shape, context_dim=5)

# # UNet kind generator without skip connections
# g_model_AtoB = define_unet_generator(image_shape, context_dim=5, n_resnet=9)
# g_model_BtoA = define_unet_generator(image_shape, context_dim=5, n_resnet=9)

# # Generator with skip connections (U-Net like)
# g_model_AtoB = define_unet_skip_generator(image_shape, context_dim=5, n_resnet=6)
# g_model_BtoA = define_unet_skip_generator(image_shape, context_dim=5, n_resnet=6)

# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
d_model_A.compile(
        loss=DISC_LOSS,
        optimizer=Adam(
            learning_rate=DISC_LEARNING_RATE,
            beta_1=DISC_BETA_1
        ),
        loss_weights=DISC_LOSS_WEIGHTS
    )
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
d_model_B.compile(
        loss=DISC_LOSS,
        optimizer=Adam(
            learning_rate=DISC_LEARNING_RATE,
            beta_1=DISC_BETA_1
        ),
        loss_weights=DISC_LOSS_WEIGHTS
    )

# composite: A -> B -> [real/fake, A]
# c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape, context_dim=5)
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape, context_dim=5)

from datetime import datetime

start1 = datetime.now()

# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA,  gen_small, gen_large, steps_per_epoch, output_dir=OUTPUT_DIR, epochs=EPOCHS, n_slices=num_slices, initial_lr=INITIAL_LR, n_iter=N_ITER, n_iter_decay=N_ITER_DECAY, small_domain=small_domain)

stop1 = datetime.now()
#Execution time of the model
execution_time = stop1-start1
print("Execution time is: ", execution_time)