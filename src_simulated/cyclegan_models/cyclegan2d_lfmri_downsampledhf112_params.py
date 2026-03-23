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

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, LeakyReLU,
    Activation, Concatenate, Add, UpSampling2D, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

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
from src_simulated.cyclegan_models.train_utils_params import *

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

# dataset path

# Example usage
# Can load true acquired LF volumes
path_lf = config_lf.path_lf

dataA_all = load_nii_volumes(path_lf, target_spacing=(1,1,2),TARGET_H = 128, TARGET_W = 128, TARGET_D = 35, add_channel=False)
print('Loaded dataA: ', dataA_all.shape)

from sklearn.utils import resample
#To get a subset of all images, for faster training during demonstration
dataA = resample(dataA_all,
                 replace=False,
                 n_samples=15,
                 random_state=42)

dataA = dataA[:, :, :, 3:-2]   # new depth = 30

# dataA = np.array([cv2.cvtColor(dataA[i], cv2.COLOR_RGB2GRAY) for i in range(len(dataA))])
visualize_slices(dataA[1, :, :, :])

# display range , min and max
print("DataA range: ", np.min(dataA), np.max(dataA))

# Load Data
path_raw_t2 = "niv_raw_data/Nipah_IRF_data/data_niv/IRF_3T_NIFTI"
substring_filter = "T2_n100"

def load_nii_volumes_downsampled(path, target_spacing=(1,1,2), add_channel=False, 
                     target_h=140, target_w=140, target_d=35, 
                     substring=None, rotate=False, test=False, save_path=None):
    
    """
    Load NIfTI volumes, optionally filter by substring in filename,
    resample to target voxel spacing, accept only target in-plane resolution,
    fix depth, normalize, optionally add channel dimension.
    
    Parameters:
    - test : bool : if True, load only 5 volumes to save time
    """
    volumes = []

    # Recursively find NIfTI files
    nii_files = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        filenames.sort()
        for fname in filenames:
            if fname.endswith((".nii", ".nii.gz")) and (substring is None or substring in fname):
                nii_files.append(os.path.join(dirpath, fname))

    nii_files.sort(key=lambda x: (os.path.dirname(x), os.path.basename(x)))
    import shutil
    # copy all nii.gz files to save folder if save_path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        for i, fpath in enumerate(nii_files, 1):
            fname = os.path.basename(fpath)
            # change substring in filename with image {i}
            fname = fname.replace(substring, f"image_{i}")
            dest_path = os.path.join(save_path, fname)
            if not os.path.exists(dest_path):
                shutil.copy2(fpath, dest_path)
        print(f"[INFO] Copied {len(nii_files)} files to {save_path}")

    # Limit number of files if test mode is on
    if test:
        nii_files = nii_files[:5]

    for fpath in nii_files:
        nii = nib.load(fpath)
        vol = np.abs(nii.get_fdata().astype(np.float32))  # fix negatives
        header = nii.header
        current_spacing = header.get_zooms()[:3]  # get voxel size

        # Resample to target voxel size
        zoom_factors = (
            current_spacing[0] / target_spacing[0],
            current_spacing[1] / target_spacing[1],
            current_spacing[2] / target_spacing[2]
        )
        vol_iso = zoom(vol, zoom_factors, order=1)
        vol_iso = np.ascontiguousarray(vol_iso)
        h, w, d = vol_iso.shape
        print(f"[INFO] {os.path.basename(fpath)} resampled → {vol_iso.shape} (target spacing {target_spacing})")

        # Check in-plane resolution
        if h != target_h or w != target_w:
            print(f"[SKIP] {os.path.basename(fpath)} skipped — incorrect in-plane size: {vol_iso.shape}")
            continue

        # Crop/pad depth
        vol_fixed = crop_or_pad_depth(vol_iso, target_d)

        # Normalize to [-1,1]
        vol_fixed = normalize_volume(vol_fixed)

        # Rotate 90° counter-clockwise in-plane (H x W)
        if rotate:
            vol_fixed = np.rot90(vol_fixed, k=3, axes=(0, 1))

        # Optional channel dimension
        if add_channel:
            vol_fixed = vol_fixed[..., None]  # (H,W,D,1)

        print(f"[LOAD] {os.path.basename(fpath)}: final shape {vol_fixed.shape}")
        volumes.append(vol_fixed)

        # Stop early if test mode
        if test and len(volumes) >= 5:
            break

    if not volumes:
        raise ValueError("No valid volumes loaded. Check dimensions, substring, and input path.")
    
    return np.stack(volumes, axis=0)

dataB_all = load_nii_volumes_downsampled(
    path=path_raw_t2,
    target_spacing=(1,1,2),
    add_channel=False,
    target_h=140,
    target_w=140,
    target_d=35,
    substring=substring_filter,
    rotate=True,
    test=TEST,
    save_path="Data/Nipah_IRF_data/IRF_3T_t1-t2/T2"
)

print("Loaded volumes shape:", dataB_all.shape)

from sklearn.utils import resample

# take first 40 samples from dataB_all
dataB = dataB_all[:40]
print('Loaded dataB: ', dataB.shape)

# display range , min and max
print("DataB range: ", np.min(dataB), np.max(dataB))

# Crop dataB to match dataA height and width if needed
dataB = dataB[:, :dataA.shape[1], :dataA.shape[2], :]

dataB = dataB[:, :, :, :-5]

visualize_slices(dataB[1, :, :, :])

# display range , min and max
print("DataB range: ", np.min(dataB), np.max(dataB))

# discard first three slices and last 2 slices of dataA and last five slices of dataB to have depth 30

# load image data
data =  [dataB, dataA]

print('Loaded', data[0].shape, data[1].shape)

# ----------------------------
# Convert Domain A → 2D (256x256)
# ----------------------------
A_slices = []
for i in range(dataA.shape[0]):          # number of volumes
    for z in range(dataA.shape[3]):      # number of slices
        slice_2d = dataA[i, :, :, z]
        # slice_2d = cv2.resize(slice_2d, (128, 128), interpolation=cv2.INTER_LINEAR)
        A_slices.append(slice_2d)

A_2D = np.array(A_slices)
print("A_2D:", A_2D.shape)   # expected → (40*35, 256, 256)

# ----------------------------
# Convert Domain B → 2D (256x256)
# ----------------------------
B_slices = []
for i in range(dataB.shape[0]):
    for z in range(dataB.shape[3]):
        slice_2d = dataB[i, :, :, z]
        # slice_2d = cv2.resize(slice_2d, (128, 128), interpolation=cv2.INTER_LINEAR)
        B_slices.append(slice_2d)

B_2D = np.array(B_slices)
print("B_2D:", B_2D.shape)   # expected → (40*35, 256, 256)

# Suppose A_2D.shape = (N, 256, 256)
A_2D = A_2D[..., np.newaxis]   # (N, 256, 256, 1)
# A_2D = np.repeat(A_2D, 3, axis=-1)  # (N, 256, 256, 3)

B_2D = B_2D[..., np.newaxis]
# B_2D = np.repeat(B_2D, 3, axis=-1)  # (N, 256, 256, 3)
print(A_2D.shape, B_2D.shape)


#print datatype of each
print("A_2D dtype:", A_2D.dtype)
print("B_2D dtype:", B_2D.dtype)

if SLICES_TEST:
    min_samples = min(5, 5)
    A_2D = A_2D[:min_samples]
    B_2D = B_2D[:min_samples]
    # print shape of each
    print("A_2D shape for test:", A_2D.shape)
    print("B_2D shape for test:", B_2D.shape)


# rotate A_2d all slices by 90 degrees clockwise for better alignment
A_2D = np.rot90(A_2D, k=1, axes=(1, 2))

data = [A_2D, B_2D]

#print datatype of each
print("A_2D dtype:", A_2D.dtype)
print("B_2D dtype:", B_2D.dtype)


# addign context to the print statements about min and max values

# ----------------------------
# CONTEXT CREATION (N, 9)
# ----------------------------

N_A = A_2D.shape[0]
N_B = B_2D.shape[0]

def create_context(N, field_strength, visit):
    TE = np.random.uniform(80, 120, N)
    TR = np.random.uniform(2000, 3000, N)
    bandwidth = np.random.uniform(150, 250, N)
    rxGain = np.random.uniform(20, 40, N)
    etLength = np.random.uniform(8, 16, N)
    dwellTime = np.random.uniform(5, 10, N)

    # ⭐ SNR (IMPORTANT)
    SNR = (field_strength * 10) + np.random.normal(0, 2, N)

    context = np.stack([
        TE, TR, bandwidth, rxGain,
        etLength, dwellTime,
        np.full(N, field_strength),
        np.full(N, visit),
        SNR
    ], axis=1)

    return context.astype(np.float32)

A_context = create_context(N_A, field_strength=0.05, visit=1)
B_context = create_context(N_B, field_strength=3.0, visit=1)

print("A_context:", A_context.shape)
print("B_context:", B_context.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
A_context = scaler.fit_transform(A_context)
B_context = scaler.transform(B_context)

dataset = [A_2D, B_2D, A_context, B_context]
# dataset = data

if VISUALIZE:
    inspect_domains(dataset[0], dataset[1], n_samples=5)

# from cycleGAN_model import define_generator, define_discriminator, define_composite_model, train
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
print(image_shape)

# print min and max of dataset sample points

print("Dataset A min:", np.min(dataset[0][0]), "max:", np.max(dataset[0][0]))
print("Dataset B min:", np.min(dataset[1][0]), "max:", np.max(dataset[1][0]))


# generator: A -> B
# g_model_AtoB = define_generator(image_shape)
# g_model_BtoA = define_generator(image_shape)

g_model_AtoB = define_generator(image_shape, context_dim=9)
g_model_BtoA = define_generator(image_shape, context_dim=9)

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

c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape, context_dim=9)
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape, context_dim=9)

from datetime import datetime
start1 = datetime.now()

# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, output_dir=OUTPUT_DIR, epochs=EPOCHS, initial_lr=INITIAL_LR, n_iter=N_ITER, n_iter_decay=N_ITER_DECAY)

stop1 = datetime.now()
#Execution time of the model
execution_time = stop1-start1
print("Execution time is: ", execution_time)