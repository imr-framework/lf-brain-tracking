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
from src_simulated.cyclegan_models.config import config
# import losses from losses.py
from src_simulated.cyclegan_models.losses_cyclegan import *

# models.py load all models
from src_simulated.cyclegan_models.models import *
from src_simulated.cyclegan_models.train_utils import *

EPOCHS = config.EPOCHS

# test mode
TEST = config.TEST
SLICES_TEST = config.SLICES_TEST

#parameters for descriminator
DISC_LOSS = config.DISC_LOSS
DISC_LEARNING_RATE = config.DISC_LEARNING_RATE
DISC_BETA_1 = config.DISC_BETA_1
DISC_LOSS_WEIGHTS = config.DISC_LOSS_WEIGHTS

#parameters for generator
GEN_LOSS_1 = config.GEN_LOSS_1
GEN_LOSS_2 = config.GEN_LOSS_2
GEN_LOSS_3 = config.GEN_LOSS_3
GEN_LOSS_4 = config.GEN_LOSS_4
GEN_LEARNING_RATE = config.GEN_LEARNING_RATE
GEN_BETA_1 = config.GEN_BETA_1
GEN_LOSS_WEIGHTS = config.GEN_LOSS_WEIGHTS

#train parameters
INITIAL_LR = config.INITIAL_LR
N_ITER = config.N_ITER
N_ITER_DECAY = config.N_ITER_DECAY

# Output directories
OUTPUT_DIR = config.OUTPUT_DIR
# Visualization parameters
VISUALIZE = config.VISUALIZE  # Whether to visualize test examples during training

# model evaluation loading
MODEL_EVAL_PATH = config.MODEL_EVAL_PATH
# all model names to load using MODEL_EVAL_PATH
MODEL_NAME_D_A = config.MODEL_NAME_D_A
MODEL_NAME_D_B = config.MODEL_NAME_D_B
MODEL_NAME_G_A2B = config.MODEL_NAME_G_A2B
MODEL_NAME_G_B2A = config.MODEL_NAME_G_B2A

# output_path_test
OUTPUT_PATH_TEST = config.OUTPUT_PATH_TEST

# Read dataset

data_folder = "Data/Nipah IRF data/IRF_3T_NIFTI"
subjects = ["26184", "30366","34507", "35547", "59877","59233"]
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


def load_nii_volumes(path, target_spacing=(1,1,2), add_channel=False, 
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
        nii_files = nii_files[-5:]

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
data_path = "Data/Nipah_IRF_data/IRF_3T_NIFTI"
substring_filter = "T1_n100__00001"

dataA_all = load_nii_volumes(
    path=data_path,
    target_spacing=(1,1,2),
    add_channel=False,
    target_h=140,
    target_w=140,
    target_d=35,
    substring=substring_filter,
    rotate=True,
    test=True,
    save_path="Data/Nipah IRF data/IRF_3T_t1-t2/T1"
)

print("Loaded volumes shape:", dataA_all.shape)


from sklearn.utils import resample

# take first 40 samples from dataA_all
dataA = dataA_all[:40]
print('Loaded dataA: ', dataA.shape)

dataA = dataA[:, :, :, :-10]

# display range , min and max
print("DataA range: ", np.min(dataA), np.max(dataA))

# Example usage
substring_filter = "T2_n100"

dataB_all = load_nii_volumes(
    path=data_path,
    target_spacing=(1,1,2),
    add_channel=False,
    target_h=140,
    target_w=140,
    target_d=35,
    substring=substring_filter,
    rotate=True,
    test=True,
    save_path="Data/Nipah IRF data/IRF_3T_t1-t2/T2"
)

print("Loaded volumes shape:", dataB_all.shape)

from sklearn.utils import resample

# take first 40 samples from dataB_all
dataB = dataB_all[:40]
print('Loaded dataB: ', dataB.shape)

# Load Data

dataB = dataB[:, :, :, :-10]

# display range , min and max
print("DataB range: ", np.min(dataB), np.max(dataB))

data = [dataA, dataB]

for i in range(len(data)):
    # Rotate 90° counter-clockwise in-plane (H x W)
    data[i] = np.rot90(data[i], k=1, axes=(0, 1))

# Now data contains the rotated volumes
print("Rotated volumes shapes:", [d.shape for d in data])

print('Loaded', data[0].shape, data[1].shape)

# ----------------------------
# Convert 3D to slices
# ----------------------------
A_slices = []
for i in range(dataA.shape[0]):          # number of volumes
    for z in range(dataA.shape[3]):      # number of slices
        slice_2d = dataA[i, :, :, z]
        # slice_2d = cv2.resize(slice_2d, (140, 140), interpolation=cv2.INTER_LINEAR)
        A_slices.append(slice_2d)

A_2D = np.array(A_slices)
print("A_2D:", A_2D.shape)   # expected → (40*35, 256, 256)

# crop to 128, 128 of 140, 140 images
A_2D_cropped = []
for i in range(A_2D.shape[0]):
    slice_2d = A_2D[i]
    start_h = (140 - 128) // 2
    start_w = (140 - 128) // 2
    slice_cropped = slice_2d[start_h:start_h+128, start_w:start_w+128]
    A_2D_cropped.append(slice_cropped)    
A_2D = np.array(A_2D_cropped)
print("A_2D cropped:", A_2D.shape)   # expected → (40*35, 128, 128)

# ------------------------------
# Convert Domain B
# ------------------------------

B_slices = []
for i in range(dataB.shape[0]):
    for z in range(dataB.shape[3]):
        slice_2d = dataB[i, :, :, z]
        # slice_2d = cv2.resize(slice_2d, (140, 140), interpolation=cv2.INTER_LINEAR)
        B_slices.append(slice_2d)

B_2D = np.array(B_slices)
print("B_2D:", B_2D.shape)   # expected → (40*35, 256, 256)

# crop to 128, 128 of 140, 140 images
B_2D_cropped = []
for i in range(B_2D.shape[0]):
    slice_2d = B_2D[i]
    start_h = (140 - 128) // 2
    start_w = (140 - 128) // 2
    slice_cropped = slice_2d[start_h:start_h+128, start_w:start_w+128]
    B_2D_cropped.append(slice_cropped)
B_2D = np.array(B_2D_cropped)
print("B_2D cropped:", B_2D.shape)   # expected → (40*35, 128, 128)

# Suppose A_2D.shape = (N, 256, 256)
A_2D = A_2D[..., np.newaxis]   # (N, 256, 256, 1)
# A_2D = np.repeat(A_2D, 3, axis=-1)  # (N, 256, 256, 3)

B_2D = B_2D[..., np.newaxis]
# B_2D = np.repeat(B_2D, 3, axis=-1)  # (N, 256, 256, 3)
print(A_2D.shape, B_2D.shape)

# print shape of each
print("A_2D shape:", A_2D.shape)
print("B_2D shape:", B_2D.shape)

if SLICES_TEST:
    min_samples = min(5, 5)
    A_2D = A_2D[:min_samples]
    B_2D = B_2D[:min_samples]
    # print shape of each
    print("A_2D shape for test:", A_2D.shape)
    print("B_2D shape for test:", B_2D.shape)

data = [B_2D, A_2D]

#print datatype of each
print("A_2D dtype:", A_2D.dtype)
print("B_2D dtype:", B_2D.dtype)

dataset = data

if VISUALIZE:
    inspect_domains(dataset[0], dataset[1], n_samples=5)

# from cycleGAN_model import define_generator, define_discriminator, define_composite_model, train
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
print(image_shape)

# print min and max of dataset sample points

print("Dataset A min:", np.min(dataset[0][0]), "max:", np.max(dataset[0][0]))
print("Dataset B min:", np.min(dataset[1][0]), "max:", np.max(dataset[1][0]))

def show_plot_domains(A_real, A_gen, A_rec, B_real, B_gen, B_rec, save_path=None):

    def get_middle_slice(vol):
        if vol.ndim == 5:       # (1,H,W,D,C)
            vol = vol[0, ..., vol.shape[3] // 2, 0]
        elif vol.ndim == 4:     # (H,W,D,C)
            vol = vol[..., vol.shape[2] // 2, 0]
        elif vol.ndim == 3:
            vol = vol[..., vol.shape[2] // 2] if vol.shape[2] > 1 else vol[..., 0]

        # EXACT same scaling as summarize_performance
        vol = (vol + 1.0) / 2.0
        return vol

    A_real_slice = get_middle_slice(A_real[0])
    A_gen_slice  = get_middle_slice(A_gen[0])
    A_rec_slice  = get_middle_slice(A_rec[0])

    B_real_slice = get_middle_slice(B_real[0])
    B_gen_slice  = get_middle_slice(B_gen[0])
    B_rec_slice  = get_middle_slice(B_rec[0])

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    col_titles = ['Input x', 'Output G(x)', 'Reconstruction F(G(x))']
    row_labels = ['A Domain', 'B Domain']

    for row_idx, row_imgs in enumerate([
        [A_real_slice, A_gen_slice, A_rec_slice],
        [B_real_slice, B_gen_slice, B_rec_slice]
    ]):
        for col_idx, img in enumerate(row_imgs):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap='gray', aspect='auto')
            ax.axis('off')

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=12, color='black')

            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx],
                              rotation=90,
                              fontsize=12,
                              labelpad=10,
                              color='black')

    plt.subplots_adjust(wspace=0, hspace=0)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)

    plt.show()

# load the models
g_model_AtoB = load_model(MODEL_NAME_G_A2B, custom_objects={'InstanceNormalization': InstanceNormalization})
g_model_BtoA = load_model(MODEL_NAME_G_B2A, custom_objects={'InstanceNormalization': InstanceNormalization})

trainA, trainB = dataset

summarize_performance(6, g_model_AtoB, g_model_BtoA, trainA, trainB, 'AtoB', 5, OUTPUT_PATH_TEST)
summarize_performance(6, g_model_BtoA, g_model_AtoB, trainB, trainA, 'BtoA', 5, OUTPUT_PATH_TEST)

# # generate images
A_generated = g_model_BtoA.predict(trainB)
B_generated = g_model_AtoB.predict(trainA)

# print shapes
print("A_generated shape:", A_generated.shape)
print("B_generated shape:", B_generated.shape)

# #Convert to range [-1, 1] to 0,1
# # A_generated = (A_generated + 1) / 2
# # B_generated = (B_generated + 1) / 2

# #Display A_generated slices
visualize_slices(A_generated[0, :, :, :])
visualize_slices(B_generated[0, :, :, :])

# # check min and max of generated images
# print("A_generated range:", np.min(A_generated), np.max(A_generated))
# print("B_generated range:", np.min(B_generated), np.max(B_generated))

# # reconstruct images
# A_reconstructed = g_model_AtoB.predict(A_generated)
# B_reconstructed = g_model_BtoA.predict(B_generated)

# # print shapes
# print("A_reconstructed shape:", A_reconstructed.shape)
# print("B_reconstructed shape:", B_reconstructed.shape)

# # plot all results
# # print("A domain:")
# # show_plot(A_real, A_generated, A_reconstructed)
# # print("B domain:")
# # show_plot(B_real, B_generated, B_reconstructed)

# # convert all to float32, A_generated, A_reconstructed
# A_real = A_real.astype(np.float32)
# B_real = B_real.astype(np.float32)
# A_generated = A_generated.astype(np.float32)
# B_generated = B_generated.astype(np.float32)
# A_reconstructed = A_reconstructed.astype(np.float32)
# B_reconstructed = B_reconstructed.astype(np.float32)


# show_plot_domains(B_real, A_generated, A_reconstructed,
#                   A_real, B_generated, B_reconstructed,
#                   save_path=None)

# # ##########################
# # Load a single custom image
# test_image = load_img('monet2photo/sunset256.jpg')
# test_image = img_to_array(test_image)
# test_image_input = np.array([test_image])  # Convert single image to a batch.
# test_image_input = (test_image_input - 127.5) / 127.5
# print("Test image shape:", test_image_input.shape)
# # plot B->A->B (Photo to Monet to Photo)
# monet_generated  = g_model_BtoA.predict(test_image_input)
# photo_reconstructed = g_model_AtoB.predict(monet_generated)
# show_plot(test_image_input, monet_generated, photo_reconstructed)