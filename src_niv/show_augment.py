import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./data_read_code')
from src_niv.prep_data import data_ops
from src_niv.read_lf5_data import process_subject
from src_niv.utils import srr_generator, display_pred, load_and_preprocess_hf, load_and_preprocess_lf, visualize_hf_slices,padding_LF, visualize_lf_slices,rotate_slices, visualize_resampled, resample_volume, visualize_planes,visualize_pair, normalize_volume
from src_niv.prep_lf import normalize, resize_mri_volume
from src_niv.zssr import  zero_shot_super_resolution, extract_brain, extract_lf_volumes
from src_niv.niv_srr_main import train
from src_niv.models.ResUNet import residual_srr_unet, residual_att_unet_3d
from src_niv.models.DenseUNet import build_dense_unet_3d
from src_niv.models.Inception import build_inception_unet_3d
from demo_read_data import read_lf_data
from src_niv.prep_lf import register_to_hf

import os
import cv2
import nibabel as nib
from nilearn import plotting
from nibabel.viewers import OrthoSlicer3D
import tensorflow as tf
import pydicom
import numpy as np
import matplotlib
matplotlib.use('tkagg')  # or 'Qt5Agg' depending on what's installed
import matplotlib.pyplot as plt
import ants
import nibabel as nib
from sklearn.feature_extraction import image
import os
from skimage.transform import resize  # Required for 3D resizing

# 59228
subjects1 = ['26184'] # '59877', '59175','59233', '59877', '35547'

# Mismatch due to high field shape

# Define the path to the IRF_3T folder (High Field Data)
nhp_base_path = './Data/IRF_3T'
day_idx = 2
visualize = False
visualize_pairs = False
padding = False
register2_hf = False
augmentation = True

# Training parameters
steps_per_epoch = 60
epochs = 1500
batch_size = 2

# Training data

lf_input_volume_combined = []
hf_input_volume_combined = []
hf_target_volume_combined = []

for subject in subjects1:
    for day_idx in [2]:  # Assuming 0 = Day 1, 1 = Day 2
        print(f"\n=============================== Processing subject: {subject}, Day: {day_idx + 1} ===============================")
        # ----- Load HF data -----
        print(f"\n=============================== HF_MRI data processing started .............")
        resampled_volume_hf_norm = load_and_preprocess_hf(subject, day_idx, visualize)
        # ----- Load LF data -----
        print(f"\n=============================== LF_MRI data processing started .............")
        resampled_volume_lf_be_norm = load_and_preprocess_lf(subject, day_idx, visualize)
        print("Resampled LF volume shape:", resampled_volume_lf_be_norm.shape)
        print("Resampled HF volume shape:", resampled_volume_hf_norm.shape)
        # Register LF to HF
        if register2_hf:
            resampled_volume_lf_be_norm = register_to_hf(resampled_volume_lf_be_norm, resampled_volume_hf_norm)
        # Padding
        if padding:
            resampled_volume_lf_be_norm = padding_LF(
                resampled_volume_lf_be_norm,
                resampled_volume_hf_norm,
                target_slices=64
            )
            print("After padding LF volume shape:", resampled_volume_lf_be_norm.shape)
        # Visualization (optional)
        if visualize_pairs:
            visualize_pair(
                resampled_volume_lf_be_norm,
                resampled_volume_hf_norm,
                slice_indices=list(range(31))
            )
        # ----- Final Volume Preparation -----
        lf_input_volume = resampled_volume_lf_be_norm.astype(np.float32)
        hf_input_volume = resampled_volume_hf_norm.astype(np.float32)
        hf_target_volume = resampled_volume_hf_norm.astype(np.float32)
        # Slice selection
        lf_input_volume = lf_input_volume[0:32, :, :]
        hf_input_volume = hf_input_volume[0:32, :, :]
        hf_target_volume = hf_target_volume[0:32, :, :]

        print("LF Input shape:", lf_input_volume.shape)
        print("HF Input shape:", hf_input_volume.shape)
        print("HF volume shape:", hf_target_volume.shape)
        # ----- Append to Combined Lists -----
        lf_input_volume_combined.append(lf_input_volume)
        hf_input_volume_combined.append(hf_input_volume)
        hf_target_volume_combined.append(hf_target_volume)

lf_input_volume_combined = np.stack(lf_input_volume_combined)
hf_input_volume_combined = np.stack(hf_input_volume_combined)
hf_target_volume_combined = np.stack(hf_target_volume_combined)

print("LF Input shape:", lf_input_volume_combined.shape)
print("HF Input shape:", hf_input_volume_combined.shape)
print("HF volume shape:", hf_target_volume_combined.shape)

# Validation data
print('-----------------------------\n\nLoading validation data .................................-----------------')
subjects_val = ['26184']
for subject_v in subjects_val:
    for day_idx in [1]:  # Assuming 0 = Day 1, 1 = Day 2
        print(f"\n=============================== Processing subject: {subject_v}, Day: {day_idx + 1} ===============================")
        # ----- Load HF data -----
        print(f"\n=============================== HF_MRI data processing started .............")
        resampled_volume_hf_norm = load_and_preprocess_hf(subject_v, day_idx, visualize)
        # ----- Load LF data -----
        print(f"\n=============================== LF_MRI data processing started .............")
        resampled_volume_lf_be_norm = load_and_preprocess_lf(subject_v, day_idx, visualize)
        print("Resampled LF volume shape:", resampled_volume_lf_be_norm.shape)
        print("Resampled HF volume shape:", resampled_volume_hf_norm.shape)
        
        # Register LF to HF
        if register2_hf:
            resampled_volume_lf_be_norm = register_to_hf(resampled_volume_lf_be_norm, resampled_volume_hf_norm)
        
        # Padding
        if padding:
            resampled_volume_lf_be_norm = padding_LF(
                resampled_volume_lf_be_norm,
                resampled_volume_hf_norm,
                target_slices=64
            )
            print("After padding LF volume shape:", resampled_volume_lf_be_norm.shape)
        
        # Visualization (optional)
        if visualize_pairs:
            visualize_pair(
                resampled_volume_lf_be_norm,
                resampled_volume_hf_norm,
                slice_indices=list(range(31))
            )
        
        # ----- Final Volume Preparation -----
        lf_input_volume_val = resampled_volume_lf_be_norm.astype(np.float32)
        hf_input_volume_val = resampled_volume_hf_norm.astype(np.float32)
        hf_target_volume_val = resampled_volume_hf_norm.astype(np.float32)
        
        # Slice selection
        lf_input_volume_val = lf_input_volume_val[0:32, :, :]
        hf_input_volume_val = hf_input_volume_val[0:32, :, :]
        hf_target_volume_val = hf_target_volume_val[0:32, :, :]
        print("LF Input shape:", lf_input_volume_val.shape)
        print("HF Input shape:", hf_input_volume_val.shape)
        print("HF volume shape:", hf_target_volume_val.shape)

import numpy as np
import tensorflow as tf

# --------------------------
# 2D augmentations
# --------------------------
def rotate2d(img, angle_deg):
    angle_rad = angle_deg * np.pi / 180
    frac = angle_rad / (2.0 * np.pi)
    rr = tf.keras.layers.RandomRotation(factor=(frac, frac), fill_mode="nearest")
    img = tf.expand_dims(img, axis=0)
    out = rr(img)
    return out[0].numpy()

def shear2d(img, shear_x=0.1, shear_y=0.1):
    img = tf.expand_dims(img, axis=0)
    layer = tf.keras.layers.RandomTranslation(height_factor=shear_y, width_factor=shear_x, fill_mode="nearest")
    out = layer(img)
    return out[0].numpy()

def add_gaussian_noise(img, sigma=0.01):
    noise = np.random.normal(0, sigma, img.shape)
    return img + noise

def adjust_intensity(img, factor_range=(0.9, 1.1)):
    factor = np.random.uniform(*factor_range)
    return img * factor

# --------------------------
# 3D SRR patch generator
# --------------------------
def srr_generator(lf_vol, hf_vol, batch_size=2, patch_z=32, patch_xy=128, augment=True, extra_slices=0, noise_sigma=0.02):
    """
    Generate 3D SRR patches for LF-MRI -> HF-MRI training.
    Outputs:
        batch_lf: (batch_size, Z, X, Y)
        batch_hf: (batch_size, Z, X, Y)
    """

    # print("inside _ generator ......................")
    if lf_vol.ndim == 3:
        lf_vols = [lf_vol]
        hf_vols = [hf_vol]
    elif lf_vol.ndim == 4:
        lf_vols = [lf_vol[i] for i in range(lf_vol.shape[0])]
        hf_vols = [hf_vol[i] for i in range(hf_vol.shape[0])]
    else:
        raise ValueError("lf_vol must be (Z,X,Y) or (A,Z,X,Y)")

    N = len(lf_vols)
    while True:
        batch_lf, batch_hf = [], []

        volume_indices = np.random.permutation(N)
        for idx_vol in volume_indices:
            lf_current = lf_vols[idx_vol]
            hf_current = hf_vols[idx_vol]

            Z, X, Y = lf_current.shape

            # Random start along Z
            start = np.random.randint(0, max(1, Z - patch_z + 1))
            lf_patch = lf_current[start:start+patch_z].copy()
            hf_patch = hf_current[start:start+patch_z].copy()

            # Resize XY to patch_xy
            lf_patch = tf.image.resize(lf_patch[..., np.newaxis], (patch_xy, patch_xy)).numpy()[..., 0]
            hf_patch = tf.image.resize(hf_patch[..., np.newaxis], (patch_xy, patch_xy)).numpy()[..., 0]

            # Stack LF + HF for augmentation
            vol_stack = np.stack([lf_patch, hf_patch], axis=0)  # (2, Z, XY, XY)

            # Extra augmented slices
            if augment and extra_slices > 0:
                augmented_slices = []
                for _ in range(extra_slices):
                    idx = np.random.randint(0, vol_stack.shape[1])
                    slice_aug = vol_stack[:, idx].copy()  # (2, XY, XY)

                    # Apply LF-specific augmentations
                    lf_slice = slice_aug[0]
                    if np.random.rand() > 0.5:
                        lf_slice = rotate2d(lf_slice, np.random.uniform(-15,15))
                    if np.random.rand() > 0.5:
                        lf_slice = shear2d(lf_slice, np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1))
                    if np.random.rand() > 0.5:
                        lf_slice = add_gaussian_noise(lf_slice, sigma=noise_sigma)
                    if np.random.rand() > 0.5:
                        lf_slice = adjust_intensity(lf_slice, factor_range=(0.85, 1.15))

                    slice_aug[0] = lf_slice
                    augmented_slices.append(slice_aug[:, None, ...])  # add Z dim

                vol_stack = np.concatenate([vol_stack] + augmented_slices, axis=1)

            # Randomly select patch_z slices if needed
            if vol_stack.shape[1] > patch_z:
                idxs = np.random.choice(vol_stack.shape[1], patch_z, replace=False)
                vol_stack = vol_stack[:, idxs]

            batch_lf.append(vol_stack[0])
            batch_hf.append(vol_stack[1])

            # Inside your generator, replace the yield line with:
            if len(batch_lf) == batch_size:
                batch_lf_out = np.stack(batch_lf, axis=0)[..., np.newaxis]  # (B, Z, X, Y, 1)
                batch_hf_out = np.stack(batch_hf, axis=0)[..., np.newaxis]  # (B, Z, X, Y, 1)
                yield batch_lf_out, batch_hf_out
                batch_lf, batch_hf = [], []

# gen = srr_generator(lf_input_volume, hf_target_volume, batch_size=batch_size, patch_z=32, augment=True, num_augmented_copies=6)
train_gen = srr_generator(lf_input_volume, hf_target_volume, batch_size=1, patch_z=32, patch_xy=128, augment=False, extra_slices=30, noise_sigma=0.02)
lf_input, hf_target = next(train_gen)
print(lf_input.shape)  # (2, 32, 128, 128, 1)
print(hf_target.shape)  # (2, 32, 128, 128, 1)


def display_batch_slices_side_by_side(lf_volume, hf_volume, num_slices=15):
    batch_size, slices, height, width, channels = lf_volume.shape
    for b in range(batch_size):
        plt.figure(figsize=(15, 6))
        # First row: LF
        for i in range(num_slices):
            plt.subplot(2, num_slices, i + 1)
            plt.imshow(lf_volume[b, i, :, :, 0], cmap='gray')
            plt.title(f"LF Batch {b}, Slice {i}")
            plt.axis('off')
        # Second row: HF
        for i in range(num_slices):
            plt.subplot(2, num_slices, num_slices + i + 1)
            plt.imshow(hf_volume[b, i, :, :, 0], cmap='gray')
            plt.title(f"HF Batch {b}, Slice {i}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

display_batch_slices_side_by_side(lf_input, hf_target, num_slices=15)




