import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./data_read_code')
from src_niv.prep_data import data_ops
from src_niv.read_lf5_data import process_subject
from src_niv.utils import display_pred, load_and_preprocess_hf, load_and_preprocess_lf, visualize_hf_slices,padding_LF, visualize_lf_slices,rotate_slices, visualize_resampled, resample_volume, visualize_planes,visualize_pair, normalize_volume
from src_niv.prep_lf import normalize, resize_mri_volume
from src_niv.augment import srr_generator
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
subjects1 = ['30366'] # '59877', '59175','59233', '59877', '35547'

# Mismatch due to high field shape

# Define the path to the IRF_3T folder (High Field Data)
nhp_base_path = './Data/IRF_3T'
day_idx = 2
visualize = False
visualize_pairs = False
padding = False
register2_hf = True
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
    for day_idx in [1]:  # Assuming 0 = Day 1, 1 = Day 2
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

# # Validation data
# print('-----------------------------\n\nLoading validation data .................................-----------------')
# subjects_val = ['35528']
# for subject_v in subjects_val:
#     for day_idx in [2]:  # Assuming 0 = Day 1, 1 = Day 2
#         print(f"\n=============================== Processing subject: {subject_v}, Day: {day_idx + 1} ===============================")
#         # ----- Load HF data -----
#         print(f"\n=============================== HF_MRI data processing started .............")
#         resampled_volume_hf_norm = load_and_preprocess_hf(subject_v, day_idx, visualize)
#         # ----- Load LF data -----
#         print(f"\n=============================== LF_MRI data processing started .............")
#         resampled_volume_lf_be_norm = load_and_preprocess_lf(subject_v, day_idx, visualize)
#         print("Resampled LF volume shape:", resampled_volume_lf_be_norm.shape)
#         print("Resampled HF volume shape:", resampled_volume_hf_norm.shape)
        
#         # Register LF to HF
#         if register2_hf:
#             resampled_volume_lf_be_norm = register_to_hf(resampled_volume_lf_be_norm, resampled_volume_hf_norm)
        
#         # Padding
#         if padding:
#             resampled_volume_lf_be_norm = padding_LF(
#                 resampled_volume_lf_be_norm,
#                 resampled_volume_hf_norm,
#                 target_slices=64
#             )
#             print("After padding LF volume shape:", resampled_volume_lf_be_norm.shape)
        
#         # Visualization (optional)
#         if visualize_pairs:
#             visualize_pair(
#                 resampled_volume_lf_be_norm,
#                 resampled_volume_hf_norm,
#                 slice_indices=list(range(31))
#             )
        
#         # ----- Final Volume Preparation -----
#         lf_input_volume_val = resampled_volume_lf_be_norm.astype(np.float32)
#         hf_input_volume_val = resampled_volume_hf_norm.astype(np.float32)
#         hf_target_volume_val = resampled_volume_hf_norm.astype(np.float32)
        
#         # Slice selection
#         lf_input_volume_val = lf_input_volume_val[0:32, :, :]
#         hf_input_volume_val = hf_input_volume_val[0:32, :, :]
#         hf_target_volume_val = hf_target_volume_val[0:32, :, :]
#         print("LF Input shape:", lf_input_volume_val.shape)
#         print("HF Input shape:", hf_input_volume_val.shape)
#         print("HF volume shape:", hf_target_volume_val.shape)

# gen = srr_generator(lf_input_volume, hf_target_volume, batch_size=batch_size, patch_z=32, augment=True, num_augmented_copies=6)
train_gen = srr_generator(lf_input_volume, hf_target_volume, batch_size=1, patch_z=32, patch_xy=128, augment=True, extra_slices=50, noise_sigma=0.03)
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




