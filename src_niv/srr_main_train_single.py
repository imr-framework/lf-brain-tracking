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
subjects1 = ['26184', '30366', '35528'] # 59228
subjects1 = ['26184']
# Mismatch due to high field shape

# Define the path to the IRF_3T folder (High Field Data)
nhp_base_path = './Data/IRF_3T'
day_idx = 1
visualize = False
visualize_pairs = False
padding = False
register2_hf = True
augmentation = True

# Training parameters
steps_per_epoch = 2
epochs = 1
batch_size = 2

lf_input_volume_combined = []
hf_input_volume_combined = []
hf_target_volume_combined = []

for subject in subjects1:
    # for day_idx in [1, 2]:  # Assuming 0 = Day 1, 1 = Day 2
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

        # # ----- Append to Combined Lists -----
        # lf_input_volume_combined.append(lf_input_volume)
        # hf_input_volume_combined.append(hf_input_volume)
        # hf_target_volume_combined.append(hf_target_volume)

        # lf_input_volume_combined = np.stack(lf_input_volume_combined)
        # hf_input_volume_combined = np.stack(hf_input_volume_combined)
        # hf_target_volume_combined = np.stack(hf_target_volume_combined)

        # print("LF Input shape:", lf_input_volume_combined.shape)
        # print("HF Input shape:", hf_input_volume_combined.shape)
        # print("HF volume shape:", hf_target_volume_combined.shape)

        # calling the residual_srr_unet model
        model_type = 'residual_srr_unet_subjects_new'
        model_case = 'single_encoder_unet'
        model_ = residual_srr_unet

        train(lf_input_volume, hf_input_volume, hf_target_volume, 
              lf_input_volume, hf_input_volume, hf_target_volume,
              model_type, model_case, model_, subject,day_idx, steps_per_epoch=steps_per_epoch,
              epochs=epochs, batch_size=batch_size, visualize_pairs=visualize_pairs)
