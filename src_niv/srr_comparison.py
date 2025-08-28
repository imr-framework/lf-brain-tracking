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

subjects = ['26184'] # '59175','59233', '59877', '35547'
# Mismatch due to high field shape 

# Define the path to the IRF_3T folder (High Field Data)
nhp_base_path = './Data/IRF_3T'
day_idx = 1
visualize = True
visualize_pairs = True
padding = False
register2_hf = False
augmentation = True

# Training parameters
steps_per_epoch = 100
epochs = 75
batch_size = 2

for subject in subjects:
    print(f"\n=============================== Processing subject: {subject} ===============================")
    # Initialize data object and load data (HFMRI_data_IRF)
    print(f"\n===============================HF_MRI data processing  started .............")
    resampled_volume_hf_norm = load_and_preprocess_hf(subject, day_idx, visualize)
    # Load and preprocess LF data
    print(f"\n===============================LF_MRI data processing  started .............")
    resampled_volume_lf_be_norm = load_and_preprocess_lf(subject, day_idx, visualize)

    print("Resampled LF volume shape:", resampled_volume_lf_be_norm.shape)
    print("Resampled HF volume shape:", resampled_volume_hf_norm.shape)

    if register2_hf:
        resampled_volume_lf_be_norm = register_to_hf(resampled_volume_lf_be_norm, resampled_volume_hf_norm)

    if padding:
        resampled_volume_lf_be_norm = padding_LF(resampled_volume_lf_be_norm, resampled_volume_hf_norm, target_slices=64)
        print("After padding LF volume shape:", resampled_volume_lf_be_norm.shape) 

    if visualize_pairs:
        # Example: visualize slices 0-30
        visualize_pair(resampled_volume_lf_be_norm, resampled_volume_hf_norm, slice_indices=list(range(31)))

    # ------ Final HF and LF inputs -----
    lf_input_volume = resampled_volume_lf_be_norm.astype(np.float32)
    hf_input_volume = resampled_volume_hf_norm.astype(np.float32)
    hf_target_volume = resampled_volume_hf_norm.astype(np.float32)

    # Take slices 0:32 along (z, x, y) axis for each volume
    lf_input_volume = lf_input_volume[0:32, :, :]
    hf_input_volume = hf_input_volume[0:32, :, :]
    hf_target_volume = hf_target_volume[0:32, :, :]

    print("LF Input shape:", lf_input_volume.shape)
    print("HF Input shape:", hf_input_volume.shape) 
    print("HF volume shape:", hf_target_volume.shape)
   
    # calling the residual_srr_unet model
    model_type = 'residual_srr_unet2'
    model_case = 'single_encoder_unet'
    model_ = residual_srr_unet
    
    train_bilinear(lf_input_volume, hf_input_volume, hf_target_volume, model_type, model_case, model_, subject,
          day_idx, steps_per_epoch=steps_per_epoch, epochs=epochs, batch_size=batch_size, visualize_pairs=visualize_pairs)
