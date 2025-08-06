# Subject-Specific Super-Resolution Reconstruction (SRR) Framework
# STEP 1: Load High Field (HF) and Low Field (LF) MRI Data [FIVE TIMEPOINTS] -- [DONE]
# Read HF;
# Check Voxel Sizes; # Read HF inputs resize 1,1,2 to 128,128, 32 --> PREPROCESS hf
# Read LF Images; --> # Check voxel size; # Read Low field 64,64, 16 ---> 128, 128, 32 --> Perform ZSSR
# Perform [Normalization] --> if required
# Perform [Brain extraction,Bias field correction, Contrast enhancement,Registration,Histogram matching ] --> if required
# Output HF and LF inputs same to Perform [Resampling] --> if required
# [Optional] Perform [Denoising model learning based on high Field and without high field] --> if required
# Subject specific learning model
# Subject-specific SRR model [One encoder model with optimization] and [Two encoder model with optimization] and [Teacher Student two encoder model]
# Take the first subject and predict the second, third, fourth, and fifth subjects.
# Save the SRR results to NIfTI files
# Comparison of SRR model with other models

import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./data_read_code')
from src_niv.prep_data import data_ops
from src_niv.read_lf5_data import process_subject
from src_niv.prep_lf import normalize, resize_mri_volume
from demo_read_data import read_lf_data

import os
import nibabel as nib
from nilearn import plotting
from nibabel.viewers import OrthoSlicer3D
import tensorflow as tf
# import matplotlib.pyplot as plt
import pydicom
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on what's installed
from tensorflow.keras import layers, models, Input
import ants
import nibabel as nib
from sklearn.feature_extraction import image
import os

# Define the path to the IRF_3T folder ( High Field Data)
nhp_base_path = './Data/IRF_3T'
subject = '26184'  # Example subject number, adjust as needed

subjects = os.listdir(nhp_base_path)
subjects = sorted(subjects)
print(f"Available subjects: {subjects}")
print(f"Selected subject: {subject}")

nhp_data_path = f'{nhp_base_path}/{subject}' #truct the full path to the DICOM folder

# Initialize data object and load data (IRF_3T)
data_obj = data_ops(nhp_data_path)

# Retrieve dictionary of 3D volumes (day1 to day5)
all_volumes = data_obj.data

# Select and visualize Day 1: 10 slices spaced 10 apart
day_idx = 1
volume_26184 = all_volumes[day_idx]

print(f"Type: {type(volume_26184)}")
print(f"Shape: {volume_26184.shape}")
print(f"Dtype: {volume_26184.dtype}")
print(f"Min: {np.min(volume_26184)}, Max: {np.max(volume_26184)}")
print(f"Mean: {np.mean(volume_26184):.2f}, Std: {np.std(volume_26184):.2f}")

if day_idx in all_volumes:
    vol = all_volumes[day_idx]
    slice_indices = list(range(0, 100, 10))  # [0, 10, 20, ..., 90]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f"Day {day_idx} - Every 10th Slice", fontsize=16)
    for ax, idx in zip(axes.flat, slice_indices):
        if idx < vol.shape[0]:
            ax.imshow(vol[idx], cmap='gray')
            ax.set_title(f"Slice {idx}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print(f"Day {day_idx} data not found.")

# Initialize data object and load data (LFMRI_data_IRF)
all_volumes_lf = process_subject(subject=subject)

# data_folder = 'Data/LFMRI_DATA_IRF/IRF_071E_2_C1_20240709/34507_D_minus28'
# output_folder = '/home/ajay/Documents/lf-brain-tracking/Data/LFMRI_DATA_IRF_nifti'
# subject = subject
# sub_folder ='3DTSE/3'
# file_name ='20240829_day2_3DTSC_12.nii'
# name = file_name
# im = read_lf_data(data_folder, output_folder, subject, sub_folder, file_name)
im = all_volumes_lf[day_idx-1]

print(f"LF_MRI data processing  started .............")
print("Max value:", np.max(np.abs(im)))
print("Min value:", np.min(np.abs(im)))
print("Data type of np.abs(im):", np.abs(im).dtype)
print("Shape of im:", im.shape)

num_slices = im.shape[2]
fig, axes = plt.subplots(2, 8, figsize=(20, 8))
# fig.suptitle(f'All Axial Slices for {name}\n{subject}\n{Visit_id}\n3DTSE/{subf}', fontsize=16)
axes = axes.flatten()

for i in range(16):
    if i < num_slices:
        slice_img = np.flipud(np.abs(im[:, :, i]).T)
        axes[i].imshow(slice_img, cmap='gray')
        axes[i].set_title(f'Slice {i + 1}')
        axes[i].axis('off')
    else:
        axes[i].axis('off')

plt.tight_layout()
# plt.savefig(f'Figures/{subject}/{fig_name}')
plt.show()
plt.close()

# # Prepare the HF data for SRR (e.g. normalization, resizing, etc.)

# hf = volume_26184  # HF image
# lf = im            # LF image

# # Normalize both
# hf = normalize(hf)
# lf = normalize(lf)

# # Resample HF to (256, 256, 64)
# # Original MRI shape: (100, 512, 512)
# hf_resized = resize_mri_volume(volume_26184, target_shape=(64, 256, 256))  # Output shape: (256, 256, 64)
# volume = np.random.rand(64, 256, 256)  # (D, H, W)
# # reordered = np.transpose(volume, (1, 2, 0))  # Now shape: (H, W, D)
# # print(reordered.shape)  # (512, 512, 100)
# # vol = reordered

# slice_indices = list(range(0, 64, 10))  # [0, 10, 20, ..., 90]

# fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# fig.suptitle(f"Day {day_idx} - Every 10th Slice", fontsize=16)
# for ax, idx in zip(axes.flat, slice_indices):
#     if idx < vol.shape[2]:
#         ax.imshow(volume[idx], cmap='gray')
#         ax.set_title(f"Slice {idx}")
#         ax.axis('off')
# plt.tight_layout()
# plt.show()

# num_slices = lf.shape[2]
# print(f"After resampling LF shape: {lf.shape}, dtype: {lf.dtype}, min: {np.min(lf)}, max: {np.max(lf)}, mean: {np.mean(lf):.2f}, std: {np.std(lf):.2f}")
# fig, axes = plt.subplots(2, 8, figsize=(20, 8))
# # fig.suptitle(f'All Axial Slices for {name}\n{subject}\n{Visit_id}\n3DTSE/{subf}', fontsize=16)
# axes = axes.flatten()

# for i in range(16):
#     if i < num_slices:
#         slice_img = np.flipud(np.abs(lf[:, :, i]).T)
#         axes[i].imshow(slice_img, cmap='gray')
#         axes[i].set_title(f'Slice {i + 1}')
#         axes[i].axis('off')
#     else:
#         axes[i].axis('off')

# plt.tight_layout()
# plt.show()
# plt.close()




# # Reshape to [H, W, D, C]
# lf_input = lf[..., np.newaxis]  # shape (64, 64, 16, 1)
# hf_target = hf_resized[..., np.newaxis]  # shape (256, 256, 64, 1)

# # --------------------------------------
# # 🔧 3D Residual UNet Definition
# # --------------------------------------

# def residual_block(x, filters, kernel_size=3):
#     shortcut = x
#     x = layers.Conv3D(filters, kernel_size, padding='same', activation='relu')(x)
#     x = layers.Conv3D(filters, kernel_size, padding='same')(x)
#     x = layers.add([shortcut, x])
#     x = layers.Activation('relu')(x)
#     return x

# def build_resunet_sr(input_shape=(64, 64, 16, 1), output_shape=(256, 256, 64, 1)):
#     inputs = Input(shape=input_shape)

#     # Encoder
#     c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
#     c1 = residual_block(c1, 32)
#     p1 = layers.MaxPooling3D()(c1)

#     c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(p1)
#     c2 = residual_block(c2, 64)
#     p2 = layers.MaxPooling3D()(c2)

#     # Bottleneck
#     bn = layers.Conv3D(128, 3, activation='relu', padding='same')(p2)
#     bn = residual_block(bn, 128)

#     # Decoder with Upsampling to match (256, 256, 64)
#     u2 = layers.UpSampling3D(size=(2, 2, 2))(bn)
#     u2 = layers.Conv3D(64, 3, padding='same', activation='relu')(u2)

#     u1 = layers.UpSampling3D(size=(2, 2, 2))(u2)
#     u1 = layers.Conv3D(32, 3, padding='same', activation='relu')(u1)

#     u0 = layers.UpSampling3D(size=(2, 2, 2))(u1)
#     u0 = layers.Conv3D(16, 3, padding='same', activation='relu')(u0)

#     out = layers.Conv3D(1, 1, activation='linear')(u0)

#     model = models.Model(inputs, out)
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# model = build_resunet_sr(input_shape=lf_input.shape, output_shape=hf_target.shape)

# # --------------------------------------
# # 🚀 Train Subject-Specific SRR Model
# # --------------------------------------

# model.fit(x=lf_input[np.newaxis, ...], y=hf_target[np.newaxis, ...], epochs=100)

# # Save model
# model.save("subject_srr_resunet_256x256x64.h5")

# # --------------------------------------
# # 🔮 Predict HF from Next Visit LF
# # --------------------------------------

# # Replace `next_lf` with next visit LF volume
# next_lf = normalize(next_visit_lf)[..., np.newaxis]  # (64, 64, 16, 1)
# hf_pred = model.predict(next_lf[np.newaxis, ...])[0, ..., 0]

# Save predicted HF as NIfTI (optional)
# nib.save(nib.Nifti1Image(hf_pred, affine=np.eye(4)), "predicted_hf.nii.gz")


# # Add channel dimension
# hf_norm = hf_norm[..., np.newaxis]
# lf_norm = lf_norm[..., np.newaxis]

# # Optional patch extraction (for large volumes)
# hf_patches = extract_patches_3d(hf_norm, patch_shape=(32, 64, 64), max_patches=200)
# lf_patches = extract_patches_3d(lf_norm, patch_shape=(32, 64, 64), max_patches=200)

# --------------------------------------
# 🔧 3D Residual UNet Definition
# --------------------------------------

# def residual_block(x, filters, kernel_size=3):
#     shortcut = x
#     x = layers.Conv3D(filters, kernel_size, padding='same', activation='relu')(x)
#     x = layers.Conv3D(filters, kernel_size, padding='same')(x)
#     x = layers.add([shortcut, x])
#     x = layers.Activation('relu')(x)
#     return x

# def build_resunet(input_shape=(32, 64, 64, 1)):
#     inputs = Input(shape=input_shape)

#     # Encoder
#     c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
#     c1 = residual_block(c1, 32)
#     p1 = layers.MaxPooling3D()(c1)

#     c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(p1)
#     c2 = residual_block(c2, 64)
#     p2 = layers.MaxPooling3D()(c2)

#     c3 = layers.Conv3D(128, 3, activation='relu', padding='same')(p2)
#     c3 = residual_block(c3, 128)

#     # Decoder
#     u2 = layers.UpSampling3D()(c3)
#     concat2 = layers.concatenate([u2, c2])
#     c4 = layers.Conv3D(64, 3, activation='relu', padding='same')(concat2)
#     c4 = residual_block(c4, 64)

#     u1 = layers.UpSampling3D()(c4)
#     concat1 = layers.concatenate([u1, c1])
#     c5 = layers.Conv3D(32, 3, activation='relu', padding='same')(concat1)
#     c5 = residual_block(c5, 32)

#     outputs = layers.Conv3D(1, 1, activation='linear')(c5)

#     model = models.Model(inputs, outputs)
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# model = build_resunet(input_shape=lf_patches.shape[1:])

# # --------------------------------------
# # 🚀 Train Subject-Specific Model
# # --------------------------------------

# model.fit(x=lf_patches, y=hf_patches, batch_size=4, epochs=50, validation_split=0.1)

# # Save model
# model.save("subject_srr_resunet.h5")

# # --------------------------------------
# # 🔮 Predict on Future LF Volume
# # --------------------------------------

# # Load next visit LF (replace with actual data)
# lf_next = normalize(resample_to_target(next_visit_lf, target_shape=(512, 512, 100)))
# lf_next = lf_next[..., np.newaxis]

# # Predict directly (if memory allows)
# hf_pred = model.predict(lf_next[np.newaxis, ...])[0, ..., 0]

# # Save predicted HF as NIfTI (optional)
# nib.save(nib.Nifti1Image(hf_pred, affine=np.eye(4)), "predicted_hf.nii.gz")

# Display the data

# Perform and call SRR reconstruction framework

# Save the SRR results to NIfTI files

# Display the SRR results

# Compare the SRR results with the original data

# Identify and visualize differences using XAI techniques
