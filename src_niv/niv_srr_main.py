# Subject-Specific Super-Resolution Reconstruction (SRR) Framework
# STEP 1: Load High Field (HF) and Low Field (LF) MRI Data [FIVE TIMEPOINTS] -- [DONE]

# Read HF --> Check Voxel Sizes; # Read HF inputs resize 1,1,2 to 128,128, 32 --> PREPROCESS hf
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
from src_niv.utils import visualize_hf_slices, visualize_lf_slices,visualize_resampled,resample_volume
from src_niv.prep_lf import normalize, resize_mri_volume
from src_niv.zssr import  zero_shot_super_resolution, extract_brain, extract_lf_volumes
from src_niv.subject_model import build_dual_encoder_unet
from demo_read_data import read_lf_data

import os
import cv2
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
from skimage.transform import resize  # ✅ Required for 3D resizing

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
voxel_sizes = data_obj.voxel_sizes
# Select and visualize Day 1: 10 slices spaced 10 apart
day_idx = 1
visualize = False

volume_26184 = all_volumes[day_idx]
voxel_sizes_26184 = voxel_sizes[day_idx]

print(f'Day {day_idx} and voxel size {voxel_sizes_26184}')

if visualize == True:
    
    print(f"Type: {type(volume_26184)}")
    print(f"Shape: {volume_26184.shape}")
    print(f"Dtype: {volume_26184.dtype}")
    print(f"Min: {np.min(volume_26184)}, Max: {np.max(volume_26184)}")
    print(f"Mean: {np.mean(volume_26184):.2f}, Std: {np.std(volume_26184):.2f}")
    visualize_hf_slices(all_volumes,voxel_sizes, day_idx)

# Resample the HF volume
# Define the new desired voxel spacing (z, y, x) in mm
new_spacing = [2.2, 1.09, 1.09] # z=2mm, y=1mm, x=1mm

resampled_volume = resample_volume(volume_26184, voxel_sizes_26184, new_spacing)

if visualize == False:
    visualize_resampled(resampled_volume)

# Initialize data object and load data (LFMRI_data_IRF)
all_volumes_lf = process_subject(subject=subject)

im = all_volumes_lf[day_idx-1]
print(f"LF_MRI data processing  started .............")

if visualize == True:
    
    print("Max value:", np.max(np.abs(im)))
    print("Min value:", np.min(np.abs(im)))
    print("Data type of np.abs(im):", np.abs(im).dtype)
    print("Shape of im:", im.shape)
    visualize_lf_slices(im)

valid_orig_volume, valid_refined_be_volume, super_resolved_volume= extract_lf_volumes(im)

print(super_resolved_volume.shape)
print(resampled_volume.shape)

# --- Data Preparation ---

# Assuming 'super_resolved_volume' is the 3D volume from the ZSSR placeholder (input for encoder 1)
# Assuming 'volume_26184' is the original HF 3T volume (input for encoder 2 and target)

# Check if required volumes exist
if 'super_resolved_volume' not in locals() or 'resampled_volume' not in locals():
    print("\nRequired volumes (super_resolved_volume and resampled_volume) not available. Cannot proceed.")
else:
    # Define target output shape
    target_output_shape_spatial = (128, 128, 32) # H, W, D
    target_output_shape = target_output_shape_spatial + (1,) # Add channel dimension

    # Prepare ZSSR input (must be scaled to target spatial resolution)
    # The placeholder ZSSR currently outputs 4x the original slice dimensions.
    # Original slice shape was e.g., (H, W). Let's assume it was (128, 128).
    # ZSSR output slice shape would be (512, 512).
    # The ZSSR output volume shape would be (512, 512, num_slices).
    # This needs to be resized to the U-Net input spatial shape (256, 256, 64).

    # First, ensure super_resolved_volume is in (H, W, D) format if needed.
    # The placeholder stacks along the last axis, so it should be (H_sr, W_sr, D_orig)
    # where H_sr = H_orig * scale_factor, W_sr = W_orig * scale_factor.
    # D_orig is the original number of slices from 'im'.

    # We need to resize 'super_resolved_volume' to target_output_shape_spatial (256, 256, 64)
    # Assuming super_resolved_volume is (H_sr, W_sr, D_orig)
    print(f"\nResizing ZSSR volume to {target_output_shape_spatial}...")
    # try:
    # Ensure data type is float32 for resizing
    super_resolved_volume_float = super_resolved_volume.astype(np.float32)
    resized_zssr_volume = resize(super_resolved_volume_float, target_output_shape_spatial, anti_aliasing=True)
    print(f"Resized ZSSR volume shape: {resized_zssr_volume.shape}")

    # Prepare HF input
    # volume_26184 is the original HF 3T, assumed (slices, H, W).
    # Transpose to (H, W, slices) and resize to (256, 256, 64).
    if resampled_volume.ndim == 3:
        original_hf_volume = np.transpose(resampled_volume, (1, 2, 0)) # Shape (H, W, slices)
        print(f"Original HF volume shape (after transpose): {original_hf_volume.shape}")

        # Resize the original HF volume to the target shape (256, 256, 64)
        # Ensure data type is float32
        original_hf_volume_float = original_hf_volume.astype(np.float32)
        resized_hf_input_volume = resize(original_hf_volume_float, target_output_shape_spatial, anti_aliasing=True)
        print(f"Resized HF input volume shape: {resized_hf_input_volume.shape}")

        # Also use the resized HF volume as the target output
        resized_hf_target_volume = resized_hf_input_volume # Same data for target

        # Normalize the data (e.g., to [0, 1])
        # Normalize resized ZSSR input
        zssr_min, zssr_max = np.min(resized_zssr_volume), np.max(resized_zssr_volume)
        if zssr_max - zssr_min > 1e-6:
            resized_zssr_volume_norm = (resized_zssr_volume - zssr_min) / (zssr_max - zssr_min)
        else:
            resized_zssr_volume_norm = np.zeros_like(resized_zssr_volume)

        # Normalize resized HF input
        hf_in_min, hf_in_max = np.min(resized_hf_input_volume), np.max(resized_hf_input_volume)
        if hf_in_max - hf_in_min > 1e-6:
              resized_hf_input_volume_norm = (resized_hf_input_volume - hf_in_min) / (hf_in_max - hf_in_min)
        else:
              resized_hf_input_volume_norm = np.zeros_like(resized_hf_input_volume)

        # Normalize HF target
        hf_target_min, hf_target_max = np.min(resized_hf_target_volume), np.max(resized_hf_target_volume)
        if hf_target_max - hf_target_min > 1e-6:
              resized_hf_target_volume_norm = (resized_hf_target_volume - hf_target_min) / (hf_target_max - hf_target_min)
        else:
              resized_hf_target_volume_norm = np.zeros_like(resized_hf_target_volume)


        # Add batch and channel dimensions
        # Input shapes for the model should be (Batch, H, W, D, C)
        x_zssr_train = np.expand_dims(resized_zssr_volume_norm, axis=0) # Add batch dim
        x_zssr_train = np.expand_dims(x_zssr_train, axis=-1)           # Add channel dim (1 channel)

        x_hf_train = np.expand_dims(resized_hf_input_volume_norm, axis=0)   # Add batch dim
        x_hf_train = np.expand_dims(x_hf_train, axis=-1)             # Add channel dim (1 channel)

        y_train = np.expand_dims(resized_hf_target_volume_norm, axis=0)   # Add batch dim
        y_train = np.expand_dims(y_train, axis=-1)             # Add channel dim (1 channel)


        print(f"Prepared ZSSR training input shape: {x_zssr_train.shape}")
        print(f"Prepared HF training input shape: {x_hf_train.shape}")
        print(f"Prepared training target shape: {y_train.shape}")

        # --- Build and Compile the Dual-Encoder Model ---
        # Input shapes for the model should match the shape of a single sample (H, W, D, C)
        model = build_dual_encoder_unet(input_shape_zssr=x_zssr_train.shape[1:],
                                        input_shape_hf=x_hf_train.shape[1:],
                                        output_shape=y_train.shape[1:]) # Ensure output shape matches target


        # Compile the model
        model.compile(optimizer='adam', loss='mse') # Mean Squared Error is common for regression
        model.summary()

        # --- Train the CNN ---
        print("\nStarting dual-encoder model training...")

        epochs = 100 # Number of training epochs
        batch_size = 1 # Batch size (1 for a single volume)

        # Add a callback to save the model during training
        checkpoint_filepath_dual = 'SRR_models/dual_encoder_unet_checkpoint.h5'
        model_checkpoint_callback_dual = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath_dual,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True)

        # Train the model using both inputs
        history = model.fit([x_zssr_train, x_hf_train], y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[model_checkpoint_callback_dual]
                            # validation_data=([x_zssr_val, x_hf_val], y_val) # Add validation data if available
                            )

        print("\nDual-encoder model training finished.")

        # --- Save the Final Model ---
        final_model_path_dual = 'SRR_models/dual_encoder_unet_model.h5'
        model.save(final_model_path_dual)
        print(f"\nFinal dual-encoder model saved to: {final_model_path_dual}")


        # Load the best model saved by the checkpoint
        # If 'mse' was used as a metric:
        from tensorflow.keras.metrics import MeanSquaredError

        # Try loading the model, potentially with custom_objects if the direct load fails
        try:
            # Attempt loading directly first, as it should work for standard 'mse'
            best_dual_model = models.load_model(checkpoint_filepath_dual)
            print(f"Best dual-encoder model based on checkpoint loaded from: {checkpoint_filepath_dual}")

        except Exception as e_direct:
            print(f"Direct loading failed: {e_direct}")
            print("Attempting to load with 'mse' explicitly provided in custom_objects...")
            # If direct loading fails, try providing MeanSquaredError as a custom object.
            # Note: The error message specifically mentions 'function', not a class instance.
            # If 'mse' was used directly as a string in compile (e.g., `loss='mse'`),
            # Keras expects to find it registered or as a standard name.
            # Let's try providing the standard metric class, though the error points to a function.
            # A common pattern is using the class instance `MeanSquaredError()`.
            # If the model was compiled with `loss=tf.keras.metrics.mse` (which is unusual, mse is typically a loss),
            # then providing `tf.keras.metrics.mse` might work if it's a function alias.

            # Let's try the most likely case: 'mse' string for loss or metric.
            # Providing the Metric class is a common way to handle potential custom object issues for standard names.
            # If it was used as a loss, provide the Loss class.
            # The error context implies it's looking for a 'function'. If you compiled with
            # `metrics=['mse']`, it's looking for the metric function/class.
            # If you compiled with `loss='mse'`, it's looking for the loss function/class.
            # Assuming `mse` was used as a loss string based on typical practice.
            # Let's import the loss function directly just in case.
            from tensorflow.keras.losses import MeanSquaredError as mse_loss_fn # Alias it to mse for custom_objects

            try:
                # Try loading by mapping the string 'mse' to the standard MeanSquaredError class/function
                # Note: Passing functions/classes directly can be tricky if the model was saved with strings.
                # A safer bet might be to re-compile the loaded model after loading only weights,
                # but the current code loads the full model.

                # If the model was compiled with `loss='mse'`, this might work:
                best_dual_model = models.load_model(
                    checkpoint_filepath_dual,
                    custom_objects={'mse': mse_loss_fn()} # Provide the loss class instance
                    # If it was a metric and used as 'mse':
                    # custom_objects={'mse': MeanSquaredError()} # Provide the metric class instance
                    # If it was specifically looking for a function aliased as 'mse':
                    # custom_objects={'mse': tf.keras.metrics.mse} # If this alias exists and was used
                )
                print(f"Best dual-encoder model based on checkpoint loaded from: {checkpoint_filepath_dual} using custom_objects.")

            except Exception as e_custom:
                print(f"Loading with custom_objects failed as well: {e_custom}")
                print("Could not load the best dual-encoder model.")
                # Handle the case where the model cannot be loaded (e.g., exit, use a fallback model)
                # For now, the code will stop here if loading fails.

            # --- Evaluate the Model ---
            # Predict using the best model on the training data
            predicted_volume_dual = best_dual_model.predict([x_zssr_train, x_hf_train])

            # Remove batch and channel dimensions
            predicted_volume_dual = np.squeeze(predicted_volume_dual)

            print(f"Predicted volume shape (dual-encoder): {predicted_volume_dual.shape}")

            # Display some slices from the predicted volume vs the target HF volume
            print("\nDisplaying slices: Target HF vs. Predicted Dual-Encoder Output...")

            num_slices_to_show = 10 # Number of slices to display
            step = max(1, predicted_volume_dual.shape[-1] // num_slices_to_show)
            display_indices = list(range(0, predicted_volume_dual.shape[-1], step))[:num_slices_to_show]


            if display_indices:
                fig, axes = plt.subplots(2, len(display_indices), figsize=(3 * len(display_indices), 6))
                fig.suptitle('Target HF vs. Predicted Dual-Encoder SRR Output', fontsize=16)

                for i, slice_idx in enumerate(display_indices):
                    # Target HF slice (using the normalized target volume)
                    target_slice = resized_hf_target_volume_norm[:, :, slice_idx]
                    axes[0, i].imshow(np.flipud(target_slice.T), cmap='gray')
                    axes[0, i].set_title(f'Slice {slice_idx}\nTarget HF')
                    axes[0, i].axis('off')

                    # Predicted SRR slice (using the predicted volume)
                    predicted_slice = predicted_volume_dual[:, :, slice_idx]
                    axes[1, i].imshow(np.flipud(predicted_slice.T), cmap='gray')
                    axes[1, i].set_title(f'Slice {slice_idx}\nPredicted SRR')
                    axes[1, i].axis('off')

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()

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
