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
from src_niv.utils import visualize_hf_slices, visualize_lf_slices,rotate_slices, visualize_resampled, resample_volume, visualize_planes,visualize_pair, normalize_volume
from src_niv.prep_lf import normalize, resize_mri_volume
from src_niv.zssr import  zero_shot_super_resolution, extract_brain, extract_lf_volumes
from src_niv.subject_model import build_dual_encoder_unet
from src_niv.metrics import psnr, ssim
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
from tensorflow.keras.metrics import MeanSquaredError
import ants
import nibabel as nib
from sklearn.feature_extraction import image
import os
from skimage.transform import resize  # ✅ Required for 3D resizing

# Define the path to the IRF_3T folder ( High Field Data)
nhp_base_path = './Data/IRF_3T'
subject = '26184'  # Example subject number, adjust as needed
day_idx = 1
visualize = False

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

volume_26184 = all_volumes[day_idx]
voxel_sizes_26184 = voxel_sizes[day_idx]

print(f'Day {day_idx} and voxel size {voxel_sizes_26184}')

if visualize == True:
    
    print(f"Type: {type(volume_26184)}")
    print(f"Shape: {volume_26184.shape}")
    print(f"Dtype: {volume_26184.dtype}")
    print(f"Min: {np.min(volume_26184)}, Max: {np.max(volume_26184)}")
    print(f"Mean: {np.mean(volume_26184):.2f}, Std: {np.std(volume_26184):.2f}")
    visualize_hf_slices(all_volumes)
    # visualize_planes(all_volumes, voxel_sizes, day_idx)

# Resample the HF volume
# Define the new desired voxel spacing (z, y, x) in mm
new_spacing = [2, 1.09, 1.09] # z=2mm, y=1mm, x=1mm
resampled_volume_hf = resample_volume(volume_26184, voxel_sizes_26184, new_spacing)
resampled_volume_hf_norm = normalize_volume(resampled_volume_hf)
if visualize == True:
    visualize_resampled(resampled_volume_hf_norm)

# Initialize data object and load data (LFMRI_data_IRF)

all_volumes_lf = process_subject(subject=subject)
if visualize == True:
   visualize_lf_slices(all_volumes_lf)

im = all_volumes_lf[day_idx-1]
print(f"LF_MRI data processing  started .............")

# Define the new desired voxel spacing (z, y, x) in mm
orig_spacing = [2, 2, 5] # z=2mm, y=1mm, x=1mm
new_spacing = [1, 1, 2] # z=2mm, y=1mm, x=1mm

resampled_volume_lf = resample_volume(im, orig_spacing, new_spacing)
resampled_volume_lf = rotate_slices(resampled_volume_lf)

if visualize == True:
    visualize_resampled(resampled_volume_lf)

resampled_volume_lf_be = extract_lf_volumes(resampled_volume_lf)
resampled_volume_lf_be_norm = normalize_volume(resampled_volume_lf_be)

print("Super-resolved volume shape:", resampled_volume_lf_be_norm.shape)
print("Resampled volume shape:", resampled_volume_hf_norm.shape)

if visualize == True:
    # Example: visualize slices 10, 20, 30
    visualize_pair(resampled_volume_lf_be_norm, resampled_volume_hf_norm, slice_indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])

# --- Final HF and LF inputs ---

lf_input_volume = resampled_volume_lf_be_norm.astype(np.float32)
hf_input_volume = resampled_volume_hf_norm.astype(np.float32)
hf_target_volume = resampled_volume_hf_norm.astype(np.float32)

# Take slices 0:32 along (z, x, y) axis for each volume
lf_input_volume = lf_input_volume[0:32, :, :]
hf_input_volume = hf_input_volume[0:32, :, :]
hf_target_volume = hf_target_volume[0:32, :, :]

# Check if required volumes exist
if 'lf_input_volume' not in locals() or 'hf_target_volume' not in locals():
    print("\nRequired volumes (super_resolved_volume and resampled_volume) not available. Cannot proceed.")
else:
        # Add batch and channel dimensions
        # Input shapes for the model should be (Batch, H, W, D, C)
        x_zssr_train = np.expand_dims(lf_input_volume, axis=0) # Add batch dim
        x_zssr_train = np.expand_dims(x_zssr_train, axis=-1)           # Add channel dim (1 channel)

        x_hf_train = np.expand_dims(hf_input_volume, axis=0)   # Add batch dim
        x_hf_train = np.expand_dims(x_hf_train, axis=-1)             # Add channel dim (1 channel)

        y_train = np.expand_dims(hf_target_volume, axis=0)   # Add batch dim
        y_train = np.expand_dims(y_train, axis=-1)             # Add channel dim (1 channel)

        if visualize == True:
            # Example: visualize slices 10, 20, 30
            visualize_pair(x_zssr_train, y_train, slice_indices=[1,2,3,4,5,6,7,8,12,16,18,20,21,22,23,24,26,28])

        print(f"Prepared ZSSR training input shape: {x_zssr_train.shape}")
        print(f"Prepared HF training input shape: {x_hf_train.shape}")
        print(f"Prepared training target shape: {y_train.shape}")

        # --- Build and Compile the Dual-Encoder Model ---
        # Input shapes for the model should match the shape of a single sample (H, W, D, C)
        model = build_dual_encoder_unet(input_shape_zssr=x_zssr_train.shape[1:],
                                        input_shape_hf=x_hf_train.shape[1:],
                                        output_shape=y_train.shape[1:]) # Ensure output shape matches target

        # Compile the model
        # --- Compile the model ---
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[psnr, ssim, MeanSquaredError(name='mse')])  # Added metrics her) # Mean Squared Error is common for regression
        # Mean Squared Error is common for regression
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
            step = max(1, predicted_volume_dual.shape[0] // num_slices_to_show)
            display_indices = list(range(0, predicted_volume_dual.shape[-1], step))[:num_slices_to_show]


            if display_indices:
                fig, axes = plt.subplots(2, len(display_indices), figsize=(3 * len(display_indices), 6))
                fig.suptitle('Target HF vs. Predicted Dual-Encoder SRR Output', fontsize=16)

                for i, slice_idx in enumerate(display_indices):
                    # Target HF slice (using the normalized target volume)
                    target_slice = hf_target_volume[slice_idx, :, :]
                    axes[0, i].imshow(np.flipud(target_slice.T), cmap='gray')
                    axes[0, i].set_title(f'Slice {slice_idx}\nTarget HF')
                    axes[0, i].axis('off')

                    # Predicted SRR slice (using the predicted volume)
                    predicted_slice = predicted_volume_dual[slice_idx, :, :]
                    axes[1, i].imshow(np.flipud(predicted_slice.T), cmap='gray')
                    axes[1, i].set_title(f'Slice {slice_idx}\nPredicted SRR')
                    axes[1, i].axis('off')

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()


# Display the data

# Perform and call SRR reconstruction framework

# Save the SRR results to NIfTI files

# Display the SRR results

# Compare the SRR results with the original data

# Identify and visualize differences using XAI techniques
