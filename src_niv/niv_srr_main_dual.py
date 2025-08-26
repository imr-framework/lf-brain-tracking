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

# =================================
# Can add data Augmentation and patch based training for better results;
# Validation with last timepoint data
# =================================

import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./data_read_code')
from src_niv.prep_data import data_ops
from src_niv.read_lf5_data import process_subject
from src_niv.utils import srr_generator, load_and_preprocess_hf, load_and_preprocess_lf, visualize_hf_slices,padding_LF, visualize_lf_slices,rotate_slices, visualize_resampled, resample_volume, visualize_planes,visualize_pair, normalize_volume
from src_niv.prep_lf import normalize, resize_mri_volume
from src_niv.zssr import  zero_shot_super_resolution, extract_brain, extract_lf_volumes
from src_niv.models.subject_model import build_dual_encoder_unet
from src_niv.models.ResUNet import residual_srr_unet
from src_niv.metrics import psnr, ssim
from demo_read_data import read_lf_data
from src_niv.prep_lf import register_to_hf

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
from skimage.transform import resize  # Required for 3D resizing

# Define the path to the IRF_3T folder ( High Field Data)
nhp_base_path = './Data/IRF_3T'
model_type = 'build_dual_encoder_unet'  # Options: 'single_encoder_unet', 'dual_encoder_unet', 'teacher_student_unet'
model_case = 'single_encoder_unet'  # Options: 'single_encoder_unet', 'dual_encoder_unet'
subject = '30366'  # Example subject number, adjust as needed
day_idx = 1
visualize = False
visualize_pairs = True
padding = False
register2_hf = True
augmentation = True
output_path = f'./Data/Results/{model_type}/{subject}'
os.makedirs(output_path, exist_ok=True)

# Initialize data object and load data (HFMRI_data_IRF)
print(f"\n===============================HF_MRI data processing  started .............")
resampled_volume_hf_norm = load_and_preprocess_hf(subject, day_idx, visualize)
#Load and preprocess LF data
print(f"\n===============================LF_MRI data processing  started .............")
resampled_volume_lf_be_norm = load_and_preprocess_lf(subject, day_idx, visualize)

print("Resampled LF volume shape:", resampled_volume_lf_be_norm.shape)
print("Resampled HF volume shape:", resampled_volume_hf_norm.shape)

if register2_hf == True:
    resampled_volume_lf_be_norm = register_to_hf(resampled_volume_lf_be_norm, resampled_volume_hf_norm)

if padding == True:
    resampled_volume_lf_be_norm = padding_LF(resampled_volume_lf_be_norm,resampled_volume_hf_norm, target_slices=64)
    print("After padding LF volume shape:", resampled_volume_lf_be_norm.shape) 

if visualize_pairs == True:
    # Example: visualize slices 10, 20, 30
    visualize_pair(resampled_volume_lf_be_norm, resampled_volume_hf_norm, slice_indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])

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

#save the preprocessed volumes for reference as Nifti files in the output path with appropriate name and day index
nib.save(nib.Nifti1Image(lf_input_volume, affine=np.eye(4)), os.path.join(output_path, f'LF_input_volume_day{day_idx}.nii.gz'))
nib.save(nib.Nifti1Image(hf_target_volume, affine=np.eye(4)), os.path.join(output_path, f'HF_input_volume_day{day_idx}.nii.gz'))

gen = srr_generator(lf_input_volume, hf_target_volume, batch_size=2, patch_z=32, augment=True, extra_slices=50)

lf_input_volume, hf_target_volume = next(gen)
print(lf_input_volume.shape)  # (2, 32, 128, 128, 1)
print(hf_target_volume.shape)  # (2, 32, 128, 128, 1)


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

        if visualize_pairs == True:
            # Example: visualize slices 10, 20, 30
            visualize_pair(x_zssr_train, y_train, slice_indices=[1,2,3,4,5,6,7,8,12,16,18,20,21,22,23,24,26,28])

        print(f"Prepared ZSSR training input shape: {x_zssr_train.shape}")
        print(f"Prepared HF training input shape: {x_hf_train.shape}")
        print(f"Prepared training target shape: {y_train.shape}")


        # --- Build and Compile the Dual-Encoder Model ---

        # Input shapes for the model should match the shape of a single sample (H, W, D, C)
        if model_case == 'single_encoder_unet':
            model = residual_srr_unet(input_shape=x_zssr_train.shape[1:])
        elif model_case == 'dual_encoder_unet':
            model = build_dual_encoder_unet(input_shape_zssr=x_zssr_train.shape[1:],
                                            input_shape_hf=x_hf_train.shape[1:],
                                            output_shape=y_train.shape[1:]) # Ensure output shape matches target
        else:
            raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")
        
        # --- Compile the model ---
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[psnr, ssim, MeanSquaredError(name='mse')])  # Added metrics her) # Mean Squared Error is common for regression
        # Mean Squared Error is common for regression
        model.summary()

        # --- Train the CNN ---
        print("\nStarting dual-encoder model training...")
        
        steps_per_epoch = 100
        epochs = 100 # Number of training epochs
        batch_size = 1 # Batch size (1 for a single volume)

        # Add a callback to save the model during training
        checkpoint_filepath_dual = os.path.join(output_path, f'{model_type}_model_checkpoint_day{day_idx}.h5')
        model_checkpoint_callback_dual = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath_dual,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True)

        # Train the model using both inputs
        if model_case == 'single_encoder_unet':
            history = model.fit(gen,
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[model_checkpoint_callback_dual]
                                # validation_data=(x_zssr_val, y_val) # Add validation data if available
                                )
        elif model_case == 'dual_encoder_unet':
            history = model.fit([x_zssr_train, x_hf_train], y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[model_checkpoint_callback_dual]
                                # validation_data=([x_zssr_val, x_hf_val], y_val) # Add validation data if available
                                )
        else:
            raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")
        
        print("\nDual-encoder model training finished.")

        # --- Save the Final Model ---
        final_model_path_dual = os.path.join(output_path, f'{model_type}_model_final_day{day_idx}.h5')
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
            
            from tensorflow.keras.losses import MeanSquaredError as mse_loss_fn # Alias it to mse for custom_objects

            try:
                

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
            
            # # --- Evaluate the Model --- Change
            # if model_case == 'single_encoder_unet':
            #     eval_results_dual = best_dual_model.evaluate(x_zssr_train, y_train, verbose=0)
            # elif model_case == 'dual_encoder_unet':
            #     eval_results_dual = best_dual_model.evaluate([x_zssr_train, x_hf_train], y_train, verbose=0)
            # else:
            #     raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")
            
            # predict the SRR volume using the best model
            if model_case == 'single_encoder_unet':
                predicted_volume_dual = best_dual_model.predict(x_zssr_train)
            elif model_case == 'dual_encoder_unet':
                predicted_volume_dual = best_dual_model.predict([x_zssr_train, x_hf_train])
            else:
                raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")

            # Remove batch and channel dimensions
            predicted_volume_dual = np.squeeze(predicted_volume_dual)

            print(f"Predicted volume shape (dual-encoder): {predicted_volume_dual.shape}")
            # Save the predicted volume as a NIfTI file in the output folder with day index
            predicted_nifti = nib.Nifti1Image(predicted_volume_dual.astype(np.float32), affine=np.eye(4))
            predicted_nifti_path = os.path.join(output_path, f'Predicted_volume_day{day_idx}.nii.gz')
            nib.save(predicted_nifti, predicted_nifti_path)
            print(f"Predicted SRR volume saved to: {predicted_nifti_path}")

            # Display some slices from the predicted volume vs the target HF volume
            print("\nDisplaying slices: Target HF vs. Predicted Dual-Encoder Output...")

            num_slices_to_show = 10 # Number of slices to display
            step = max(1, predicted_volume_dual.shape[0] // num_slices_to_show)
            display_indices = list(range(0, predicted_volume_dual.shape[-1], step))[:num_slices_to_show]

            if display_indices:
                fig, axes = plt.subplots(3, num_slices_to_show, figsize=(2*num_slices_to_show, 6))
                fig.suptitle(f"Day {day_idx} - True HF (top), LF (middle), Predicted HF (bottom) Slices", fontsize=16)
                
                for i, idx in enumerate(slice_indices):
                    # True HF slice (top)
                    axes[0, i].imshow(hf_vol[idx, :, :], cmap='gray')
                    axes[0, i].set_title(f"Slice {idx}")
                    axes[0, i].axis('off')
                    
                    # LF slice (middle)
                    axes[1, i].imshow(lf_vol[idx, :, :], cmap='gray')
                    axes[1, i].axis('off')
                    
                    # Predicted HF slice (bottom)
                    axes[2, i].imshow(pred_vol[idx, :, :], cmap='gray')
                    axes[2, i].axis('off')
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()

