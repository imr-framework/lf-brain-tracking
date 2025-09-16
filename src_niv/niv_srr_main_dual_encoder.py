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
# improve loss function with perceptual loss; Hybrid loss; SSIM loss
# Add attention mechanism
# =================================

import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./data_read_code')
from src_niv.prep_data import data_ops
from src_niv.read_lf5_data import process_subject
from src_niv.utils import display_pred, load_and_preprocess_hf, load_and_preprocess_lf, visualize_hf_slices,padding_LF, visualize_lf_slices,rotate_slices, visualize_resampled, resample_volume, visualize_planes,visualize_pair, normalize_volume
from src_niv.augment import srr_generator_single, srr_generator_dual
from src_niv.prep_lf import normalize, resize_mri_volume
from src_niv.zssr import  zero_shot_super_resolution, extract_brain, extract_lf_volumes
from src_niv.models.subject_model import build_dual_encoder_unet
from src_niv.models.ResUNet import residual_srr_unet, residual_srr_att_dsunet
from src_niv.metrics import psnr, ssim, mse, composite_loss
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
matplotlib.use('tkagg')  # or 'Qt5Agg' depending on what's installed
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, Input
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import load_model

import ants
import nibabel as nib
from sklearn.feature_extraction import image
import os
from skimage.transform import resize  #Required for 3D resizing

def train(lf_input_volume, hf_input_volume, hf_target_volume,lf_input_volume_val,hf_input_volume_val,hf_target_volume_val,
            model_type, model_case, model_,subject,day_idx,steps_per_epoch = 50,
            epochs = 50,batch_size = 1,visualize_pairs = False):
    
    print("inside niv_srr_main_dual_encoder.py function ........................")
    output_path = f'./Data/Results/{model_type}/{subject}'
    os.makedirs(output_path, exist_ok=True)
    print(f"Output path: {output_path}")
    # #save the preprocessed volumes for reference as Nifti files in the output path with appropriate name and day index
    # nib.save(nib.Nifti1Image(lf_input_volume, affine=np.eye(4)), os.path.join(output_path, f'LF_input_volume_day{day_idx}.nii.gz'))
    # nib.save(nib.Nifti1Image(hf_target_volume, affine=np.eye(4)), os.path.join(output_path, f'HF_input_volume_day{day_idx}.nii.gz'))

    # if augmentation == True:
    #     lf_input_volume,hf_input_volume, hf_target_volume = crop_and_augment_3d(
    #     lf_input_volume, hf_input_volume, hf_target_volume,
    #     augment=True, extra_slices=300)

    #     print("After augmentation LF Input shape:", lf_input_volume.shape)
    #     print("After augmentation HF Input shape:", hf_input_volume.shape) 
    #     print("After augmentation HF volume shape:", hf_target_volume.shape)

    # gen = srr_generator(lf_input_volume, hf_target_volume, batch_size=batch_size, patch_z=32, augment=True, num_augmented_copies=6)
    # train_gen = srr_generator_dual(lf_input_volume,hf_input_volume, hf_target_volume, batch_size=batch_size, patch_z=32, patch_xy=128, augment=True, extra_slices=50, noise_sigma=0.02)
    # [lf_input,lf_input1], hf_target = next(train_gen)
    # print(lf_input.shape)  # (2, 32, 128, 128, 1)
    # print(lf_input1.shape)  # (2, 32, 128, 128, 1)
    # print(hf_target.shape)  # (2, 32, 128, 128, 1)
    train_gen = tf.data.Dataset.from_generator(
    lambda: srr_generator_dual(
        lf_input_volume, hf_input_volume, hf_target_volume,
        batch_size=2, patch_z=32, patch_xy=128, augment=True
    ),
    output_signature=(
        (
            tf.TensorSpec(shape=(None, 32, 128, 128, 1), dtype=tf.float32),  # LF input
            tf.TensorSpec(shape=(None, 32, 128, 128, 1), dtype=tf.float32),  # HF input
        ),
        tf.TensorSpec(shape=(None, 32, 128, 128, 1), dtype=tf.float32)       # HF target
    )
    )

    valid_gen = srr_generator_dual(lf_input_volume_val, hf_input_volume_val, hf_target_volume_val, batch_size=1, patch_z=32, patch_xy=128, augment=False, extra_slices=0, noise_sigma=0.02)
    # lf_input_val, hf_target_val = next(valid_gen)
    # print(lf_input_val.shape)  # (2, 32, 128, 128, 1)
    # print(hf_target_val.shape)  # (2, 32, 128, 128, 1)
    print("returned from generator ......................")
    # train_gen_patch = systematic_crop_generator(train_gen, target_shape=(32,64,64), stride=(32,64,64))
    # X_batch, Y_batch = next(train_gen_patch)
    # print("LF batch:", X_batch.shape)
    # print("HF batch:", Y_batch.shape)
    # # Expected output for batch_size=2:

    # import matplotlib.pyplot as plt

    # plt.subplot(1,2,1)
    # plt.imshow(X_batch[0,16,:,:,0], cmap='gray')  # LF middle slice
    # plt.title("LF")

    # plt.subplot(1,2,2)
    # plt.imshow(Y_batch[0,16,:,:,0], cmap='gray')  # HF middle slice
    # plt.title("HF")
    # plt.show()

    # Check if required volumes exist
    if 'lf_input_volume' not in locals() or 'hf_target_volume' not in locals():
        print("\nRequired volumes (super_resolved_volume and resampled_volume) not available. Cannot proceed.")
    else:
            
            # # Add batch and channel dimensions
            # # Input shapes for the model should be (Batch, H, W, D, C)
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

            # Define checkpoint path
            checkpoint_filepath_dual = os.path.join(output_path, f'{model_type}_model_checkpoint_day{day_idx}.keras')

            # --- Load or Build Model ---
            if os.path.exists(checkpoint_filepath_dual):
                print("Checkpoint path available ..........")
                # try:
                model = load_model(
                    checkpoint_filepath_dual,
                    # custom_objects={
                    #     'mse': tf.keras.losses.MeanSquaredError(),
                    #     'mean_squared_error': tf.keras.losses.MeanSquaredError(),
                    # },
                    compile=False  # continue training
                )
                print(f"\n✅ Loaded model from checkpoint: {checkpoint_filepath_dual}")
                # except Exception as e:
                #     print(f"\n⚠️ Failed to load checkpoint, rebuilding model. Error: {e}")
                #     if model_case == 'single_encoder_unet':
                #         model = model_(input_shape=(32,128,128,1))
                #     else:
                #         raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")
                #     print(f"\nBuilt {model_type} model for training.")
            else:
                if model_case == 'single_encoder_unet':
                    model = model_(input_shape=(32,128,128,1))
                elif model_case == 'multi_encoder_unet':
                    zssr_input_shape = (32, 128, 128, 1)  # Example ZSSR output shape
                    hf_input_shape = (32, 128, 128, 1)     # Example HF input shape
                    model = model_(zssr_input_shape, hf_input_shape)
                else:
                    raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")
                
                print(f"\nBuilt {model_type} model for training.")

            # --- Add checkpoint callback ---
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath_dual,
                save_weights_only=False,
                monitor='loss',
                mode='min',
                save_best_only=True
            )

            # --- Reduce Learning Rate on Plateau ---
            reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',            # monitor training loss
                factor=0.5,                # reduce LR by half
                patience=30,               # wait for 30 epochs with no improvement
                verbose=1,                 # print messages when LR changes
                mode='max',
                min_lr=1e-8                # optional, don't reduce below this
            )

            # --- Early Stopping ---
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=60,               # stop training if no improvement for 50 epochs
                verbose=1,
                mode='max',
                restore_best_weights=True  # restore weights from the best epoch
            )

            from tensorflow.keras.optimizers import Adam

            # Example: Adam with custom learning rate
            # optimizer = Adam(learning_rate=5.0000e-05)
            optimizer = Adam(learning_rate=0.001)

            # --- Compile ---
            model.compile(
                optimizer=optimizer,
                loss=composite_loss,
                metrics=[psnr, ssim, MeanSquaredError(name='mse')]
            )

            model.summary()
            # --- Build and Compile the Dual-Encoder Model ---


            # --- Train the CNN ---
            print("\nStarting encoder model training...")
            
            steps_per_epoch = steps_per_epoch # Number of steps per epoch
            epochs = epochs # Number of training epochs
            batch_size = batch_size # Batch size (1 for a single volume)


            print("training started ...............................")
            # Train the model using both inputs
            if model_case == 'multi_encoder_unet':
                history = model.fit(train_gen,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    callbacks=[model_checkpoint_callback, reduce_lr_callback],
                                    # validation_data=valid_gen,
                                    # validation_steps=1,
                                    verbose=1
                                    )
            else:
                raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")
            
            # Save final model explicitly after training
            final_model_path = os.path.join(output_path, f'{model_type}_model_checkpoint_final_day{day_idx}.keras')
            model.save(final_model_path)
            print(f"✅ Final model saved at: {final_model_path}")
            
            print("\nSingle-encoder model training finished.")

            # --- Load the Best Model from Checkpoint ---
            try:
                best_dual_model = load_model(
                    checkpoint_filepath_dual,
                    custom_objects={
                        'mse': tf.keras.losses.MeanSquaredError(),
                        'mean_squared_error': tf.keras.losses.MeanSquaredError()
                    },
                    compile=True  # compile=True if you want to continue training
                )
                print(f"Best dual-encoder model loaded from: {checkpoint_filepath_dual}")

            except Exception as e_custom:
                print(f"Loading with custom_objects failed: {e_custom}")
                print("Trying to load without compilation...")
                try:
                    best_dual_model = load_model(checkpoint_filepath_dual, compile=False)
                    print(f"Best dual-encoder model loaded for inference from: {checkpoint_filepath_dual}")
                except Exception as e_fail:
                    raise RuntimeError(f"Failed to load model even without compilation: {e_fail}")

            # # --- Predict the SRR Volume ---
            # if model_case == 'single_encoder_unet':
            #     # If you have the full LF volume as x_zssr_train
            #     predicted_volume_dual = best_dual_model.predict(x_zssr_train)
            # else:
            #     raise ValueError("Invalid model_case. Choose from 'single_encoder_unet'.")

            # # Remove batch and channel dimensions
            # predicted_volume_dual = np.squeeze(predicted_volume_dual)  # shape -> (Z, X, Y)
            # print(f"Predicted volume shape (dual-encoder): {predicted_volume_dual.shape}")

            # # --- Save the Predicted Volume as NIfTI ---
            # predicted_nifti = nib.Nifti1Image(predicted_volume_dual.astype(np.float32), affine=np.eye(4))
            # predicted_nifti_path = os.path.join(output_path, f'Predicted_volume_day{day_idx}.nii.gz')
            # nib.save(predicted_nifti, predicted_nifti_path)
            # print(f"Predicted SRR volume saved to: {predicted_nifti_path}")

            # print("\nDisplaying slices: Target HF vs. Predicted Dual-Encoder Output...")
            # # Display a few slices from the true HF and predicted volumes for comparison
            # if visualize_pairs == True:
            #     display_pred(hf_target_volume,lf_input_volume, predicted_volume_dual)
            #     print("\nProcessing complete.")

# if __name__ == "__main__":
#     train(lf_input_volume, hf_input_volume, hf_target_volume, model_type, model_case, model_,subject,
#            day_idx,steps_per_epoch = 3,epochs = 3,batch_size = 1,visualize_pairs = visualize_pairs)