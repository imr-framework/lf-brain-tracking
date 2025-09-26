import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
from src_niv.utils import display_pred, load_and_preprocess_hf, load_and_preprocess_lf, visualize_hf_slices,padding_LF, visualize_lf_slices,rotate_slices, visualize_resampled, resample_volume, visualize_planes,visualize_pair, normalize_volume
from src_niv.augment import srr_generator_single
from src_niv.prep_lf import normalize, resize_mri_volume
from src_niv.models.ResUNet import residual_srr_unet
from src_niv.metrics import psnr, ssim, mse, composite_loss
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

from tensorflow.keras import layers, models, Input
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import load_model

import ants
import nibabel as nib
from sklearn.feature_extraction import image
import os
from skimage.transform import resize  # Required for 3D resizing

def train(x_train, y_train, x_val, y_val, model_type, model_, steps_per_epoch = 50,
          epochs = 50,batch_size = 1,visualize_pairs = False):
    
    print("inside niv_srr_main_single.py function ........................")
    output_path = f'./Data/monash_results/{model_type}/{subject}'
    os.makedirs(output_path, exist_ok=True)
    print(f"Output path: {output_path}")

    train_gen = srr_generator_single(x_train, y_train, batch_size=batch_size, patch_z=32, patch_xy=128, augment=True, extra_slices=50, noise_sigma=0.02)
    # lf_input, hf_target = next(train_gen)
    # print(lf_input.shape)  # (2, 32, 128, 128, 1)
    # print(hf_target.shape)  # (2, 32, 128, 128, 1)

    valid_gen = srr_generator_single(lf_input_volume_val, hf_target_volume_val, batch_size=1, patch_z=32, patch_xy=128, augment=False, extra_slices=0, noise_sigma=0.02)
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
                mode='min',
                min_lr=1e-8                # optional, don't reduce below this
            )

            # --- Early Stopping ---
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_ssim',
                patience=50,               # stop training if no improvement for 50 epochs
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

            # # Input shapes for the model should match the shape of a single sample (H, W, D, C)
            # if model_case == 'single_encoder_unet':
            #     model = model_(input_shape=(32,128,128,1))
            # else:
            #     raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")
            
            # print(f"\nBuilt {model_type} model for training.")
            # # --- Compile the model ---
            # model.compile(
            #     optimizer='adam',
            #     loss=composite_loss,
            #     metrics=[psnr, ssim, MeanSquaredError(name='mse')])  # Added metrics her) # Mean Squared Error is common for regression
            # # Mean Squared Error is common for regression
            # model.summary()

            # --- Train the CNN ---
            print("\nStarting encoder model training...")
            
            steps_per_epoch = steps_per_epoch # Number of steps per epoch
            epochs = epochs # Number of training epochs
            batch_size = batch_size # Batch size (1 for a single volume)


            print("training started ...............................")
            # Train the model using both inputs
            if model_case == 'single_encoder_unet':
                history = model.fit(train_gen,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    callbacks=[model_checkpoint_callback, reduce_lr_callback, early_stopping_callback],
                                    validation_data=valid_gen,
                                    validation_steps=1,
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