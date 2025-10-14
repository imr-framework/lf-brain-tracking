import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
from src_niv.utils import display_pred, load_and_preprocess_hf, load_and_preprocess_lf, visualize_hf_slices,padding_LF, visualize_lf_slices,rotate_slices, visualize_resampled, resample_volume, visualize_planes,visualize_pair, normalize_volume
from SRR_SS_Mon.utils1 import srr_generator
from src_niv.prep_lf import normalize, resize_mri_volume
from src_niv.models.ResUNet import residual_srr_unet
from src_niv.metrics import psnr, ssim, mse, composite_loss
from src_niv.prep_lf import register_to_hf
from SRR_SS_Mon.utils1 import evaluate_model

import os
import cv2
import nibabel as nib
from nilearn import plotting
from nibabel.viewers import OrthoSlicer3D
import tensorflow as tf
import pydicom
import numpy as np
import matplotlib
matplotlib.use('Agg')  # or 'Qt5Agg' depending on what's installed
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, Input
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import load_model

import ants
import nibabel as nib
from sklearn.feature_extraction import image
import os
from skimage.transform import resize  # Required for 3D resizing

# def train(x_train, y_train, x_val, y_val, model_type, model_, steps_per_epoch = 50,
#           epochs = 50,batch_size = 1,visualize_pairs = False):
    
#     print("inside niv_srr_main_single.py function ........................")
#     output_path = f'./Data/Monash_results/{model_type}'
#     os.makedirs(output_path, exist_ok=True)
#     print(f"Output path: {output_path}")

#     train_gen = srr_generator(x_train, y_train, batch_size=batch_size, patch_size=None, full_slices=True, augment=False, noise_sigma=0.02)
#     lf_input, hf_target = next(train_gen)
#     print(lf_input.shape)  # (2, 32, 128, 128, 1)
#     print(hf_target.shape)  # (2, 32, 128, 128, 1)

#     valid_gen = srr_generator(x_val, y_val, batch_size=1, patch_size=None, full_slices=True, augment=False, noise_sigma=0.02)
#     lf_input_val, hf_target_val = next(valid_gen)
#     print(lf_input_val.shape)  # (2, 32, 128, 128, 1)
#     print(hf_target_val.shape)  # (2, 32, 128, 128, 1)
#     print("returned from generator ......................")
    

#     if 'lf_input_volume' not in locals() or 'hf_target_volume' not in locals():
#         print("\nRequired volumes (super_resolved_volume and resampled_volume) not available. Cannot proceed.")
#     else:
            
#             # # Add batch and channel dimensions
#             # # Input shapes for the model should be (Batch, H, W, D, C)
#             x_zssr_train = np.expand_dims(lf_input_volume, axis=0) # Add batch dim
#             x_zssr_train = np.expand_dims(x_zssr_train, axis=-1)           # Add channel dim (1 channel)

#             x_hf_train = np.expand_dims(hf_input_volume, axis=0)   # Add batch dim
#             x_hf_train = np.expand_dims(x_hf_train, axis=-1)             # Add channel dim (1 channel)

#             y_train = np.expand_dims(hf_target_volume, axis=0)   # Add batch dim
#             y_train = np.expand_dims(y_train, axis=-1)             # Add channel dim (1 channel)

#             if visualize_pairs == True:
#                 # Example: visualize slices 10, 20, 30
#                 visualize_pair(x_zssr_train, y_train, slice_indices=[1,2,3,4,5,6,7,8,12,16,18,20,21,22,23,24,26,28])

#             print(f"Prepared ZSSR training input shape: {x_zssr_train.shape}")
#             print(f"Prepared HF training input shape: {x_hf_train.shape}")
#             print(f"Prepared training target shape: {y_train.shape}")

#             # Define checkpoint path
#             checkpoint_filepath_dual = os.path.join(output_path, f'{model_type}_model_checkpoint_day{day_idx}.keras')

#             # --- Load or Build Model ---
#             if os.path.exists(checkpoint_filepath_dual):
#                 print("Checkpoint path available ..........")
#                 # try:
#                 model = load_model(
#                     checkpoint_filepath_dual,
#                     # custom_objects={
#                     #     'mse': tf.keras.losses.MeanSquaredError(),
#                     #     'mean_squared_error': tf.keras.losses.MeanSquaredError(),
#                     # },
#                     compile=False  # continue training
#                 )
#                 print(f"\n✅ Loaded model from checkpoint: {checkpoint_filepath_dual}")
#                 # except Exception as e:
#                 #     print(f"\n⚠️ Failed to load checkpoint, rebuilding model. Error: {e}")
#                 #     if model_case == 'single_encoder_unet':
#                 #         model = model_(input_shape=(32,128,128,1))
#                 #     else:
#                 #         raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")
#                 #     print(f"\nBuilt {model_type} model for training.")
#             else:
#                 if model_case == 'single_encoder_unet':
#                     model = model_(input_shape=(32,128,128,1))
#                 else:
#                     raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")
#                 print(f"\nBuilt {model_type} model for training.")

#             # --- Add checkpoint callback ---
#             model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#                 filepath=checkpoint_filepath_dual,
#                 save_weights_only=False,
#                 monitor='loss',
#                 mode='min',
#                 save_best_only=True
#             )

#             # --- Reduce Learning Rate on Plateau ---
#             reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
#                 monitor='val_loss',            # monitor training loss
#                 factor=0.5,                # reduce LR by half
#                 patience=30,               # wait for 30 epochs with no improvement
#                 verbose=1,                 # print messages when LR changes
#                 mode='min',
#                 min_lr=1e-8                # optional, don't reduce below this
#             )

#             # --- Early Stopping ---
#             early_stopping_callback = tf.keras.callbacks.EarlyStopping(
#                 monitor='val_ssim',
#                 patience=50,               # stop training if no improvement for 50 epochs
#                 verbose=1,
#                 mode='max',
#                 restore_best_weights=True  # restore weights from the best epoch
#             )
#             from tensorflow.keras.optimizers import Adam

#             # Example: Adam with custom learning rate
#             # optimizer = Adam(learning_rate=5.0000e-05)
#             optimizer = Adam(learning_rate=0.001)

#             # --- Compile ---
#             model.compile(
#                 optimizer=optimizer,
#                 loss=composite_loss,
#                 metrics=[psnr, ssim, MeanSquaredError(name='mse')]
#             )

#             model.summary()
#             # --- Build and Compile the Dual-Encoder Model ---

#             # # Input shapes for the model should match the shape of a single sample (H, W, D, C)
#             # if model_case == 'single_encoder_unet':
#             #     model = model_(input_shape=(32,128,128,1))
#             # else:
#             #     raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")
            
#             # print(f"\nBuilt {model_type} model for training.")
#             # # --- Compile the model ---
#             # model.compile(
#             #     optimizer='adam',
#             #     loss=composite_loss,
#             #     metrics=[psnr, ssim, MeanSquaredError(name='mse')])  # Added metrics her) # Mean Squared Error is common for regression
#             # # Mean Squared Error is common for regression
#             # model.summary()

#             # --- Train the CNN ---
#             print("\nStarting encoder model training...")
            
#             steps_per_epoch = steps_per_epoch # Number of steps per epoch
#             epochs = epochs # Number of training epochs
#             batch_size = batch_size # Batch size (1 for a single volume)


#             print("training started ...............................")
#             # Train the model using both inputs
#             if model_case == 'single_encoder_unet':
#                 history = model.fit(train_gen,
#                                     steps_per_epoch=steps_per_epoch,
#                                     epochs=epochs,
#                                     batch_size=batch_size,
#                                     callbacks=[model_checkpoint_callback, reduce_lr_callback, early_stopping_callback],
#                                     validation_data=valid_gen,
#                                     validation_steps=1,
#                                     verbose=1
#                                     )
#             else:
#                 raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")
            
#             # Save final model explicitly after training
#             final_model_path = os.path.join(output_path, f'{model_type}_model_checkpoint_final_day{day_idx}.keras')
#             model.save(final_model_path)
#             print(f"✅ Final model saved at: {final_model_path}")
            
#             print("\nSingle-encoder model training finished.")

#             # --- Load the Best Model from Checkpoint ---
#             try:
#                 best_dual_model = load_model(
#                     checkpoint_filepath_dual,
#                     custom_objects={
#                         'mse': tf.keras.losses.MeanSquaredError(),
#                         'mean_squared_error': tf.keras.losses.MeanSquaredError()
#                     },
#                     compile=True  # compile=True if you want to continue training
#                 )
#                 print(f"Best dual-encoder model loaded from: {checkpoint_filepath_dual}")

#             except Exception as e_custom:
#                 print(f"Loading with custom_objects failed: {e_custom}")
#                 print("Trying to load without compilation...")
#                 try:
#                     best_dual_model = load_model(checkpoint_filepath_dual, compile=False)
#                     print(f"Best dual-encoder model loaded for inference from: {checkpoint_filepath_dual}")
#                 except Exception as e_fail:
#                     raise RuntimeError(f"Failed to load model even without compilation: {e_fail}")

def train_and_evaluate(x_train, y_train,
                       x_val, y_val,
                       x_test=None, y_test=None,
                       model_type='single_encoder_unet',
                       model_fn=None,
                       steps_per_epoch=50,
                       epochs=50,
                       batch_size=1,
                       patch_size=None,
                       full_slices=True,
                       augment=False,
                       noise_sigma=0.02,
                       visualize_pairs=False,
                       day_idx=0,
                       composite_loss=None,
                       psnr=None,
                       ssim=None,
                       results_dir='./Data/Monash_results'):
    """
    Train a 3D SRR model and automatically evaluate on validation and test sets.

    Returns:
        model: trained Keras model
        history: training history
        val_metrics: PSNR/SSIM metrics on validation set
        test_metrics: PSNR/SSIM metrics on test set (None if not provided)
    """

    import os
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, model_type)
    os.makedirs(output_path, exist_ok=True)
    print(f"Output path: {output_path}")

    # --- Generators ---
    train_gen = srr_generator(x_train, y_train,
                              batch_size=batch_size,
                              patch_size=patch_size,
                              full_slices=full_slices,
                              augment=augment,
                              noise_sigma=noise_sigma)

    valid_gen = srr_generator(x_val, y_val,
                              batch_size=batch_size,
                              patch_size=patch_size,
                              full_slices=full_slices,
                              augment=False,
                              noise_sigma=0.0)

    # --- Sample batch to get input shape ---
    lf_input, _ = next(train_gen)
    input_shape = lf_input.shape[1:]  # (X,Y,Z,1)

    # --- Model building / loading ---
    checkpoint_filepath = os.path.join(output_path, f'{model_type}_checkpoint_day{day_idx}.keras')
    if os.path.exists(checkpoint_filepath):
        print("Loading model from checkpoint...")
        model = load_model(checkpoint_filepath, compile=False)
        print(f"✅ Loaded model from checkpoint: {checkpoint_filepath}")
    else:
        model = model_fn(input_shape=input_shape)
        print(f"✅ Built new model: {model_type} with input shape {input_shape}")

    # --- Callbacks ---
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                           monitor='val_loss',
                                           mode='min',
                                           save_best_only=True,
                                           save_weights_only=False,
                                           verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                             factor=0.5,
                                             patience=30,
                                             min_lr=1e-8,
                                             verbose=1,
                                             mode='min'),
        tf.keras.callbacks.EarlyStopping(monitor='val_ssim',
                                         patience=50,
                                         mode='max',
                                         restore_best_weights=True,
                                         verbose=1)
    ]

    # --- Compile ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    metrics_list = [MeanSquaredError(name='mse')]
    if psnr: metrics_list.append(psnr)
    if ssim: metrics_list.append(ssim)

    model.compile(optimizer=optimizer,
                  loss=composite_loss,
                  metrics=metrics_list)
    model.summary()

    # --- Train ---
    print("\nStarting training...")
    history = model.fit(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=valid_gen,
                        validation_steps=1,
                        callbacks=callbacks,
                        verbose=1)

    # --- Save final model ---
    final_model_path = os.path.join(output_path, f'{model_type}_final_day{day_idx}.keras')
    model.save(final_model_path)
    print(f"✅ Final model saved at: {final_model_path}")

    # --- Evaluate Validation Set ---
    val_save_path = os.path.join(output_path, 'val_preds')
    val_metrics, val_preds = evaluate_model(model, x_val, y_val,
                                            batch_size=batch_size,
                                            patch_size=patch_size,
                                            full_slices=full_slices,
                                            save_path=val_save_path,
                                            dataset_name='val')

    # --- Evaluate Test Set ---
    test_metrics = None
    if x_test is not None and y_test is not None:
        test_save_path = os.path.join(output_path, 'test_preds')
        test_metrics, test_preds = evaluate_model(model, x_test, y_test,
                                                  batch_size=batch_size,
                                                  patch_size=patch_size,
                                                  full_slices=full_slices,
                                                  save_path=test_save_path,
                                                  dataset_name='test')

    return model, history, val_metrics, test_metrics


