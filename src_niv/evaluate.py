#Load data for LF and HF
#Load different models saved at first timepoint for each subject and evaluate on subsequent timepoints
#Evaluate on day 2,3,4,5
#save predicted volumes as NIfTI
#Compute PSNR, SSIM, MSE for each day compared to true HF
#Save results to CSV
#Visualize some slices from input LF, true HF, and predicted HF for qualitative assessment

# Evaluate any just give folder of model and make folder of subject inside prediction and evaluate based on the subject given

import os
import numpy as np
import nibabel as nib
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from tensorflow.keras.models import load_model
import os
import numpy as np
import nibabel as nib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# Read HF and LF data

import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./data_read_code')
from src_niv.prep_data import data_ops
from src_niv.read_lf5_data import process_subject
from src_niv.utils import visualize_hf_slices, visualize_lf_slices,rotate_slices, visualize_resampled, resample_volume, visualize_planes,visualize_pair, normalize_volume
from src_niv.prep_lf import normalize, resize_mri_volume
from src_niv.zssr import  zero_shot_super_resolution, extract_brain, extract_lf_volumes
from src_niv.models.subject_model import build_dual_encoder_unet
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
from src_niv.prep_lf import register_to_hf
from src_niv.utils import padding_LF
from src_niv.utils import load_and_preprocess_hf, load_and_preprocess_lf
from src_niv.utils import visualize_pair
import random

# Define the path to the IRF_3T folder (High Field Data)
nhp_base_path = './Data/IRF_3T'
model_type = 'residual_srr_unet5_subjects_500_d1'  # Options: 'single_encoder_unet', 'dual_encoder_unet', 'teacher_student_unet'
model_case = 'single_encoder_unet'
multi_subject_train = True

day_idx = 2

visualize = False
visualize_pairs = False
padding = False
register2_hf = True

subject = '26184'  # Example subject number, adjust as needed 
# '59233' visit 1 can be evaluation
subject_train = '59228'

if multi_subject_train == False:
    output_path = f'./Data/Results/{model_type}/{subject_train}'
    predictions_dir = os.path.join(output_path, 'predictions')
    model_name = f'{model_type}_model_checkpoint_day1.keras'
    model_path = os.path.join(output_path, model_name)
else:
    output_path_model = f'./Data/Results/{model_type}/{subject_train}'
    # predictions_dir = os.path.join(output_path, 'predictions')
    output_path = f'./Data/Results/{model_type}'
    predictions_d = os.path.join(output_path, 'predictions')
    predictions_dir = f'{predictions_d}/{subject}'
    model_name = f'{model_type}_model_checkpoint_day1.keras'
    model_path = os.path.join(output_path_model, model_name)

os.makedirs(output_path, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

# Initialize data object and load data (HFMRI_data_IRF)
print(f"\n===============================HF_MRI data processing  started .............")
# Day1 HF data for model input
resampled_volume_hf_norm_D1 = load_and_preprocess_hf(subject, day_idx = 1, visualize = visualize)
# Current day HF data for target
resampled_volume_hf_norm = load_and_preprocess_hf(subject, day_idx, visualize)

# Load and preprocess LF data
print(f"\n=============================== LF_MRI data processing  started .............")
resampled_volume_lf_be_norm = load_and_preprocess_lf(subject, day_idx, visualize)

print("Resampled HF volume shape Day 1:", resampled_volume_hf_norm_D1.shape)
print("Resampled LF volume shape:", resampled_volume_lf_be_norm.shape)
print("Resampled HF volume shape:", resampled_volume_hf_norm.shape)

if register2_hf == True:
    resampled_volume_lf_be_norm = register_to_hf(resampled_volume_lf_be_norm, resampled_volume_hf_norm_D1)
    resampled_volume_hf_norm = register_to_hf(resampled_volume_hf_norm, resampled_volume_hf_norm_D1)

if padding == True:
    resampled_volume_lf_be_norm = padding_LF(resampled_volume_lf_be_norm,resampled_volume_hf_norm, target_slices=64)
    print("After padding LF volume shape:", resampled_volume_lf_be_norm.shape) 

if visualize_pairs == True:
    # Example: visualize slices 10, 20, 30
    visualize_pair(resampled_volume_lf_be_norm, resampled_volume_hf_norm, slice_indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])

# ------ Final HF and LF inputs -----
hf_input_volume_D1 = resampled_volume_hf_norm_D1.astype(np.float32)
lf_input_volume = resampled_volume_lf_be_norm.astype(np.float32)
hf_target_volume = resampled_volume_hf_norm.astype(np.float32)

# Take slices 0:32 along (z, x, y) axis for each volume
hf_input_volume_d1 = hf_input_volume_D1[0:32, :, :]
lf_input_volume = lf_input_volume[0:32, :, :]
hf_target_volume = hf_target_volume[0:32, :, :]

# #save the preprocessed volumes for reference as Nifti files in the output path with appropriate name and day index
# nib.save(nib.Nifti1Image(lf_input_volume, affine=np.eye(4)), os.path.join(predictions_dir, f'LF_input_volume_day{day_idx}.nii.gz'))
# nib.save(nib.Nifti1Image(hf_target_volume, affine=np.eye(4)), os.path.join(predictions_dir, f'HF_input_volume_day{day_idx}.nii.gz'))

# --- Predict and evaluate ---  
def predict_and_evaluate(
    model_path,
    volumes_hf_d1,        # dict or list: day -> HF volume (D,H,W)
    volumes_lf,       # dict or list: day -> LF volume (D,H,W)
    volumes_hf,
    predictions_dir="predictions",
    visualize_fn=None       # optional function to visualize input-target pairs
):
    os.makedirs(predictions_dir, exist_ok=True)
    results = []
    # Load model once
    
    model = load_model(model_path, compile=False)
    print(f"Model Loaded from {model_path} ...")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"Model {model_name} loaded.")
    print(f"\nProcessing Day {day_idx} ...")

    hf_vol_d1 = volumes_hf_d1  # shape (D,H,W)
    hf_vol = volumes_hf  # shape (D,H,W)
    lf_vol = volumes_lf  # shape (D,H,W)
    
    # Crop or pad to first 32 slices along depth (adjust as needed)
    hf_vol_d1 = hf_vol_d1[0:32, :, :]  # shape (D,H,W)
    hf_vol = hf_vol[0:32, :, :]
    lf_vol = lf_vol[0:32, :, :]
    
    # Add batch and channel dims: (batch=1, depth, height, width, channel=1)
    x_hf_d1 = np.expand_dims(hf_vol_d1, axis=0)
    x_hf_d1 = np.expand_dims(x_hf_d1, axis=-1)

    x_lf = np.expand_dims(lf_vol, axis=0)
    x_lf = np.expand_dims(x_lf, axis=-1)
    
    x_hf = np.expand_dims(hf_vol, axis=0)
    x_hf = np.expand_dims(x_hf, axis=-1)
    
    # Prediction: model expects inputs like [LF_input, HF_input] (dual input)
    if model_case == 'single_encoder_unet':
        pred_vol = model.predict(x_lf)
    elif model_case == 'dual_encoder_unet':
        pred_vol = model.predict([x_lf, x_hf_d1])
    else:
        raise ValueError("Invalid model_type. Choose from 'single_encoder_unet', 'dual_encoder_unet'.")
    
    pred_vol = np.squeeze(pred_vol)  # remove batch & channel dims → (D,H,W)
    
    # Optional visualization function user-provided
    if visualize_fn is not None:
        visualize_fn(x_lf, np.expand_dims(np.expand_dims(hf_vol, 0), -1), slice_indices=[5, 10, 15, 20, 25])
        visualize_pair(resampled_volume_lf_be_norm, resampled_volume_hf_norm, slice_indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    else:
        # Built-in visualization: show 15 random non-overlapping slices
        total_slices = hf_vol.shape[0]
        num_slices_to_show = min(20, total_slices)
        np.random.seed(42)
        slice_indices = [int(i) for i in np.linspace(0, total_slices - 1, num_slices_to_show)]
        print(f"Visualizing slices: {slice_indices}")
        
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
        plt.savefig(os.path.join(predictions_dir, f"Day{day_idx}_slices.png"))
        plt.show()
    
        # Save predicted volume as NIfTI (identity affine here, update if needed)
        pred_path = os.path.join(predictions_dir, f"Predicted_volume_day{day_idx}.nii.gz")
        nib.save(nib.Nifti1Image(pred_vol.astype(np.float32), affine=np.eye(4)), pred_path)
        print(f"Saved prediction to: {pred_path}")
        
        # Compute metrics (make sure volumes are normalized or scaled consistently!)
        psnr_val = peak_signal_noise_ratio(hf_vol, pred_vol, data_range=hf_vol.max() - hf_vol.min())
        ssim_val = structural_similarity(hf_vol, pred_vol, data_range=hf_vol.max() - hf_vol.min())
        mse_val = mean_squared_error(hf_vol, pred_vol)
        
        results.append({
            "day": day_idx,
            "PSNR": psnr_val,
            "SSIM": ssim_val,
            "MSE": mse_val,
            # "prediction_path": pred_path
        })
        
        print(f"Day {day_idx} metrics: PSNR={psnr_val:.3f}, SSIM={ssim_val:.3f}, MSE={mse_val:.6f}")
    
    # Path for saving CSV
    csv_path = os.path.join(predictions_dir, f"{model_name}_metrics.csv")

    # Convert results (dict) into DataFrame (1 row)
    df = pd.DataFrame([results])  # wrap in list so it's one row

    # If CSV exists, append without headers
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

    print(f"\nMetrics appended to: {csv_path}")
    
    return df

# -----------------
# Usage example:
# -----------------

if visualize_pairs == True:
    # Visualize a range of slices from both volumes
    visualize_pair(lf_input_volume, hf_target_volume, slice_indices=list(range(0, 31)))

df_metrics = predict_and_evaluate(
    model_path=model_path,
    volumes_hf_d1=hf_input_volume_d1,        # dict or list with HF volumes
    volumes_lf=lf_input_volume,     # dict or list with LF volumes
    volumes_hf=hf_target_volume,          # dict or list with HF volumes
    predictions_dir=predictions_dir,
    visualize_fn=None                   # optional visualization function, if you have one
)
print("\nEvaluation complete. Metrics:")
print(df_metrics)