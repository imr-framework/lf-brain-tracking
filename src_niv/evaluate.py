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
day_idx = 2
visualize = True

def load_and_preprocess_hf(subject, day_idx, visualize=True):
    """
    Loads and preprocesses HF MRI data for a given subject and day index.
    Returns the normalized, resampled HF volume.
    """
    subjects = os.listdir(nhp_base_path)
    subjects = sorted(subjects)
    print(f"Available subjects: {subjects}")
    print(f"Selected subject: {subject}")

    nhp_data_path = f'{nhp_base_path}/{subject}'  # Construct the full path to the DICOM folder

    # Initialize data object and load data (IRF_3T)
    data_obj = data_ops(nhp_data_path)

    # Retrieve dictionary of 3D volumes (day1 to day5)
    all_volumes = data_obj.data
    voxel_sizes = data_obj.voxel_sizes

    volume_hf = all_volumes[day_idx]
    voxel_sizes_hf = voxel_sizes[day_idx]

    print(f'Day {day_idx} and voxel size {voxel_sizes_hf}')

    if visualize:
        print(f"Type: {type(volume_hf)}")
        print(f"Shape: {volume_hf.shape}")
        print(f"Dtype: {volume_hf.dtype}")
        print(f"Min: {np.min(volume_hf)}, Max: {np.max(volume_hf)}")
        print(f"Mean: {np.mean(volume_hf):.2f}, Std: {np.std(volume_hf):.2f}")
        visualize_hf_slices(all_volumes)
        # visualize_planes(all_volumes, voxel_sizes, day_idx)

    # Resample the HF volume
    # Define the new desired voxel spacing (z, y, x) in mm
    new_spacing = [2, 1.09, 1.09]  # z=2mm, y=1mm, x=1mm
    resampled_volume_hf = resample_volume(volume_hf, voxel_sizes_hf, new_spacing)
    resampled_volume_hf_norm = normalize_volume(resampled_volume_hf)
    if visualize:
        visualize_resampled(resampled_volume_hf_norm)
    print("High-field volume shape:", resampled_volume_hf_norm.shape)
    hf_input_volume = resampled_volume_hf_norm.astype(np.float32)
    return hf_input_volume

def load_and_preprocess_lf(subject, day_idx, visualize=True):
    """
    Loads and preprocesses LF MRI data for a given subject and day index.
    Returns the normalized, resampled LF volume.
    """
    # Initialize data object and load data (LFMRI_data_IRF)
    all_volumes_lf = process_subject(subject=subject)
    if visualize:
        visualize_lf_slices(all_volumes_lf)

    im = all_volumes_lf[day_idx-1]
    print(f"LF_MRI data processing started .............")

    # Define the new desired voxel spacing (z, y, x) in mm
    orig_spacing = [2, 2, 5]  # original spacing
    new_spacing = [1, 1, 2]   # desired spacing

    resampled_volume_lf = resample_volume(im, orig_spacing, new_spacing)
    resampled_volume_lf = rotate_slices(resampled_volume_lf)

    if visualize:
        visualize_resampled(resampled_volume_lf)

    resampled_volume_lf_be = extract_lf_volumes(resampled_volume_lf)
    resampled_volume_lf_be_norm = normalize_volume(resampled_volume_lf_be)
    lf_input_volume = resampled_volume_lf_be_norm.astype(np.float32)
    print("Low-field volume shape:", lf_input_volume.shape)
    return lf_input_volume



# --- Predict and evaluate ---  

def predict_and_evaluate(
    model_path,
    volumes_hf_d1,
    volume_hf,        # dict or list: day -> HF volume (D,H,W)
    volumes_lf,       # dict or list: day -> LF volume (D,H,W)
    days_to_process=None,  # list of days or None for all keys
    output_dir="SRR_models/predictions",
    visualize_fn=None       # optional function to visualize input-target pairs
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model once
    print(f"Loading model from {model_path} ...")
    model = load_model(model_path, compile=False)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    if days_to_process is None:
        days_to_process = sorted(all_volumes_hf.keys() if isinstance(all_volumes_hf, dict) else range(len(all_volumes_hf)))
    
    results = []
    
    for day in days_to_process:
        print(f"\nProcessing Day {day} ...")

        hf_vol = volumes_hf  # shape (D,H,W)
        lf_vol = volumes_lf  # shape (D,H,W)
        
        # Crop or pad to first 32 slices along depth (adjust as needed)
        hf_vol = hf_vol[0:32, :, :]
        lf_vol = lf_vol[0:32, :, :]
        
        # Add batch and channel dims: (batch=1, depth, height, width, channel=1)
        x_lf = np.expand_dims(lf_vol, axis=0)
        x_lf = np.expand_dims(x_lf, axis=-1)
        
        x_hf = np.expand_dims(hf_vol, axis=0)
        x_hf = np.expand_dims(x_hf, axis=-1)
        
        # Prediction: model expects inputs like [LF_input, HF_input] (dual input)
        pred_vol = model.predict([x_lf, x_hf])
        pred_vol = np.squeeze(pred_vol)  # remove batch & channel dims → (D,H,W)
        
        # Optional visualization function user-provided
        if visualize_fn is not None:
            visualize_fn(x_lf, np.expand_dims(np.expand_dims(hf_vol, 0), -1), slice_indices=[5, 10, 15, 20, 25])
        else:
            # Built-in visualization: show 15 random non-overlapping slices
            total_slices = hf_vol.shape[0]
            num_slices_to_show = min(15, total_slices)
            np.random.seed(42)
            slice_indices = np.random.choice(total_slices, size=num_slices_to_show, replace=False)
            
            fig, axes = plt.subplots(2, num_slices_to_show, figsize=(2*num_slices_to_show, 4))
            fig.suptitle(f"Day {day} - True (top) vs Predicted (bottom) HF Slices", fontsize=16)
            
            for i, idx in enumerate(slice_indices):
                # True slice
                axes[0, i].imshow(hf_vol[idx, :, :], cmap='gray')
                axes[0, i].set_title(f"Slice {idx}")
                axes[0, i].axis('off')
                
                # Predicted slice
                axes[1, i].imshow(pred_vol[idx, :, :], cmap='gray')
                axes[1, i].axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        
        # Save predicted volume as NIfTI (identity affine here, update if needed)
        pred_path = os.path.join(output_dir, f"{model_name}_day{day}.nii.gz")
        nib.save(nib.Nifti1Image(pred_vol.astype(np.float32), affine=np.eye(4)), pred_path)
        print(f"Saved prediction to: {pred_path}")
        
        # Compute metrics (make sure volumes are normalized or scaled consistently!)
        psnr_val = peak_signal_noise_ratio(hf_vol, pred_vol, data_range=hf_vol.max() - hf_vol.min())
        ssim_val = structural_similarity(hf_vol, pred_vol, data_range=hf_vol.max() - hf_vol.min())
        mse_val = mean_squared_error(hf_vol, pred_vol)
        
        results.append({
            "day": day,
            "PSNR": psnr_val,
            "SSIM": ssim_val,
            "MSE": mse_val,
            "prediction_path": pred_path
        })
        
        print(f"Day {day} metrics: PSNR={psnr_val:.3f}, SSIM={ssim_val:.3f}, MSE={mse_val:.6f}")
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nAll metrics saved to: {csv_path}")
    
    return df

# -----------------
# Usage example:
# -----------------
# Load and preprocess HF and LF volumes for the selected subject and day
hf_input_volume = load_and_preprocess_hf(subject, day_idx = 1, visualize=visualize) #HF day1 input
hf_input_volume_d2 = load_and_preprocess_hf(subject, day_idx, visualize=visualize) #HF day2 input
lf_input_volume_d2 = load_and_preprocess_lf(subject, day_idx, visualize=visualize) #LF day2 input

# For evaluation, HF target is the same as input (after normalization/resampling)
hf_target_volume = hf_input_volume

if visualize:
    # Visualize a range of slices from both volumes
    visualize_pair(lf_input_volume, hf_input_volume, slice_indices=list(range(0, 31)))

df_metrics = predict_and_evaluate(
    model_path="SRR_models/dual_encoder_unet_model.h5",
    volumes_hf_d1=hf_input_volume,        # dict or list with HF volumes
    volumes_lf=lf_input_volume,     # dict or list with LF volumes
    days_to_process=[2],         # optional: days to predict, or None for all
    visualize_fn=None                   # optional visualization function, if you have one
)

# Print summary for day 3 only:
print(df_metrics[df_metrics["day"] == 3])