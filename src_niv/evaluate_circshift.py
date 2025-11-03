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

visualize = False
visualize_pairs = False
padding = False
register2_hf = True
# Define the path to the IRF_3T folder (High Field Data)
nhp_base_path = './Data/IRF_3T'
model_type = 'residual_srr_unet4_subjects_500_d1'  # Options: 'single_encoder_unet', 'dual_encoder_unet', 'teacher_student_unet'
model_case = 'single_encoder_unet'
multi_subject_train = True

day_idx = 3
# unet4 as att resor resunet
# 20184,30366,35528,59081,59228
subject = '26184'  # Example subject number, adjust as needed 
# '59233' visit 1 can be evaluation
subject_train = '59228'

if multi_subject_train == False:
    output_path = f'./Data/Results/{model_type}/{subject_train}'
    predictions_dir = os.path.join(output_path, 'predictions')
    model_name = f'{model_type}_model_checkpoint_day2.keras'
    model_path = os.path.join(output_path, model_name)
else:
    output_path_model = f'./Data/Results/{model_type}/{subject_train}'
    # predictions_dir = os.path.join(output_path, 'predictions')
    output_path = f'./Data/Results/{model_type}'
    predictions_d = os.path.join(output_path, 'predictions')
    predictions_dir = f'{predictions_d}/{subject}'
    model_name = f'{model_type}_model_checkpoint_day2.keras'
    model_path = os.path.join(output_path_model, model_name)

os.makedirs(output_path, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)


# ----------------------------
# Visualization Utility
# ----------------------------
def visualize_volume(volume, title="Volume Slices", cols=8):
    """Display all slices of a 3D volume in a grid."""
    num_slices = volume.shape[0]
    rows = int(np.ceil(num_slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(num_slices):
        axes[i].imshow(volume[i,:, :].T, cmap='gray', origin='lower')
        axes[i].set_title(f"Slice {i}")
        axes[i].axis('off')

    for j in range(num_slices, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Resampling Function
# ----------------------------
def resample_volume(volume, original_spacing=(2, 2, 5), new_spacing=(1, 1, 2)):
    """Resample 3D volume to new voxel spacing using zoom."""
    zoom_factors = np.array(original_spacing) / np.array(new_spacing)
    resampled = zoom(volume, zoom_factors, order=1)
    print(f"Resampled shape: {resampled.shape}")
    return resampled

# ----------------------------
# Morphological Head Extraction
# ----------------------------
def extract_head(volume):
    """Extract head region using morphological operations."""
    thresh = np.percentile(volume, 20)
    mask = volume > thresh

    mask = binary_closing(mask, footprint=ball(3))
    mask = binary_opening(mask, footprint=ball(2))
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=500)

    labeled = label(mask)
    regions = regionprops(labeled)
    if regions:
        largest = max(regions, key=lambda r: r.area)
        mask = labeled == largest.label

    return volume * mask

def shift_volume_circular(volume, shift_x=0, shift_y=0):
    """Shift 3D volume without losing pixels (wrap-around)."""
    transformed = np.roll(volume, shift=shift_y, axis=0)  # vertical shift
    transformed = np.roll(transformed, shift=shift_x, axis=1)  # horizontal shift
    return transformed


def rotate_volume_circular(volume, angle_deg):
    """Rotate 3D volume slice-wise without losing pixels (wrap-around)."""
    h, w, d = volume.shape
    transformed = np.zeros_like(volume)
    center = np.array([w / 2, h / 2])

    for i in range(d):
        slice_img = volume[:, :, i]

        # Compute coordinates grid
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.stack([xs - center[0], ys - center[1]], axis=-1)

        # Rotation matrix
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])

        # Apply rotation
        rotated_coords = coords @ R.T
        rotated_coords += center

        # Wrap coordinates around the image size
        rotated_coords[..., 0] %= w  # x-axis wrap
        rotated_coords[..., 1] %= h  # y-axis wrap

        # Map rotated coordinates back to pixel values
        x_floor = np.floor(rotated_coords[..., 0]).astype(int)
        y_floor = np.floor(rotated_coords[..., 1]).astype(int)
        transformed[:, :, i] = slice_img[y_floor, x_floor]

    return transformed

# ----------------------------
# Transform Function (Shift + Rotate)
# ----------------------------
def transform_volume(volume, shift_x=0, shift_y=0, rotate_deg=0):
    """Apply affine shift and rotation to each slice of a 3D volume."""
    h, w, d = volume.shape
    transformed = np.zeros_like(volume)
    for i in range(d):
        im = volume[:, :, i]
        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(im, M_shift, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        center = (w // 2, h // 2)
        M_rot = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
        rotated = cv2.warpAffine(shifted, M_rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        transformed[:, :, i] = rotated
    return transformed

# ----------------------------
# Circular Shift Function
# ----------------------------
def circshift_volume(volume, shift_x=0, shift_y=0, shift_z=0):
    """Apply circular shift to a 3D volume (wrap-around translation)."""
    shifted = np.roll(volume, shift=(shift_y, shift_x, shift_z), axis=(0, 1, 2))
    return shifted

import cv2
import numpy as np
from tqdm import tqdm

def extract_head_slices(im):
    """
    Extract head/brain region slice-by-slice using morphological refinement.
    Equivalent to extract_brain(), but for 3D MRI volumes.
    Args:
        im (np.ndarray): 3D MRI volume, shape (H, W, D)
    Returns:
        orig_slices (list): Original normalized 2D slices
        head_slices (list): Brain-extracted slices
        head_volume (np.ndarray): Combined 3D head-extracted volume
    """
    print("==================================== Inside Head Extraction (Slice-wise) ================================")
    orig_slices = []
    head_slices = []
    mask_stack = []

    num_slices = im.shape[2]
    print(f"Processing {num_slices} slices for head extraction...")

    for i in tqdm(range(num_slices)):
        slice_data = im[:, :, i]

        if slice_data is None or slice_data.size == 0:
            print(f"Skipping slice {i} due to empty data.")
            head_slices.append(np.zeros_like(slice_data))
            mask_stack.append(np.zeros_like(slice_data))
            continue

        try:
            # Normalize slice to 8-bit
            normalized_slice = cv2.normalize(np.abs(slice_data), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            orig_slices.append(normalized_slice)

            # Binary mask using Otsu’s threshold
            _, thresh = cv2.threshold(normalized_slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            refined_mask = np.zeros_like(normalized_slice, dtype=np.uint8)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(normalized_slice, dtype=np.uint8)
                cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

                # Morphological refinement
                kernel_dilate = np.ones((10, 10), np.uint8)
                mask_dilated = cv2.dilate(mask, kernel_dilate, iterations=1)

                kernel_close = np.ones((5, 5), np.uint8)
                refined_mask = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel_close)

            # Apply mask to slice
            extracted_slice = cv2.bitwise_and(normalized_slice, normalized_slice, mask=refined_mask)
            head_slices.append(extracted_slice)
            mask_stack.append(refined_mask)

        except Exception as e:
            print(f"Error on slice {i}: {e}")
            head_slices.append(np.zeros_like(slice_data))
            mask_stack.append(np.zeros_like(slice_data))

    print("✅ Head extraction completed for all slices.")

    # Stack 2D slices back into a 3D volume
    head_volume = np.stack(head_slices, axis=2)
    return orig_slices, head_slices, head_volume


# ----------------------------
# Full Workflow
# ----------------------------
def process_and_visualize(subject_volume, orig_spacing=(2, 2, 5), shift_x=0, shift_y=0, angle_deg=0):
    print(f"Original shape: {subject_volume.shape}")
    visualize_volume(subject_volume, "Original Volume")

    # volume_shifted = shift_volume_circular(subject_volume, shift_x=shift_x, shift_y=shift_y)
    # volume_transformed = rotate_volume_circular(volume_shifted, angle_deg=angle_deg)

    transform = transform_volume(subject_volume, shift_x=12, shift_y=-0, rotate_deg=26)
    print("✅ Applied Shift and Rotation Transformations")
    # visualize_volume(volume_transformed, "Transformed Volume Before Resampling")

    # Step 1: Resample
    # resampled = resample_volume(volume_transformed, original_spacing=orig_spacing, new_spacing=(1, 1, 2))
    visualize_volume(transform, "Resampled Volume (1×1×2)")
  
    # Step 2: Head Extraction
    _ ,_ , head = extract_head_slices(resampled)
    visualize_volume(head, "Head Extracted Volume")

    # Step 3: Circular Shift before augmentations
    # circ_shifted = circshift_volume(head, shift_x=10, shift_y=-10, shift_z=10)
    # visualize_volume(circ_shifted, "Circularly Shifted Volume")

    #apply K-means clustring  to show segmentation of clusters
    # from sklearn.cluster import KMeans
    # h, w, d = transform.shape
    # flat_volume = transform.reshape(-1, 1)
    # kmeans = KMeans(n_clusters=3, random_state=0).fit(flat_volume)
    # clustered = kmeans.labels_.reshape(h, w, d)
    # visualize_volume(clustered, "K-means Clustering (3 clusters)")
    
    # Step 4: Rotation Augmentations
    for idx, angle in enumerate([10, 20, -15,60,-50], start=1):
        rotated = transform_volume(head, rotate_deg=angle)
        visualize_volume(rotated, f"Rotation Augmentation {idx}: {angle}°")

    print("✅ Processing Complete")
    return head

# Initialize data object and load data (HFMRI_data_IRF)
print(f"\n===============================HF_MRI data processing  started .............")
# Day1 HF data for model input
resampled_volume_hf_norm_D1 = load_and_preprocess_hf(subject, day_idx = 3, visualize = visualize)

# visualize all the slices of resampled_volume_hf_norm_D1 with z, x,y as stored in the volume
import matplotlib.pyplot as plt
import numpy as np

# # Assume your high-field volume is stored as hf_volume with shape (35, 128, 128)
# num_slices = resampled_volume_hf_norm_D1.shape[0]

# # Compute rows and columns for a nice grid
# cols = 7  # number of slices per row
# rows = int(np.ceil(num_slices / cols))

# fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))

# for i in range(rows * cols):
#     r, c = divmod(i, cols)
#     ax = axes[r, c] if rows > 1 else axes[c]
#     if i < num_slices:
#         ax.imshow(np.rot90(resampled_volume_hf_norm_D1[i, :, :].T), cmap='gray', origin='lower')
#         ax.set_title(f"Slice {i}", fontsize=8)
#     ax.axis('off')

# plt.suptitle("All High-Field MRI Slices", fontsize=16, y=0.92)
# plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=0.95, bottom=0)

# plt.savefig('Data/high_field_all_slices.png', dpi=400, bbox_inches='tight', pad_inches=0)
# plt.show()

    
# Current day HF data for target
resampled_volume_hf_norm = load_and_preprocess_hf(subject, day_idx, visualize)

# Load and preprocess LF data
print(f"\n=============================== LF_MRI data processing  started .............")
resampled_volume_lf_be_norm = load_and_preprocess_lf(subject, day_idx, visualize)

print("Resampled HF volume shape Day 1:", resampled_volume_hf_norm_D1.shape)
print("Resampled LF volume shape:", resampled_volume_lf_be_norm.shape)
print("Resampled HF volume shape:", resampled_volume_hf_norm.shape)

head = process_and_visualize(resampled_volume_lf_be_norm, shift_x=10, shift_y=0, angle_deg=10)  #26184

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
    from keras import config
    config.enable_unsafe_deserialization()
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
        print("\nVisualizing random slices from LF input, HF target, and Predicted HF ...")
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
        # Save predicted volume as NIfTI (identity affine here, update if needed)
        pred_path = os.path.join(predictions_dir, f"HF_volume_day{day_idx}.nii.gz")
        nib.save(nib.Nifti1Image(hf_vol.astype(np.float32), affine=np.eye(4)), pred_path)
        pred_path = os.path.join(predictions_dir, f"LF_volume_day{day_idx}.nii.gz")
        nib.save(nib.Nifti1Image(lf_vol.astype(np.float32), affine=np.eye(4)), pred_path)
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