import sys
sys.path.append('./data_read_code')
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import os
import glob
import pydicom
from kea2nifti import make_nifti
# import re
from tensorflow.keras.models import load_model

from demo_read_data import read_lf_data
import json

# Define a dictionary mapping subject IDs to their corresponding lists

subject_data = {
    # Predefined subfolder sequences for specific subjects
    26184 : ["3DTSE/6", "3DTSE/9", "3DTSE/10", "3DTSE/8", "3DTSE/10"],  # 3
    30366 : ["3DTSE/4", "3DTSE/5", "3DTSE/8", "3DTSE/2", "3DTSE/8"],    # 3, 
    59175 : ["3DTSE/2", "3DTSE/4", "3DTSE/5", "3DTSE/9", "3DTSE/6"],    # 3, 4
    34507 : ["3DTSE/4", "3DTSE/4", "3DTSE/3", "3DTSE/4", "3DTSE/13"],   # 4,
    35547 : ["3DTSE/7", "3DTSE/4", "3DTSE/5", "3DTSE/14", "3DTSE/5"],   # 3
    59228 : ["3DTSE/8", "3DTSE/13", "3DTSE/12", "3DTSE/9", "3DTSE/13"], # 3, 4
    59877 : ["3DTSE/2", "3DTSE/2", "3DTSE/1", "3DTSE/2", "3DTSE/8"],    # Bad quality data
    59081 : ["3DTSE/2", "3DTSE/4", "3DTSE/7", "3DTSE/6", "3DTSE/2"],    # 3,4,5  --
    35528 : ["3DTSE/4", "3DTSE/2", "3DTSE/3", "3DTSE/8", "3DTSE/3"],    # v4, v5
    59233 : ["3DTSE/10", "3DTSE/6", "3DTSE/1", "3DTSE/10", "3DTSE/13"]  # Visit 3 near bright, visit 4 bad t2
}

# Access using variable

def process_subject(subject='26184', fix_wrap = True, wrap_around = int(2), fov_3T = [140, 140, 70]):
    """
    Process a specific subject's data and return the loaded dataset.
    Args:
        subject (str): Subject ID to process.
        fix_wrap (bool): Whether to fix wrap-around issues.
        wrap_around (int): Number of wrap-around slices to fix.
    Returns:
        lf_dataset (list): List of 3D volumes for the subject.
    """

    sub_list = subject_data[int(subject)]
    
    output_folder = 'Data/LFMRI_DATA_IRF_nifti'
    data_dir = 'Data/IRF_3T'
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    folders.sort()

    Figures_folder = 'Figures'
    Figures = os.path.join(Figures_folder, subject)
    if not os.path.exists(Figures):
        os.makedirs(Figures)

    search_string = subject

    print(f"Searching for subfolders containing '{search_string}' in each IRF folder...")

    irf_folders = [
        f for f in os.listdir('Data/LFMRI_DATA_IRF')
        if 'IRF_071E_2_C1_' in f and os.path.isdir(
            os.path.join('Data/LFMRI_DATA_IRF', f)
        )
    ]

    irf_folders.sort()
    visit = 0
    lf_dataset = []

    for irf_folder in irf_folders:
        folder_path = os.path.join('Data/LFMRI_DATA_IRF', irf_folder)
        subfolders = [sf for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))]
        matching_subfolders = [sf for sf in subfolders if search_string in sf]
        if matching_subfolders:
            for sf in matching_subfolders:
                # print(f"  {sf}")
                visit += 1
                data_folder = os.path.join(folder_path, sf)
                three_dtse_folders = [f for f in os.listdir(data_folder) if '3DTSE' in f and os.path.isdir(os.path.join(data_folder, f))]
                for t_folder in three_dtse_folders:
                    t_folder_path = os.path.join(data_folder, t_folder)
                    print(f"Processing Subject: {subject}, Visit: {visit}, Subfolder: {sf}, 3DTSE Folder: {t_folder}")
                    subfolders_3dtse = [subf for subf in os.listdir(t_folder_path) if os.path.isdir(os.path.join(t_folder_path, subf))]
                    subfolders_3dtse.sort()
                    for subf in subfolders_3dtse:
                        # print(f"        {subf}")
                        sub_folder = os.path.join(t_folder, subf)
                        Visit_id = f'V{visit:02d}'
                        
                        name = '&'.join([irf_folder, sf, Visit_id, t_folder, subf]) + '.nii.gz'
                        fig_name = '&'.join([irf_folder, sf, Visit_id, t_folder, subf]) + '.png'
                        print(f"Reading data for {name}...")
                        im, im_props = read_lf_data(data_folder, output_folder, subject, sub_folder, name)
                        
                        if im_props is not None:
                            fov_im = [im_props.res_dim1 * im.shape[0],  im_props.res_dim2 * im.shape[1], im_props.res_dim3 * im.shape[2]]
                            print(f"Field of View (mm): {fov_im}")
                            if wrap_around > 0 and fix_wrap:
                                # Fix wrap-around slices if needed
                                num_wrap_slices = int(im.shape[2] / wrap_around)
                                im = np.roll(im, -int(num_wrap_slices), axis=2)
                                print(f"Fixed wrap-around for {subject}, {Visit_id}, {sub_folder}")

                            # Check if the field of view is greater than 3T, if so, skip the number of slices to ensure FOV is constant across both fields
                            if  fov_im[2] > fov_3T[2]:
                                num_slices_to_skip = int((fov_im[2] - fov_3T[2]) / im_props.res_dim3)
                                im = im[:, :, num_slices_to_skip:] # skips the lower two slices closer to the throat of the monkey - matching 3T
                                print(f"Skipped {num_slices_to_skip} slices for {subject}, {Visit_id}, {sub_folder}")
                            
                        print(f"Loaded data for {subject}, {Visit_id}, {sub_folder}")
                        print(sub_list[visit-1])

                        if sub_folder == sub_list[visit-1]:
                            # fov_im = [im_props.res_dim1 * im.shape[0],  im_props.res_dim2 * im.shape[1], im_props.res_dim3 * im.shape[2]]
                            # if wrap_around > 0 and fix_wrap:
                            #     # Fix wrap-around slices if needed
                            #     num_wrap_slices = int(im.shape[2] / wrap_around)
                            #     im = np.roll(im, -int(num_wrap_slices), axis=2)
                            #     print(f"Fixed wrap-around for {subject}, {Visit_id}, {sub_folder}")

                            # # Check if the field of view is greater than 3T, if so, skip the number of slices to ensure FOV is constant across both fields
                            # if fov_im[0] > fov_3T[0] or fov_im[1] > fov_3T[1] or fov_im[2] > fov_3T[2]:
                            #     num_slices_to_skip = int((fov_im[2] - fov_3T[2]) / im_props.res_dim3)
                            #     im = im[:, :, :num_slices_to_skip-1:]
                            #     print(f"Skipped {num_slices_to_skip} slices for {subject}, {Visit_id}, {sub_folder}")
                            config_path = 'lf-mri.json'
                            entry = {
                                "subject": subject,
                                "visit": Visit_id,
                                "data_folder": data_folder,
                                "sub_folder": sub_folder
                            }
                            
                            lf_dataset.append(im)

                            # Make nifti in case of need for further inputs to other software
                            
                            subject_folder = os.path.join('Data/LFMRI_DATA_IRF_wrap1', subject)
                            if not os.path.exists(subject_folder):
                                os.makedirs(subject_folder)

                            fname_nii = os.path.join(subject_folder, name)
                            make_nifti(
                                im,
                                fname=fname_nii,
                                mask=False,
                                res=[im_props.res_dim1,  im_props.res_dim2, im_props.res_dim3],
                                dim_info=[0, 1, 2]
                            )
                            # Load existing config or create new
                            if os.path.exists(config_path):
                                with open(config_path, 'r') as f:
                                    config = json.load(f)
                            else:
                                config = []

                            config.append(entry)

                            with open(config_path, 'w') as f:
                                json.dump(config, f, indent=2)

                        if im is None:
                            print("Skipping due to read error.")
                            continue

                        print(np.max(np.abs(im)))
                        print("Min value:", np.min(np.abs(im)))
                        print("Data type of np.abs(im):", np.abs(im).dtype)
                        print("Shape of im:", im.shape)
                        
                        # save im in npy format
                        # np.save(f'./data/{subject}_{Visit_id}_{sub_folder}.npy', im)

                        # num_slices = im.shape[2]
                        # fig, axes = plt.subplots(2, 8, figsize=(20, 8))
                        # fig.suptitle(f'All Axial Slices for {name}\n{subject}\n{Visit_id}\n3DTSE/{subf}', fontsize=16)
                        # axes = axes.flatten()

                        # for i in range(16):
                        #     if i < num_slices:
                        #         slice_img = np.flipud(np.abs(im[:, :, i]).T)
                        #         axes[i].imshow(slice_img, cmap='gray')
                        #         axes[i].set_title(f'Slice {i + 1}')
                        #         axes[i].axis('off')
                        #     else:
                        #         axes[i].axis('off')

                        # plt.tight_layout()
                        # plt.savefig(f'Figures/{subject}/{fig_name}')
                        # plt.show()
                        # plt.close()

    return lf_dataset


import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, binary_fill_holes
from skimage.morphology import ball, binary_closing, binary_opening, remove_small_objects, label
from skimage.measure import regionprops

# ----------------------------
# Visualization Utility
# ----------------------------
def visualize_volume(volume, title="Volume Slices", cols=8):
    """Display all slices of a 3D volume in a grid."""
    num_slices = volume.shape[2]
    rows = int(np.ceil(num_slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(num_slices):
        axes[i].imshow(volume[:, :, i].T, cmap='gray', origin='lower')
        axes[i].set_title(f"Slice {i}")
        axes[i].axis('off')

    for j in range(num_slices, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_volume1(volume, title="Volume Slices", cols=8):
    """Display all slices of a 3D volume in a grid."""
    num_slices = volume.shape[0]
    rows = int(np.ceil(num_slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(num_slices):
        axes[i].imshow(volume[i, :, :].T, cmap='gray', origin='lower')
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
    # visualize_volume(subject_volume, "Original Volume")

    volume_shifted = shift_volume_circular(subject_volume, shift_x=shift_x, shift_y=shift_y)
    volume_transformed = rotate_volume_circular(volume_shifted, angle_deg=angle_deg)

    # transform = transform_volume(subject_volume, shift_x=12, shift_y=-0, rotate_deg=26)
    print("✅ Applied Shift and Rotation Transformations")
    # visualize_volume(volume_transformed, "Transformed Volume Before Resampling")

    # Step 1: Resample
    resampled = resample_volume(volume_transformed, original_spacing=orig_spacing, new_spacing=(1, 1, 2))
    visualize_volume(resampled, "Resampled Volume (1×1×2)")
  
    # Step 2: Head Extraction
    _ ,_ , head = extract_head_slices(resampled)
    visualize_volume(head, "Head Extracted Volume")

    # change axis from x,y,z to z,x,y
    transform = np.transpose(head, (2, 0, 1))
    visualize_volume1(transform, "Transposed Head Volume (Z,X,Y)")
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
    # for idx, angle in enumerate([10, 20, -15,60,-50], start=1):
    #     rotated = transform_volume(head, rotate_deg=angle)
    #     visualize_volume(rotated, f"Rotation Augmentation {idx}: {angle}°")


    print("✅ Processing Complete")
    return transform

# ----------------------------
# Example Usage
# ----------------------------
# process_and_visualize(first_tp)


# ---------------------------------------------------
# Example Call
# ---------------------------------------------------
# Suppose `first_tp` is your 3D MRI volume from your dataset
# Example:
# first_tp = lf_dataset[2]

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
        # visualize_pair(resampled_volume_lf_be_norm, resampled_volume_hf_norm, slice_indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
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

if __name__ == "__main__":
    
    # subjects = ['30366']
    subjects = os.listdir('Data/IRF_3T')
    #print subjects

    print(subjects)

    for subject in subjects:
        
        lf_dataset = process_subject(subject=subject)  # Returns list of 3D volumes (e.g., 5 timepoints)

        # print(f"Total timepoints loaded: {len(lf_dataset)}")
        # for i, volume in enumerate(lf_dataset):
        #     print(f"Timepoint {i+1} shape: {volume.shape}")

        # # Extract the third timepoint (index 2)
        # first_tp = lf_dataset[2]

        # print("Shape:", first_tp.shape)
        # h, w, d = first_tp.shape
        # print(f"Height (H): {h}, Width (W): {w}, Depth (D): {d}")

        
        head = process_and_visualize(first_tp, shift_x=12, shift_y=0, angle_deg=26)  #26184
        # head = process_and_visualize(first_tp, shift_x=12, shift_y=0, angle_deg=-9)  #30366
        # head = process_and_visualize(first_tp, shift_x=12, shift_y=0, angle_deg=-6)  #26184
        # # head = np.abs(head)
        # # Visualize the four slices by start and end to display and save in high dpi

        # #print the shape of head
        # print("Head shape:", head.shape)

        # # discard slices to make shape 32, 128,128
        # lf_input_volume = head[3:36, :, :]
        # print("Head shape after discarding slices:", head.shape)

        # #load model
        # # Define the path to the IRF_3T folder (High Field Data)
        # nhp_base_path = './Data/IRF_3T'
        # model_type = 'residual_srr_unet5_subjects_1500_d1'  # Options: 'single_encoder_unet', 'dual_encoder_unet', 'teacher_student_unet'
        # model_case = 'single_encoder_unet'
        # multi_subject_train = True

        # day_idx = 3
        # # unet4 as att resor resunet
        # # 20184,30366,35528,59081,59228
        # subject = '26184'  # Example subject number, adjust as needed 
        # # '59233' visit 1 can be evaluation
        # subject_train = '59228'
        # subject_train = '59228'

        # if multi_subject_train == False:
        #     output_path = f'./Data/Results/{model_type}/{subject_train}'
        #     predictions_dir = os.path.join(output_path, 'predictions')
        #     model_name = f'{model_type}_model_checkpoint_day2.keras'
        #     model_path = os.path.join(output_path, model_name)
        # else:
        #     output_path_model = f'./Data/Results/{model_type}/{subject_train}'
        #     # predictions_dir = os.path.join(output_path, 'predictions')
        #     output_path = f'./Data/Results/{model_type}'
        #     predictions_d = os.path.join(output_path, 'predictions')
        #     predictions_dir = f'{predictions_d}/{subject}'
        #     model_name = f'{model_type}_model_checkpoint_day2.keras'
        #     model_path = os.path.join(output_path_model, model_name)

        
        # df_metrics = predict_and_evaluate(
        #                     model_path=model_path,
        #                     volumes_hf_d1=lf_input_volume,        # dict or list with HF volumes
        #                     volumes_lf=lf_input_volume,     # dict or list with LF volumes
        #                     volumes_hf=lf_input_volume,          # dict or list with HF volumes
        #                     predictions_dir=predictions_dir,
        #                     visualize_fn=None                   # optional visualization function, if you have one
        #                     )
        # print("\nEvaluation complete. Metrics:")
        # print(df_metrics)

        # slice_indices = [5,6,7,8,9]  # Slices to display

        # num_slices = len(slice_indices)

        # fig = plt.figure(figsize=(23, 5))

        # # Parameters to control overlap (fraction of figure width per slice)
        # slice_width = 1.0 / num_slices
        # overlap = 0.1  # 0 = no overlap, 0.1 = 10% overlap

        # for idx, slice_idx in enumerate(slice_indices):
        #     # Left position of each axis
        #     left = idx * slice_width - idx * slice_width * overlap
        #     ax = fig.add_axes([left, 0, slice_width, 1])  # [left, bottom, width, height]
        #     ax.imshow(head[slice_idx, :,: ].T, cmap='gray', origin='lower')
        #     ax.set_title(f"Slice {slice_idx}", fontsize=10)
        #     ax.axis('off')

        # plt.suptitle(f"Head Extracted Slices for Subject {subject}", fontsize=16, y=1.05)
        # plt.savefig(f'Data/{subject}_head_extracted_slices_overlap.png', dpi=500, bbox_inches='tight', pad_inches=0)
        # plt.show()