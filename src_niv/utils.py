import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./data_read_code')
from src_niv.prep_data import data_ops
from src_niv.read_lf5_data import process_subject
from demo_read_data import read_lf_data
from src_niv.zssr import  zero_shot_super_resolution, extract_brain, extract_lf_volumes

import os
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
from scipy.ndimage import zoom
print(tf.__version__)
# import tensorflow_addons as tfa


# Define the path to the IRF_3T folder ( High Field Data)
nhp_base_path = './Data/IRF_3T'

def load_and_preprocess_hf(subject, day_idx, visualize=True):
    """
    Loads and preprocesses HF MRI data for a given subject and day index.
    Returns the normalized, resampled HF volume.
    """
    # Initialize data object and load data (LFMRI_data_IRF)
    print(f"===============================\n\nInside utils load_and_preprocess_hf data processing day {day_idx} .............\n")
    print(f"\n====================This Function loads all day data and returns specific day========")
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

    print(f'========================HF-MRI: Day {day_idx} and voxel size {voxel_sizes_26184} ==================')

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
    print("High-field volume shape:", resampled_volume_hf_norm.shape)
    hf_input_volume = resampled_volume_hf_norm.astype(np.float32)
    return hf_input_volume

def load_and_preprocess_lf(subject, day_idx, visualize=True):
    """
    Loads and preprocesses LF MRI data for a given subject and day index.
    Returns the normalized, resampled LF volume.
    """
    # Initialize data object and load data (LFMRI_data_IRF)
    print(f"\n\n===============================Inside utils load_and_preprocess_lf data processing {day_idx} .............\n")
    print(f"\n====================This Function loads all day data and returns specific day========")

    all_volumes_lf = process_subject(subject=subject)
    if visualize == True:
        visualize_lf_slices(all_volumes_lf)

    im = all_volumes_lf[day_idx-1]

    # Define the new desired voxel spacing (z, y, x) in mm
    orig_spacing = [2, 2, 5] # z=2mm, y=1mm, x=1mm
    new_spacing = [1, 1, 2] # z=2mm, y=1mm, x=1mm

    resampled_volume_lf = resample_volume(im, orig_spacing, new_spacing)
    resampled_volume_lf = rotate_slices(resampled_volume_lf)

    if visualize == True:
        visualize_resampled(resampled_volume_lf)

    resampled_volume_lf_be = extract_lf_volumes(resampled_volume_lf)
    resampled_volume_lf_be_norm = normalize_volume(resampled_volume_lf_be)
    lf_input_volume = resampled_volume_lf_be_norm.astype(np.float32)
    print("Low-field volume shape:", lf_input_volume.shape)

    return lf_input_volume

def dicom_info(ds):

    print("\n======================================Inside utils ------------------------------------------------")
    # Shape of the pixel array (rows, columns)
    print("Pixel Array Shape:", ds.pixel_array.shape)

    # Type and value of pixel data
    print("Pixel Data Type:", ds.pixel_array.dtype)

    print("Rows:", ds.Rows)
    print("Columns:", ds.Columns)

    # Pixel spacing (in mm) — typically [row_spacing, col_spacing]
    if 'PixelSpacing' in ds:
        print("Pixel Spacing:", ds.PixelSpacing)

    # Slice thickness (in mm), relevant in 3D scans
    if 'SliceThickness' in ds:
        print("Slice Thickness:", ds.SliceThickness)

    # Image Position Patient (coordinates of upper-left corner of the image)
    if 'ImagePositionPatient' in ds:
        print("Image Position (Patient):", ds.ImagePositionPatient)

    # Image Orientation Patient (direction cosines of row and column axes)
    if 'ImageOrientationPatient' in ds:
        print("Image Orientation (Patient):", ds.ImageOrientationPatient)
    # Instance Number (DICOM slice index, useful for sorting)
    if 'InstanceNumber' in ds:
        print("Instance Number:", ds.InstanceNumber)
    print("exit utils ------------------------------------------------")

def visualize_planes(all_volumes, voxel_sizes, day_idx):
    print("\n================================Inside utils ........")
    
    if day_idx not in all_volumes:
        print(f"Day {day_idx} data not found.")
        return
    
    vol = all_volumes[day_idx]  # shape assumed (Z, X, Y)
    
    z_mid = vol.shape[0] // 2  # axial slice index
    x_mid = vol.shape[1] // 2  # sagittal slice index
    y_mid = vol.shape[2] // 2  # coronal slice index
    
    axial = vol[z_mid, :, :]       # shape (X, Y)
    sagittal = vol[:, x_mid, :]    # shape (Z, Y)
    coronal = vol[:, :, y_mid]     # shape (Z, X)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Day {day_idx} - Central Slices (Z,X,Y={vol.shape})", fontsize=14)
    
    # Axial: transpose and flip vertically for correct orientation
    axes[0].imshow(np.flipud(axial.T), cmap='gray')
    axes[0].set_title("Axial (Z fixed)")
    axes[0].axis('off')
    
    # Coronal: transpose and flip vertically
    axes[1].imshow(np.flipud(coronal.T), cmap='gray')
    axes[1].set_title("Coronal (Y fixed)")
    axes[1].axis('off')
    
    # Sagittal: transpose and flip vertically
    axes[2].imshow(np.flipud(sagittal.T), cmap='gray')
    axes[2].set_title("Sagittal (X fixed)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


import numpy as np
import tensorflow as tf

# Rotate 2D slice
def rotate2d(img, angle_rad):
    frac = angle_rad / (2.0 * np.pi)
    rr = tf.keras.layers.RandomRotation(factor=(frac, frac), fill_mode="nearest")
    img = tf.expand_dims(img, axis=0)  # add batch
    out = rr(img)                      # (1,H,W,C)
    return out[0]                      # (H,W,C)

# Shear 2D slice using ImageProjectiveTransformV3 (no TFA)
def shear2d(img, shear_x, shear_y):
    X, Y, C = img.shape
    # Shear matrix
    transform = [1.0, shear_x, 0.0,
                 shear_y, 1.0, 0.0,
                 0.0,     0.0]
    transform = tf.convert_to_tensor([transform], dtype=tf.float32)

    # ImageProjectiveTransformV3 expects (N,H,W,C)
    img = tf.expand_dims(img, axis=0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img,
        transforms=transform,
        output_shape=[X, Y],
        interpolation="BILINEAR"
    )
    return out[0]  # (H,W,C)


def rotate2d(img, angle_deg):
    """Rotate a single 2D slice."""
    angle_rad = angle_deg * np.pi / 180
    frac = angle_rad / (2.0 * np.pi)
    rr = tf.keras.layers.RandomRotation(factor=(frac, frac), fill_mode="nearest")
    img = tf.expand_dims(img, axis=0)
    out = rr(img)
    return out[0]

def shear2d(img, shear_x=0.1, shear_y=0.1):
    """Approximate shear with RandomTranslation."""
    img = tf.expand_dims(img, axis=0)
    layer = tf.keras.layers.RandomTranslation(height_factor=shear_y, width_factor=shear_x, fill_mode="nearest")
    out = layer(img)
    return out[0]

def srr_generator(lf_vol, hf_vol, batch_size=1, patch_z=32, augment=True, extra_slices=0):
    """
    On-the-fly generator for 3D SRR patches with z,x,y input shape.
    lf_vol, hf_vol: full volumes (Z, X, Y)
    batch_size: patches per batch
    patch_z: number of slices per patch
    augment: apply augmentation
    extra_slices: number of augmented slices per patch
    """
    Z, X, Y = lf_vol.shape
    
    while True:
        batch_lf = []
        batch_hf = []

        for _ in range(batch_size):
            # Random start along Z
            if Z > patch_z:
                start = np.random.randint(0, Z - patch_z + 1)
            else:
                start = 0
            lf_patch = lf_vol[start:start+patch_z].copy()
            hf_patch = hf_vol[start:start+patch_z].copy()

            # Stack channels for augmentation
            vol_stack = np.stack([lf_patch, hf_patch], axis=-1)  # shape (Z_patch, X, Y, 2)

            augmented_slices = []

            if augment and extra_slices > 0:
                for _ in range(extra_slices):
                    idx = np.random.randint(0, vol_stack.shape[0])
                    slice_aug = vol_stack[idx].copy()

                    # flips
                    if np.random.rand() > 0.5:
                        slice_aug = np.flip(slice_aug, axis=0)
                    if np.random.rand() > 0.5:
                        slice_aug = np.flip(slice_aug, axis=1)

                    # rotation
                    if np.random.rand() > 0.5:
                        angle = np.random.uniform(-15, 15)
                        slice_aug = rotate2d(slice_aug, angle).numpy()

                    # shear
                    if np.random.rand() > 0.5:
                        sx = np.random.uniform(-0.2, 0.2)
                        sy = np.random.uniform(-0.2, 0.2)
                        slice_aug = shear2d(slice_aug, sx, sy).numpy()

                    # random center crop + resize
                    if np.random.rand() > 0.5:
                        crop_frac = np.random.uniform(0.7, 0.95)
                        cx, cy = int(X * crop_frac), int(Y * crop_frac)
                        x0, y0 = (X - cx)//2, (Y - cy)//2
                        cropped = slice_aug[x0:x0+cx, y0:y0+cy, :]
                        slice_aug = tf.image.resize(cropped, (X, Y)).numpy()

                    # intensity scaling LF only
                    scale = 0.9 + 0.2 * np.random.rand()
                    slice_aug[...,0] *= scale

                    # gaussian noise LF only
                    if np.random.rand() > 0.5:
                        noise = np.random.normal(0, 0.01, size=slice_aug[...,0].shape)
                        slice_aug[...,0] += noise

                    slice_aug = slice_aug[None, ...]  # add Z dim
                    augmented_slices.append(slice_aug)

                # concatenate along Z
                vol_stack = np.concatenate([vol_stack] + augmented_slices, axis=0)

            # Randomly sample patch_z slices if Z changed
            if vol_stack.shape[0] > patch_z:
                idxs = np.random.choice(vol_stack.shape[0], patch_z, replace=False)
                vol_stack = vol_stack[idxs]

            # Split back
            lf_patch_out = vol_stack[..., 0]
            hf_patch_out = vol_stack[..., 1]

            # add channel axis for Keras
            batch_lf.append(lf_patch_out[..., np.newaxis])
            batch_hf.append(hf_patch_out[..., np.newaxis])

        yield np.stack(batch_lf, axis=0), np.stack(batch_hf, axis=0)

def padding_LF(resampled_volume_lf_be_norm,resampled_volume_hf_norm, target_slices=64):
    #If Shape of HF and LF not same then Zero padding to low field
    # Check if the (y, x) shapes match; if not, pad LF to match HF
    print(f"\n\n=================================Inside utils padding LF........{lf_shape[0]}, {lf_shape[1]}, {lf_shape[2]} vs HF: {hf_shape[0]}, {hf_shape[1]}, {hf_shape[2]}")

    lf_shape = resampled_volume_lf_be_norm.shape
    hf_shape = resampled_volume_hf_norm.shape

    if lf_shape[1] != hf_shape[1] or lf_shape[2] != hf_shape[2]:
        pad_y = hf_shape[1] - lf_shape[1]
        pad_x = hf_shape[2] - lf_shape[2]
        # Only pad if needed (if pad_y or pad_x > 0)
        pad_before_y = pad_y // 2 if pad_y > 0 else 0
        pad_after_y = pad_y - pad_before_y if pad_y > 0 else 0
        pad_before_x = pad_x // 2 if pad_x > 0 else 0
        pad_after_x = pad_x - pad_before_x if pad_x > 0 else 0
        # Pad along (z, y, x): only y and x
        resampled_volume_lf_be_norm = np.pad(
            resampled_volume_lf_be_norm,
            ((0, 0), (pad_before_y, pad_after_y), (pad_before_x, pad_after_x)),
            mode='constant'
        )
        print(f"LF volume padded: new shape {resampled_volume_lf_be_norm.shape}")
        return resampled_volume_lf_be_norm
    else:
        print("LF and HF volumes already have matching (y, x) shapes.")
    

def visualize_hf_slices(all_volumes, voxel_sizes=None):
    """
    Displays 10 equally spaced slices for each day's HF volume.
    All days are shown in a single figure: one row per day,
    with the day label centered above each row.
    """
    print("\n=================================Inside utils ........")
    
    day_indices = sorted(all_volumes.keys())
    num_days = len(day_indices)
    num_slices = 10  # slices per day
    
    fig, axes = plt.subplots(num_days, num_slices, figsize=(2*num_slices, 2*num_days))
    fig.suptitle("HF MRI Volumes - 10 Sample Slices per Day", fontsize=18)
    
    for row, day_idx in enumerate(day_indices):
        vol = all_volumes[day_idx]
        total_slices = vol.shape[0]
        
        slice_indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
        
        for col, idx in enumerate(slice_indices):
            ax = axes[row, col] if num_days > 1 else axes[col]
            ax.imshow(vol[idx, :, :], cmap='gray')
            
            if row == 0:
                ax.set_title(f"Slice {idx}", fontsize=8)
            
            ax.axis('off')
        
        # Add day label spanning all columns in that row
        mid_col = num_slices // 2
        axes[row, mid_col].set_xlabel(f"Day {day_idx}", fontsize=12, labelpad=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def visualize_resampled(resampled_volume):
    
    if resampled_volume is not None:
        print(f"\n==============================Displaying slices from the resampled volume (shape: {resampled_volume.shape})...")

        depth, height, width = resampled_volume.shape
        num_slices_to_show = min(16, depth)

        if num_slices_to_show > 0:
            step = max(1, depth // num_slices_to_show)
            slice_indices_to_show = list(range(0, depth, step))[:num_slices_to_show]

            # Increase figsize so each slice is wider
            fig_width = num_slices_to_show * 2   # 2 inches per slice
            fig_height = 4
            fig, axes = plt.subplots(1, num_slices_to_show, figsize=(fig_width, fig_height))
            fig.suptitle("Slices from Resampled Volume", fontsize=16)

            if num_slices_to_show == 1:
                axes = [axes]

            for i, slice_idx in enumerate(slice_indices_to_show):
                ax = axes[i]
                ax.imshow(resampled_volume[slice_idx, :, :], cmap='gray')
                ax.set_title(f"Slice {slice_idx}")
                ax.axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        else:
            print("Resampled volume has no slices to display.")
    else:
        print("Resampled volume variable not found. Please run the resampling cell first.")

def visualize_lf_slices(all_volumes_lf):
    """
    Displays all slices for each day's LF volume from a list.
    One row per day in a single figure.
    """
    print("\n=======================================Inside utils LF........")

    num_days = len(all_volumes_lf)

    # Determine max slices among all days
    max_slices = max(im.shape[2] for im in all_volumes_lf)

    fig, axes = plt.subplots(num_days, max_slices, figsize=(max_slices * 1.5, num_days * 2))
    fig.suptitle("LF MRI Volumes - All Slices per Day", fontsize=18)

    for row, im in enumerate(all_volumes_lf):
        num_slices = im.shape[2]

        for col in range(max_slices):
            ax = axes[row, col] if num_days > 1 else axes[col]

            if col < num_slices:
                slice_img = np.abs(im[:, :, col])
                ax.imshow(slice_img, cmap='gray')
            else:
                ax.axis('off')

            if row == 0:
                ax.set_title(f"Slice {col+1}", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"Day {row+1}", fontsize=10)

            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# def visualize_lf_slices(im):
#     print("inside utils LF........")

#     # print("Max value:", np.max(np.abs(im)))
#     # print("Min value:", np.min(np.abs(im)))
#     # print("Data type of np.abs(im):", np.abs(im).dtype)
#     # print("Shape of im:", im.shape)

#     num_slices = im.shape[2]
#     fig, axes = plt.subplots(2, 8, figsize=(20, 8))
#     # fig.suptitle(f'All Axial Slices for {name}\n{subject}\n{Visit_id}\n3DTSE/{subf}', fontsize=16)
#     axes = axes.flatten()

#     for i in range(16):
#         if i < num_slices:
#             # slice_img = np.flipud(np.abs(im[:, :, i]).T)
#             slice_img = np.abs(im[:, :, i])
#             axes[i].imshow(slice_img, cmap='gray')
#             axes[i].set_title(f'Slice {i + 1}')
#             axes[i].axis('off')
#         else:
#             axes[i].axis('off')

#     plt.tight_layout()
#     # plt.savefig(f'Figures/{subject}/{fig_name}')
#     plt.show()
#     plt.close()

# Voxel size reduction
def resample_volume(volume, voxel_sizes_26184, new_spacing):
    """
    Resample a 3D volume to new voxel spacing.

    Parameters:
        volume: np.ndarray (3D) - shape (z, y, x)
        original_spacing: list or tuple of [z, y, x] spacing in mm
        new_spacing: list or tuple of [z, y, x] desired spacing in mm

    Returns:
        Resampled volume as np.ndarray
    """
    # Ensure volumes are float for interpolation
    original_spacing = [voxel_sizes_26184[0], voxel_sizes_26184[1], voxel_sizes_26184[2]]
    print(f"Inferred original spacing for (x, y, z) volume: {original_spacing} mm")
    volume = volume.astype(np.float64)

    zoom_factors = [
        original_spacing[0] / new_spacing[0],  # z factor
        original_spacing[1] / new_spacing[1],  # y factor
        original_spacing[2] / new_spacing[2],  # x factor
    ]
    print(f"Original volume shape: {volume.shape}")
    print(f"Original spacing (x, y, z): {original_spacing}")
    print(f"New spacing (x, y, z): {new_spacing}")
    print(f"Zoom factors (x, y, z): {zoom_factors}")

    # Apply zoom with linear interpolation
    # Ensure that the axes are consistent (z, y, x) for both spacing and volume
    # If your volume is (slices, height, width), this corresponds to (z, y, x)
    resampled = zoom(volume, zoom=zoom_factors, order=1)  # order=1 = linear interpolation

    print(f"Resampled volume shape: {resampled.shape}")

    return resampled

# Voxel size reduction
def resample_volume_lf(volume, original_spacing, new_spacing):
    """
    Resample a 3D volume to new voxel spacing.

    Parameters:
        volume: np.ndarray (3D) - shape (z, y, x)
        original_spacing: list or tuple of [z, y, x] spacing in mm
        new_spacing: list or tuple of [z, y, x] desired spacing in mm

    Returns:
        Resampled volume as np.ndarray
    """
    # Ensure volumes are float for interpolation
    # original_spacing = [voxel_sizes_26184[0], voxel_sizes_26184[1], voxel_sizes_26184[2]]
    print(f"Inferred original spacing for (x, y, z) volume: {original_spacing} mm")
    volume = volume.astype(np.float64)

    zoom_factors = [
        original_spacing[0] / new_spacing[0],  # z factor
        original_spacing[1] / new_spacing[1],  # y factor
        original_spacing[2] / new_spacing[2],  # x factor
    ]
    print(f"Original volume shape: {volume.shape}")
    print(f"Original spacing (x, y, z): {original_spacing}")
    print(f"New spacing (x, y, z): {new_spacing}")
    print(f"Zoom factors (x, y, z): {zoom_factors}")

    # Apply zoom with linear interpolation
    # Ensure that the axes are consistent (z, y, x) for both spacing and volume
    # If your volume is (slices, height, width), this corresponds to (z, y, x)
    resampled = zoom(volume, zoom=zoom_factors, order=1)  # order=1 = linear interpolation

    print(f"Resampled volume shape: {resampled.shape}")

    return resampled

def normalize_volume(resized_zssr_volume):
    zssr_min, zssr_max = np.min(resized_zssr_volume), np.max(resized_zssr_volume)
    print(f" BEFORE norm: min={zssr_min:.6f}, max={zssr_max:.6f}")
    if zssr_max - zssr_min > 1e-6:
        resized_zssr_volume_norm = (resized_zssr_volume - zssr_min) / (zssr_max - zssr_min)
    else:
        resized_zssr_volume_norm = np.zeros_like(resized_zssr_volume)
    print(f" AFTER  norm: min={np.min(resized_zssr_volume_norm):.6f}, max={np.max(resized_zssr_volume_norm):.6f}")

    return resized_zssr_volume_norm

def rotate_slices(volume_3d):
    processed = []
    for i in range(volume_3d.shape[2]):
        s = volume_3d[:, :, i]
        s_proc = np.flipud(s.T)
        processed.append(s_proc)
    return np.stack(processed)

def visualize_pair(x_vol, y_vol, slice_indices):
    """
    Visualize same slice numbers from x_vol and y_vol.
    Assumes shape: (1, X, Y, Z, 1)  -> batch=1, channel=1
    First row: x_vol slices
    Second row: y_vol slices
    """
    # Remove batch & channel dims
    x_vol = np.squeeze(x_vol)  # shape (X, Y, Z)
    y_vol = np.squeeze(y_vol)  # shape (X, Y, Z)
    
    num_slices = len(slice_indices)
    fig, axes = plt.subplots(2, num_slices, figsize=(3 * num_slices, 4))
    
    for i, idx in enumerate(slice_indices):
        # Row 1: X slices
        # axes[0, i].imshow(x_vol[:, :, idx], cmap='gray')
        axes[0, i].imshow(np.abs(x_vol[idx,:, :]), cmap='gray')
        axes[0, i].set_title(f"X slice {idx}")
        axes[0, i].axis('off')
        
        # Row 2: Y slices
        axes[1, i].imshow(y_vol[idx, :, :], cmap='gray')
        axes[1, i].set_title(f"Y slice {idx}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def display_pred(hf_vol, lf_vol, pred_vol, day_idx=None, num_slices_to_show=15, random_seed=42):
    """
    Visualize true HF, LF, and predicted HF slices side by side.

    Parameters:
        hf_vol (np.ndarray): High-field (ground truth) volume, shape (Z, Y, X)
        lf_vol (np.ndarray): Low-field input volume, shape (Z, Y, X)
        pred_vol (np.ndarray): Predicted high-field volume, shape (Z, Y, X)
        day_idx (int, optional): Day index for title. Default: None.
        num_slices_to_show (int): Number of slices to display. Default: 15.
        random_seed (int): Seed for reproducibility. Default: 42.
    """
    total_slices = hf_vol.shape[0]
    num_slices_to_show = min(num_slices_to_show, total_slices)
    np.random.seed(random_seed)
    slice_indices = [int(i) for i in np.linspace(0, total_slices - 1, num_slices_to_show)]
    print(f"Visualizing slices: {slice_indices}")

    fig, axes = plt.subplots(3, num_slices_to_show, figsize=(2 * num_slices_to_show, 6))
    title = f"True HF (top), LF (middle), Predicted HF (bottom) Slices"
    if day_idx is not None:
        title = f"Day {day_idx} - {title}"
    fig.suptitle(title, fontsize=16)

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