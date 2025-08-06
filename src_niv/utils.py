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


def visualize_hf_slices(all_volumes,voxel_sizes, day_idx):
    print("inside utils ........")
    if day_idx in all_volumes:
        vol = all_volumes[day_idx]
        slice_indices = list(range(0, 100, 10))  # [0, 10, 20, ..., 90]

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f"Day {day_idx} - Every 10th Slice", fontsize=16)
        for ax, idx in zip(axes.flat, slice_indices):
            if idx < vol.shape[0]:
                ax.imshow(vol[idx], cmap='gray')
                ax.set_title(f"Slice {idx}")
                ax.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Day {day_idx} data not found.")


def visualize_resampled(resampled_volume):
        # Check if resampled_volume exists
    if 'resampled_volume' in locals():
        print(f"Displaying slices from the resampled volume (shape: {resampled_volume.shape})...")

        # Get the dimensions of the resampled volume
        depth, height, width = resampled_volume.shape

        # Determine the number of slices to display and the step size
        # Let's display a maximum of 10 slices, evenly spaced
        num_slices_to_show = min(10, depth)
        if num_slices_to_show > 0:
            step = max(1, depth // num_slices_to_show)
            slice_indices_to_show = list(range(0, depth, step))[:num_slices_to_show]

            # Create a figure and axes for displaying the slices
            fig, axes = plt.subplots(1, num_slices_to_show, figsize=(15, 4))
            fig.suptitle("Slices from Resampled Volume", fontsize=16)

            # Handle the case of a single slice gracefully
            if num_slices_to_show == 1:
                axes = [axes]

            # Display each selected slice
            for i, slice_idx in enumerate(slice_indices_to_show):
                ax = axes[i]
                # Assuming the volume is (z, y, x) or (depth, height, width)
                ax.imshow(resampled_volume[slice_idx, :, :], cmap='gray')
                ax.set_title(f"Slice {slice_idx}")
                ax.axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        else:
            print("Resampled volume has no slices to display.")

    else:
        print("Resampled volume variable not found. Please run the resampling cell first.")

def visualize_lf_slices(im):
    print("inside utils LF........")

    print("Max value:", np.max(np.abs(im)))
    print("Min value:", np.min(np.abs(im)))
    print("Data type of np.abs(im):", np.abs(im).dtype)
    print("Shape of im:", im.shape)

    num_slices = im.shape[2]
    fig, axes = plt.subplots(2, 8, figsize=(20, 8))
    # fig.suptitle(f'All Axial Slices for {name}\n{subject}\n{Visit_id}\n3DTSE/{subf}', fontsize=16)
    axes = axes.flatten()

    for i in range(16):
        if i < num_slices:
            slice_img = np.flipud(np.abs(im[:, :, i]).T)
            axes[i].imshow(slice_img, cmap='gray')
            axes[i].set_title(f'Slice {i + 1}')
            axes[i].axis('off')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    # plt.savefig(f'Figures/{subject}/{fig_name}')
    plt.show()
    plt.close()

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
    print(f"Inferred original spacing for (z, y, x) volume: {original_spacing} mm")
    volume = volume.astype(np.float64)

    zoom_factors = [
        original_spacing[0] / new_spacing[0],  # z factor
        original_spacing[1] / new_spacing[1],  # y factor
        original_spacing[2] / new_spacing[2],  # x factor
    ]
    print(f"Original volume shape: {volume.shape}")
    print(f"Original spacing (z, y, x): {original_spacing}")
    print(f"New spacing (z, y, x): {new_spacing}")
    print(f"Zoom factors (z, y, x): {zoom_factors}")

    # Apply zoom with linear interpolation
    # Ensure that the axes are consistent (z, y, x) for both spacing and volume
    # If your volume is (slices, height, width), this corresponds to (z, y, x)
    resampled = zoom(volume, zoom=zoom_factors, order=1)  # order=1 = linear interpolation

    print(f"Resampled volume shape: {resampled.shape}")

    return resampled