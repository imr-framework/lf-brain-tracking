import os
import nibabel as nib
import random

# import os
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt

# # Paths to the directories
# day_idx = 2
# base_dir = "./Data/Results/build_dual_encoder_unet/30366/predictions"
# items = os.listdir(base_dir)
# print(f"Items in base directory: {items}")
# hf = os.path.join(base_dir, f"HF_input_volume_day{day_idx}.nii.gz")
# lf = os.path.join(base_dir, f"LF_input_volume_day{day_idx}.nii.gz")
# pred = os.path.join(base_dir, f"Predicted_volume_day{day_idx}.nii.gz")

# # Load NIfTI volumes from the specified file paths
# hf_img = nib.load(hf).get_fdata()
# lf_img = nib.load(lf).get_fdata()
# pred_img = nib.load(pred).get_fdata()

# # Take the middle slice along the z-axis
# hf_slice = hf_img[hf_img.shape[0] // 2,:, : ]
# lf_slice = lf_img[hf_img.shape[0] // 2,:, :]
# pred_slice = pred_img[hf_img.shape[0] // 2,:, :]

# # Plot the slices side by side
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].imshow(hf_slice.T, cmap='gray', origin='lower')
# axes[0].set_title('HF Input Day 1')
# axes[0].axis('off')

# axes[1].imshow(lf_slice.T, cmap='gray', origin='lower')
# axes[1].set_title('LF Input Day 1')
# axes[1].axis('off')

# axes[2].imshow(pred_slice.T, cmap='gray', origin='lower')
# axes[2].set_title('Predicted Day 1')
# axes[2].axis('off')

# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt

folder_path = "Data/IRF_3T_LFsim/59233"
nii_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
print("NIfTI files in folder:")
for f in nii_files:
    print(f)
    img = nib.load(os.path.join(folder_path, f))
    data = img.get_fdata()
    header = img.header
    print(f"Shape of {f}: {data.shape}")
    print(f"Voxel size of {f}: {header.get_zooms()}")

    # Display random axial slices in one row
    num_slices = 5
    z_indices = random.sample(range(data.shape[2]), num_slices)
    fig, axes = plt.subplots(1, num_slices, figsize=(3*num_slices, 3))
    for i, z in enumerate(z_indices):
        axes[i].imshow(data[:, :, z].T, cmap='gray', origin='lower')
        axes[i].set_title(f"Axial z={z}")
        axes[i].axis('off')
    plt.suptitle(f)
    plt.tight_layout()
    plt.show()
