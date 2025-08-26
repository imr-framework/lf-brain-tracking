import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Paths to the directories
day_idx = 2
base_dir = "./Data/Results/build_dual_encoder_unet/30366/predictions"
items = os.listdir(base_dir)
print(f"Items in base directory: {items}")
hf = os.path.join(base_dir, f"HF_input_volume_day{day_idx}.nii.gz")
lf = os.path.join(base_dir, f"LF_input_volume_day{day_idx}.nii.gz")
pred = os.path.join(base_dir, f"Predicted_volume_day{day_idx}.nii.gz")

# Load NIfTI volumes from the specified file paths
hf_img = nib.load(hf).get_fdata()
lf_img = nib.load(lf).get_fdata()
pred_img = nib.load(pred).get_fdata()

# Take the middle slice along the z-axis
hf_slice = hf_img[hf_img.shape[0] // 2,:, : ]
lf_slice = lf_img[hf_img.shape[0] // 2,:, :]
pred_slice = pred_img[hf_img.shape[0] // 2,:, :]

# Plot the slices side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(hf_slice.T, cmap='gray', origin='lower')
axes[0].set_title('HF Input Day 1')
axes[0].axis('off')

axes[1].imshow(lf_slice.T, cmap='gray', origin='lower')
axes[1].set_title('LF Input Day 1')
axes[1].axis('off')

axes[2].imshow(pred_slice.T, cmap='gray', origin='lower')
axes[2].set_title('Predicted Day 1')
axes[2].axis('off')

plt.tight_layout()
plt.show()
