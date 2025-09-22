import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Path to the folder containing the files
data_folder = 'Data/MW/VLF_invivo/raw'

# List all files in the folder and pick the first NIfTI or NPY file
files = [f for f in os.listdir(data_folder) if f.endswith(('.nii', '.nii.gz', '.npy'))]
if not files:
    raise FileNotFoundError("No NIfTI or NPY files found in the specified folder.")
# Prepare a figure to show all three subjects vertically, each with three slices horizontally
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

output_folder = 'output_nifty'
os.makedirs(output_folder, exist_ok=True)

for i in range(3):  
    file_path = os.path.join(data_folder, files[i])

    # Load the image
    if file_path.endswith('.npy'):
        data = np.load(file_path)
        affine = np.eye(4)
    else:
        img = nib.load(file_path)
        data = img.get_fdata()
        affine = img.affine

    # Get the center slices for each axis
    x_center = data.shape[0] // 2
    y_center = data.shape[1] // 2
    z_center = data.shape[2] // 2

    # Prepare the slices
    slice_x = data[x_center, :, :]
    slice_y = data[:, y_center, :]
    slice_z = data[:, :, z_center]

    # Plot the slices in the i-th row (each subject is a row)
    axes[i, 0].imshow(np.abs(slice_x), cmap='gray', origin='lower')
    axes[i, 0].set_title(f'Subject {i+1} - Sagittal (X)')
    axes[i, 1].imshow(np.abs(slice_y).T, cmap='gray', origin='lower')
    axes[i, 1].set_title(f'Subject {i+1} - Coronal (Y)')
    axes[i, 2].imshow(np.abs(slice_z).T, cmap='gray', origin='lower')
    axes[i, 2].set_title(f'Subject {i+1} - Axial (Z)')

    for j in range(3):
        axes[i, j].axis('off')

    # Save sagittal, coronal, and axial slices as NIfTI files with correct orientation
    # Expand dims to ensure 3D shape for NIfTI (slice, height, width)
    sagittal_data = np.abs(slice_x)[None, :, :]  # shape (1, Y, Z)
    coronal_data = np.abs(slice_y)[:, None, :]   # shape (X, 1, Z)
    axial_data = np.abs(slice_z)[:, :, None]     # shape (X, Y, 1)

    sagittal_img = nib.Nifti1Image(sagittal_data, affine)
    sagittal_path = os.path.join(output_folder, f'subject_{i+1}_sagittal.nii.gz')
    nib.save(sagittal_img, sagittal_path)

    coronal_img = nib.Nifti1Image(coronal_data, affine)
    coronal_path = os.path.join(output_folder, f'subject_{i+1}_coronal.nii.gz')
    nib.save(coronal_img, coronal_path)

    axial_img = nib.Nifti1Image(axial_data, affine)
    axial_path = os.path.join(output_folder, f'subject_{i+1}_axial.nii.gz')
    nib.save(axial_img, axial_path)

plt.tight_layout()
plt.show()
