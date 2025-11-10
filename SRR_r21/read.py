import os
import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

# Input and output folders
data_folder = 'Data/MW/VLF_invivo/raw'
output_folder = 'SRR_r21/output_nifty'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all .npy, .nii, .nii.gz files
files = [f for f in os.listdir(data_folder) if f.endswith(('.nii', '.nii.gz', '.npy'))]

for i, file_name in enumerate(files):
    file_path = os.path.join(data_folder, file_name)

    # Load data
    if file_path.endswith('.npy'):
        data = np.load(file_path)
        affine = np.eye(4)
        data = np.abs(data)
        img = nib.Nifti1Image(data, affine)
    else:  # already NIfTI
        img = nib.load(file_path)
        data = img.get_fdata()
        data = np.abs(data)
        affine = img.affine

    data = np.abs(data)
    print(f"\n[{i+1}] Processing: {file_name}")
    print(f"Shape: {data.shape}")

    # Save as NIfTI in output_folder
    output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.nii.gz')
    nib.save(img, output_file)
    print(f"Saved NIfTI: {output_file}")

    # Optional: view interactively
    OrthoSlicer3D(data).show()
