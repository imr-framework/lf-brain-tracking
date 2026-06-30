import nibabel as nib
from nilearn.plotting import plot_img
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

import matplotlib.pyplot as plt

# Path to your NIfTI file
# nii_path = "srr_zssr_3D_full.nii.gz"

# Load the NIfTI image
# img = nib.load(nii_path)

# Load NIfTI
img = nib.load("srr_zssr_3D_full.nii.gz")

# Extract image data (this is IMPORTANT)
data = img.get_fdata()
# Print shape of the data
print("Data shape:", data.shape)
# min and max values
print("Data min:", data.min())
print("Data max:", data.max())

# Display with OrthoSlicer3D
OrthoSlicer3D(data).show()