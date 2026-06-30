
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
viewing = True
from nibabel.viewers import OrthoSlicer3D

# File paths of the NIfTI files
file_paths = [
    'Data/POCEMR104_FLAIR_slice400_zssr_noise_True.nii.gz'
]

# Load and display each NIfTI file
for file_path in file_paths:
    img = nib.load(file_path)
    data = img.get_fdata()
    plt.imshow(data[:, :, data.shape[2] // 2], cmap='gray')
    plt.show()

    if viewing == True:
        OrthoSlicer3D(data).show()