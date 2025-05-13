
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt

# File paths of the NIfTI files
file_paths = [
    './Data/output_image.nii.gz',
    './Data/output_image_LF_simulated.nii.gz',
    './Data/output_image_zssr_noise_True.nii.gz'
]

# Load and display each NIfTI file
for file_path in file_paths:
    img = nib.load(file_path)
    data = img.get_fdata()
    plt.imshow(data[:, :, data.shape[2] // 2], cmap='gray')
    plt.show()