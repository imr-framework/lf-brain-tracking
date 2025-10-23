import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt

# Load NIfTI image
img_path = 'Data/Results_ss/POCEMR001_T1_zssr_w64_d16_c32_n0.0_test1False.nii.gz'
img = nib.load(img_path)
data = img.get_fdata()

# Normalize image intensities
data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))

# Display histogram to help choose thresholds
plt.hist(data_norm.flatten(), bins=100)
plt.title('Intensity Histogram')
plt.xlabel('Normalized Intensity')
plt.ylabel('Voxel Count')
plt.show()

# Thresholds (example values, adjust after viewing histogram)
gray_matter_thresh = (0.3, 0.6)
white_matter_thresh = (0.6, 1.0)

# Segmentation masks
gray_matter_mask = np.logical_and(data_norm >= gray_matter_thresh[0], data_norm < gray_matter_thresh[1])
white_matter_mask = np.logical_and(data_norm >= white_matter_thresh[0], data_norm <= white_matter_thresh[1])

# Save masks as NIfTI images
gray_matter_img = nib.Nifti1Image(gray_matter_mask.astype(np.uint8), img.affine)
white_matter_img = nib.Nifti1Image(white_matter_mask.astype(np.uint8), img.affine)

nib.save(gray_matter_img, 'gray_matter_mask.nii.gz')
nib.save(white_matter_img, 'white_matter_mask.nii.gz')

print("Segmentation complete. Masks saved as 'gray_matter_mask.nii.gz' and 'white_matter_mask.nii.gz'.")