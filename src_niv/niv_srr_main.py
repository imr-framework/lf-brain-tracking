import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./data_read_code')
from src_niv.prep_data import data_ops
from demo_read_data import read_lf_data

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

# Define the path to the IRF_3T folder ( High Field Data)
nhp_base_path = './Data/IRF_3T'
subject = '26184'  # Example subject number, adjust as needed

subjects = os.listdir(nhp_base_path)
subjects = sorted(subjects)
print(f"Available subjects: {subjects}")
print(f"Selected subject: {subject}")

nhp_data_path = f'{nhp_base_path}/{subject}' #truct the full path to the DICOM folder

# Initialize data object and load data (IRF_3T)
data_obj = data_ops(nhp_data_path)

# Retrieve dictionary of 3D volumes (day1 to day5)
all_volumes = data_obj.data

# Select and visualize Day 1: 10 slices spaced 10 apart
day_idx = 1
volume_26184 = all_volumes[day_idx]

print(f"Type: {type(volume_26184)}")
print(f"Shape: {volume_26184.shape}")
print(f"Dtype: {volume_26184.dtype}")
print(f"Min: {np.min(volume_26184)}, Max: {np.max(volume_26184)}")
print(f"Mean: {np.mean(volume_26184):.2f}, Std: {np.std(volume_26184):.2f}")

# if day_idx in all_volumes:
#     vol = all_volumes[day_idx]
#     slice_indices = list(range(0, 100, 10))  # [0, 10, 20, ..., 90]

#     fig, axes = plt.subplots(2, 5, figsize=(15, 6))
#     fig.suptitle(f"Day {day_idx} - Every 10th Slice", fontsize=16)
#     for ax, idx in zip(axes.flat, slice_indices):
#         if idx < vol.shape[0]:
#             ax.imshow(vol[idx], cmap='gray')
#             ax.set_title(f"Slice {idx}")
#             ax.axis('off')
#     plt.tight_layout()
#     plt.show()
# else:
#     print(f"Day {day_idx} data not found.")

# Initialize data object and load data (LFMRI_data_IRF)
# Iterate inside each irf_folder and print subfolders containing a specific string
data_folder='Data/LFMRI_DATA_IRF/IRF_071E_2_C1_20240709/34507_D_minus28'
output_folder='/home/ajay/Documents/lf-brain-tracking/Data/LFMRI_DATA_IRF_nifti'
subject=subject
sub_folder='3DTSE/3'
file_name='20240829_day2_3DTSC_12.nii'
name = file_name
im = read_lf_data(data_folder, output_folder, subject, sub_folder, file_name)

# if im is None:
#     print("Skipping due to read error.")
#     continue

print(np.max(np.abs(im)))
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

# Prepare the data

# Display the data

# Perform and call SRR reconstruction framework

# Save the SRR results to NIfTI files

# Display the SRR results

# Compare the SRR results with the original data

# Identify and visualize differences using XAI techniques
