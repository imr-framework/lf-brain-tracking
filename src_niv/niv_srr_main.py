import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
from src_niv.prep_data import data_ops

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

# Define the path to the IRF_3T folder
nhp_base_path = './Data/IRF_3T'

# subjects = os.listdir(nhp_base_path)
# subjects = sorted(subjects)
# print(f"Available subjects: {subjects}")
# nhp_num = subjects[0]  # Select the first subject for processing
# print(f"Selected subject: {nhp_num}")

# nhp_data_path = f'{nhp_base_path}/{nhp_num}'  # Construct the full path to the DICOM folder
data = data_ops(nhp_base_path)


# Only proceed if there are enough slices
# if len(data[subj][day_idx]) >= 100:
#     slice_indices = [0, 19, 39, 59, 99]
#     fig, axes = plt.subplots(1, 5, figsize=(15, 5))
#     for ax, idx in zip(axes, slice_indices):
#         img = data[subj][day_idx][idx].pixel_array
#         ax.imshow(img, cmap='gray')
#         ax.set_title(f"Slice {idx+1}")
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
# Sample dummy data creation (shape 64x64)

# nhp_26184_dates_volumes = nhp_26184.volume


# Read the DICOM data

# Prepare the data

# Display the data

# Perform and call SRR reconstruction framework

# Save the SRR results to NIfTI files

# Display the SRR results

# Compare the SRR results with the original data

# Identify and visualize differences using XAI techniques
