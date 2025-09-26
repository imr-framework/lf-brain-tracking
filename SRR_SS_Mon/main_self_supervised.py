# SISR-ZSSR
# Read single image of T1 or T2 or Flair and train ZSSR


import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./LFsim')
sys.path.append('./src')
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from typing import Dict, Tuple
from SRR_SS_Mon.data_read import PairedMRI


from nilearn import plotting
from nibabel.viewers import OrthoSlicer3D
import tensorflow as tf
import pydicom
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on what's installed

from pydicom.filereader import dcmread

print(tf.config.list_physical_devices('GPU'))  # Check if GPU is visible
print(tf.config.list_physical_devices('CPU'))  # Check if CPU is visible

if tf.config.list_physical_devices('GPU'):
    print("CUDNN detected!")
else:
    print("Display the data using OrthoSlicer3D")
    OrthoSlicer3D(img.dataobj).show()
    plotting.plot_anat(img, title="3D TSC Image")
    plt.show()

# Note: Undo CUDNN detected!")

from do_zssr_collage import do_mask_image, do_ZSSR_steps, do_zssr_recon_slices
from nifti_write import make_nifti
from LF_simulation_functions import read_nifti
from colorama import Fore, Back, Style

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

viewing = True
ds_to_process = 4
target_resolution_fact = [1, 1, 1]
snr_component = True

dataset = PairedMRI("Data/ULC_img enhancement/Training data")

# List subjects
print(dataset.subjects)

subject_LF = dataset.get_subject_image(dataset.subjects[0], "T1", 'LF', visible=False)
subject_HF = dataset.get_subject_image(dataset.subjects[0], "T1", 'HF', visible=False)

# Convert both images to uint8
subject_LF = nib.Nifti1Image(subject_LF.get_fdata().astype(np.uint8), subject_LF.affine, subject_LF.header)
subject_HF = nib.Nifti1Image(subject_HF.get_fdata().astype(np.uint8), subject_HF.affine, subject_HF.header)

img_data = subject_LF.get_fdata()  # Convert to numpy array
nifti_file = 'Data/POCEMR104_T1_111.nii.gz'

# nifti_data_lf = nib.load(simulated_nifti)
hdr = subject_LF.header
pixdim  = hdr['pixdim']

# Print information from NIfTI data header
print(Fore.GREEN + 'PROCESSING NIFTI FILE METADATA' + Style.RESET_ALL)
print("NIfTI Header Information:")
print("Dimensions:", hdr.get_data_shape())
print("Voxel Sizes:", hdr.get_zooms())
print("Data Type:", hdr.get_data_dtype())
print("Intent:", hdr.get_intent())
print("Affine Matrix:")
print(hdr.get_qform())

# if viewing == True:
#     print(Fore.GREEN + 'Viewing: ', subject_LF,  Style.RESET_ALL)
#     OrthoSlicer3D(subject_LF).show()
#     # Display using interactive browser-based viewer
#     # view = plotting.view_img(subject_LF, title="LF Simulated Image")
#     # view.open_in_browser()

# 3. Pass it through the ZSSR algorithm
print('Passing through ZSSR')
im_lf_sim_zssr = do_zssr_recon_slices(img=img_data, fname=nifti_file, 
                    do_preprocess = True, do_postprocess=True, 
                    padding=False, mask_image=False, reuse_mask=False, 
                    target_resolution_fact= target_resolution_fact)

# 4. Save the output as a nifti file
zssr_fname= nifti_file[:-7] + '_zssr_noise_'+ str(snr_component) +'.nii.gz'
make_nifti(im_lf_sim_zssr, fname =  zssr_fname, mask=False, 
res=[pixdim[1], pixdim[2], pixdim[3]], dim_info=[0, 1, 2])

if viewing == True:
    OrthoSlicer3D(im_lf_sim_zssr).show()


