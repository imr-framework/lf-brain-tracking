# This file demonstrates the benefit of using ZSSR based SRR compared to bicubic interpolation
# on a single subject from the PairedMRI dataset.
# It creates a collage of all slices in the volume, applies ZSSR to the collage,
# and then splits the collage back into individual slices for comparison.
# The results are saved as NIfTI files for easy visualization.

# Step 1: Load a subject from the PairedMRI dataset
# Step 2: Subsample the low-field image to create a lower resolution version
# Step 3: Create a collage of all slices in the low-resolution volume
# Step 4: Apply ZSSR to the collage to enhance its resolution
# Step 5: Split the enhanced collage back into individual slices
# Step 6: Save the low-resolution, bicubic upsampled, ZSSR enhanced, and original high-resolution images as NIfTI files
# Step 7: Visualize the results using OrthoSlicer3D
# Step 8: Compare the Monash low-field image and ZSSR results with the original high-field image - PSNR, SSIM, AES etc.

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
from ZSSR_master import configs, configs_2, ZSSR
from src.utils import compute_aes

from nilearn import plotting
from nibabel.viewers import OrthoSlicer3D
import tensorflow as tf
import pydicom
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on what's installed
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pydicom.filereader import dcmread
from tensorflow.keras import backend as K
import scipy.io as sio



# Clear the current TensorFlow/Keras session
K.clear_session()

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

from do_zssr_collage import *
from nifti_write import make_nifti
from LF_simulation_functions import read_nifti
from colorama import Fore, Back, Style
import itertools

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

viewing = False
ds_to_process = 4
target_resolution_fact = [1, 1, 2]
snr_component = False

max_iters = 3000
min_iters = 256
# Define parameter options
widths = [64]
depths = [8]
crop_sizes = [64]
noise_stds = [0.0]

# Load dataset
training_path = "niv_raw_data/3 Monash_ULC_img enhancement/Training data"
dataset = PairedMRI(training_path)
kernel_path = '/Users/sairamgeethanath/Documents/Contributions/Tools/Projects/R21/lf-brain-tracking/src/ZSSR_master/kernel_example/BSD100_100_lr_rand_ker_c_X2_0.mat'


kernel_files = ['%s_%d.mat' % (kernel_path[:-4], ind) for ind in range(len([1, 2]))]
# List subjects
print(dataset.subjects)

for i, subject_id in enumerate(dataset.subjects[0:1]):  # take first 5 subjects
    print(Fore.CYAN + f"\n=== Processing Subject {i+1}: {subject_id} ===" + Style.RESET_ALL)

    # Get LF & HF images
    subject_LF_Monash = dataset.get_subject_image(subject_id, "T1", 'LF', visible=False, target_spacing=
                                           [1, 1, 1])
    subject_LF_ZSSR = dataset.get_subject_image(subject_id, "T1", 'LF', visible=False, 
                                           target_spacing=target_resolution_fact)
    subject_HF = dataset.get_subject_image(subject_id, "T1", 'HF', visible=False,target_spacing=
                                           [1, 1, 1])

    # Convert to uint8 NIfTI
    subject_LF_Monash = nib.Nifti1Image(subject_LF_Monash.get_fdata().astype(np.uint8),
                                 subject_LF_Monash.affine, subject_LF_Monash.header)
    subject_LF_ZSSR = nib.Nifti1Image(subject_LF_ZSSR.get_fdata().astype(np.uint8),
                                 subject_LF_ZSSR.affine, subject_LF_ZSSR.header)
    subject_HF = nib.Nifti1Image(subject_HF.get_fdata().astype(np.uint8),
                                 subject_HF.affine, subject_HF.header)

    img_data = subject_LF_ZSSR.get_fdata()

    if viewing:
        print("Displaying LF ZSSR image data using OrthoSlicer3D")
        OrthoSlicer3D(img_data).show()

    print("Shape of img_data:", img_data.shape)
    # Generate unique NIfTI filename per subject

    nifti_file = f"Data/{subject_id}_T1.nii.gz"

    hdr = subject_LF_ZSSR.header
    pixdim = hdr['pixdim']

    # Print info
    print(Fore.GREEN + 'PROCESSING NIFTI FILE METADATA' + Style.RESET_ALL)
    print("Dimensions:", hdr.get_data_shape())
    print("Voxel Sizes:", hdr.get_zooms())
    print("Data Type:", hdr.get_data_dtype())
    print("Intent:", hdr.get_intent())

    # Get the HF data for comparison
    subject_HF_data = subject_HF.get_fdata()
    subject_HF_data = subject_HF_data / np.max(subject_HF_data)
    subject_HF_data = (subject_HF_data * 255).astype(np.uint8)

    subject_LF_Monash_data = subject_LF_Monash.get_fdata()
    subject_LF_Monash_data = subject_LF_Monash_data / np.max(subject_LF_Monash_data) 
    subject_LF_Monash_data = (subject_LF_Monash_data * 255).astype(np.uint8)


    # Create all combinations
    param_combinations = list(itertools.product(widths, depths, crop_sizes, noise_stds))
    # print(param_combinations)
    for idx, (width, depth, crop_size, noise_std) in enumerate(param_combinations):
        # Print current combination
        print(Fore.YELLOW + f"Running ZSSR with width={width}, depth={depth}, crop_size={crop_size}, noise_std={noise_std}, idx = {idx}" + Style.RESET_ALL)

        # change of recon.config
        recon_config = configs.Config()
        # recon_config.scale_factors = [[np.sqrt(target_resolution_fact[0]), 1]]
        recon_config.scale_factors = [[1, 1, (target_resolution_fact[2])]]
        recon_config.max_iters = max_iters
        recon_config.min_iters = min_iters
        recon_config.width = width
        recon_config.depth = depth
        recon_config.noise_std = noise_std
        recon_config.crop_size = crop_size
        num_rows = 16
        num_cols = 14
        print('Interpolation method:', recon_config.upscale_method)
        # Run ZSSR
        print('Passing through ZSSR ..........')
        net = ZSSR.ZSSR(input_img = img_data, conf=recon_config, 
                            ground_truth=None, kernels=None)

        # Compute PSNR/SSIM/AES between im_lf_sim_zssr and subject_HF
        # (Assuming subject_HF is already loaded as a NIfTI image)
        im_lf_sim_zssr = net.run()
        # Ensure all images are in the same dynamic range 0 - 1
        im_lf_sim_zssr = im_lf_sim_zssr / np.max(im_lf_sim_zssr)
        
           

        # Convert all images to unit8 255 for all computations
        im_lf_sim_zssr = (im_lf_sim_zssr * 255).astype(np.uint8)
        
        

        # Compute PSNR
        psnr_value_monash = psnr(subject_LF_Monash_data, subject_HF_data)
        psnr_value_zssr = psnr(im_lf_sim_zssr, subject_HF_data)
        
        # Compute SSIM
        ssim_value_monash = ssim(subject_LF_Monash_data, subject_HF_data, data_range=255)
        ssim_value_zssr = ssim(im_lf_sim_zssr, subject_HF_data, data_range=255)

        # Compute AES
        aes_value_HF = compute_aes(subject_HF_data)
        aes_value_monash = compute_aes(subject_LF_Monash_data)
        aes_value_zssr = compute_aes(im_lf_sim_zssr)

        # Print PSNR, SSIM, and AES values in a table format
        print(Fore.GREEN + f"{'Method':<15}{'PSNR':<15}{'SSIM':<15}{'AES':<15}" + Style.RESET_ALL)
        print(Fore.GREEN + f"{'Monash LF':<15}{psnr_value_monash:<15.4f}{ssim_value_monash:<15.4f}{aes_value_monash:<15.4f}" + Style.RESET_ALL)
        print(Fore.GREEN + f"{'ZSSR':<15}{psnr_value_zssr:<15.4f}{ssim_value_zssr:<15.4f}{aes_value_zssr:<15.4f}" + Style.RESET_ALL)

        # # Save output with subject-specific name and config values
        zssr_fname = (
            f"./Data/Results_ss/{subject_id}_T1_zssr_w{width}_d{depth}_c{crop_size}_n{noise_std}_test1{snr_component}.nii.gz"
        )

        make_nifti(im_lf_sim_zssr, fname=zssr_fname, mask=False,
                   res=[pixdim[1], pixdim[2], pixdim[3]], dim_info=[0, 1, 2])

        print(Fore.YELLOW + f"Saved ZSSR output -> {zssr_fname}" + Style.RESET_ALL)
        if viewing:
            OrthoSlicer3D(im_lf_sim_zssr).show()
            plt.show()