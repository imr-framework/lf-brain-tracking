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
from scipy.ndimage import sobel

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

viewing = False
ds_to_process = 4
target_resolution_fact = [1, 1, 2]
scale_factor = target_resolution_fact[2]  # Z-axis scaling factor
snr_component = False
 
max_iters = 3000
min_iters = 256
# Define parameter options
widths = [32] # width of the filters in the conv layers
depths = [4] # depth of the network
crop_sizes = [32] # size of the patches to crop from the image
noise_stds = [0.0] # standard deviation of the noise to add to the image

# Load dataset
training_path = "Data/ULC_img enhancement/Training data"
dataset = PairedMRI(training_path)
kernel_path = '/Users/sairamgeethanath/Documents/Contributions/Tools/Projects/R21/lf-brain-tracking/src/ZSSR_master/kernel_example/BSD100_100_lr_rand_ker_c_X2_0.mat'

kernel_files = ['%s_%d.mat' % (kernel_path[:-4], ind) for ind in range(len([1, 2]))]
# List subjects
print(dataset.subjects)

for i, subject_id in enumerate(dataset.subjects[0:51]):  # take first 5 subjects
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

    # if viewing:
    #     print("Displaying LF ZSSR image data using OrthoSlicer3D")
    #     OrthoSlicer3D(img_data).show()

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
    subject_HF_data = (subject_HF_data * 4095).astype(np.uint16)

    subject_LF_Monash_data = subject_LF_Monash.get_fdata()
    subject_LF_Monash_data = subject_LF_Monash_data / np.max(subject_LF_Monash_data) 
    subject_LF_Monash_data = (subject_LF_Monash_data * 4095).astype(np.uint16)

    # start a clock so that we can compute the time taken for all parameter combinations
    import time
    start_time = time.time()
    # Create all combinations
    param_combinations = list(itertools.product(widths, depths, crop_sizes, noise_stds))
    # print(param_combinations)
    for idx, (width, depth, crop_size, noise_std) in enumerate(param_combinations):
        # Print current combination
        print(Fore.YELLOW + f"Running ZSSR with width={width}, depth={depth}, crop_size={crop_size}, noise_std={noise_std}, idx = {idx}" + Style.RESET_ALL)

        # change of recon.config
        recon_config = configs.Config(width=width, depth=depth, crop_size=crop_size, noise_std=noise_std)
        # recon_config.scale_factors = [[np.sqrt(target_resolution_fact[0]), 1]]
        recon_config.scale_factors = [[(target_resolution_fact[0]), 1]]
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
        
        im_lf_sim_zssr = do_ZSSR_steps(
            img=img_data, recon_conf=recon_config, num_cols=num_cols, num_rows=num_rows,
            fname_zssr=nifti_file, fspec='', scale_fact=scale_factor, dims = 1, ground_truth=None, kernel=None)

        # Compute PSNR/SSIM/AES between im_lf_sim_zssr and subject_HF
        # (Assuming subject_HF is already loaded as a NIfTI image)

        # Ensure all images are in the same dynamic range 0 - 1
        im_lf_sim_zssr_yz = im_lf_sim_zssr / np.max(im_lf_sim_zssr)

        print(Fore.GREEN + "Shape of im_lf_sim_zssr_yz:" + str(im_lf_sim_zssr_yz.shape) + Style.RESET_ALL)

        # Now let us switch the two axes to also perform ZSSR in the other plane
        img_data_xz = np.swapaxes(img_data, 0, 1)
        print(Fore.GREEN + "Shape of img_data_xz:" + str(img_data_xz.shape) + Style.RESET_ALL)

        # Run ZSSR on the swapped axes
        im_lf_sim_zssr_xz = do_ZSSR_steps(
            img=img_data_xz, recon_conf=recon_config, num_cols=num_cols, num_rows=num_rows,
            fname_zssr=nifti_file, fspec='', scale_fact=scale_factor, dims=1, ground_truth=None, kernel=None)

        # Swap axes back to original orientation
        im_lf_sim_zssr_xz = np.swapaxes(im_lf_sim_zssr_xz, 0, 1)

        # Combine the two ZSSR results
        # Combine the two ZSSR results by selecting, for each voxel, the value from the volume (yz or xz)
        # that has the higher local gradient magnitude (i.e., sharper neighborhood)

        # Compute gradient magnitude for both volumes
        grad_yz = np.sqrt(
            sobel(im_lf_sim_zssr_yz, axis=0, mode='reflect')**2 +
            sobel(im_lf_sim_zssr_yz, axis=1, mode='reflect')**2 +
            sobel(im_lf_sim_zssr_yz, axis=2, mode='reflect')**2
        )
        grad_xz = np.sqrt(
            sobel(im_lf_sim_zssr_xz, axis=0, mode='reflect')**2 +
            sobel(im_lf_sim_zssr_xz, axis=1, mode='reflect')**2 +
            sobel(im_lf_sim_zssr_xz, axis=2, mode='reflect')**2
        )

        # For each voxel, pick the value from the sharper (higher gradient) volume
        mask = grad_yz >= grad_xz
        im_lf_sim_zssr_combined = np.where(mask, im_lf_sim_zssr_yz, im_lf_sim_zssr_xz)

        # Make sure all comparisons are between 0 to 4095 to match 12 bit DICOM range
        im_lf_sim_zssr_combined = (im_lf_sim_zssr_combined * 4095).astype(np.uint16)
        im_lf_sim_zssr = im_lf_sim_zssr_combined        
        
        # Compute PSNR
        psnr_value_monash = psnr(subject_LF_Monash_data, subject_HF_data)
        psnr_value_zssr = psnr(im_lf_sim_zssr, subject_HF_data)
        
        # Compute SSIM
        ssim_value_monash = ssim(subject_LF_Monash_data, subject_HF_data, data_range=4095)
        ssim_value_zssr = ssim(im_lf_sim_zssr, subject_HF_data, data_range=4095)

        # Compute AES
        aes_value_HF = compute_aes(subject_HF_data)
        aes_value_monash = compute_aes(subject_LF_Monash_data)
        aes_value_zssr = compute_aes(im_lf_sim_zssr)
        
        # Print PSNR, SSIM, and AES values in a table format
        print(Fore.GREEN + f"{'Method':<15}{'PSNR':<15}{'SSIM':<15}{'AES':<15}" + Style.RESET_ALL)
        print(Fore.GREEN + f"{'Monash LF':<15}{psnr_value_monash:<15.4f}{ssim_value_monash:<15.4f}{aes_value_monash:<15.4f}" + Style.RESET_ALL)
        print(Fore.GREEN + f"{'ZSSR':<15}{psnr_value_zssr:<15.4f}{ssim_value_zssr:<15.4f}{aes_value_zssr:<15.4f}" + Style.RESET_ALL)
        print(Fore.GREEN + f"{'HF (Ground Truth)':<15}{'N/A':<15}{'N/A':<15}{aes_value_HF:<15.4f}" + Style.RESET_ALL)
        
        #save values in a csvfile for all subjects in same file
        results_fname = f"./Data/Results112/zssr_results_summary.csv"
        if not os.path.exists(results_fname):
            with open(results_fname, 'w') as f:
                f.write("Subject ID,Width,Depth,Crop Size,Noise Std,PSNR Monash,SSIM Monash,AES Monash,PSNR ZSSR,SSIM ZSSR,AES ZSSR,AES HF\n")
        with open(results_fname, 'a') as f:
            f.write(f"{subject_id},{width},{depth},{crop_size},{noise_std},{psnr_value_monash:.4f},{ssim_value_monash:.4f},{aes_value_monash:.4f},{psnr_value_zssr:.4f},{ssim_value_zssr:.4f},{aes_value_zssr:.4f},{aes_value_HF:.4f}\n")
        # # Save output with subject-specific name and config values
        zssr_fname = (
            f"./Data/Results112/{subject_id}_T1_zssr_w{width}_d{depth}_c{crop_size}_n{noise_std}_test1{snr_component}.nii.gz"
        )

        # make_nifti(im_lf_sim_zssr, fname=zssr_fname, mask=False,
        #            res=[pixdim[1], pixdim[2], pixdim[3]], dim_info=[0, 1, 2])

        # print(Fore.YELLOW + f"Saved ZSSR output -> {zssr_fname}" + Style.RESET_ALL)
        viewing = True
        if viewing:
            # Display a panel of the mid coronal slice for HF, Monash LF, LF input, and LF ZSSR
            mid_slice = subject_HF_data.shape[1] // 2

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(np.rot90(subject_HF_data[:, mid_slice, :], k=1), cmap='gray')
            axes[0].set_title('High Field (HF)')
            axes[0].axis('off')

            axes[1].imshow(np.rot90(img_data[:, mid_slice, :], k=1), cmap='gray')
            axes[1].set_title('LF Input')
            axes[1].axis('off')

            axes[2].imshow(np.rot90(subject_LF_Monash_data[:, mid_slice, :], k=1), cmap='gray')
            axes[2].set_title('Monash LF')
            axes[2].axis('off')

            axes[3].imshow(np.rot90(im_lf_sim_zssr[:, mid_slice, :], k=1), cmap='gray')
            axes[3].set_title('LF ZSSR')
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig(f'./Data/Results112/{subject_id}_T1_comparison_w{width}_d{depth}_c{crop_size}_n{noise_std}_test1{snr_component}.png',
                dpi=600,  # High DPI for crisp output
                bbox_inches='tight',
                pad_inches=0.1)
            # plt.show()
        
        # if viewing:
        #     OrthoSlicer3D(im_lf_sim_zssr).show()
        #     plt.show()
    end_time = time.time()
    total_time = end_time - start_time
    print(Fore.CYAN + f"Total time for all parameter combinations: {total_time:.2f} seconds" + Style.RESET_ALL)