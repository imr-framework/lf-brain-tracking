# # SISR-ZSSR
# # Read single image of T1 or T2 or Flair and train ZSSR
# # 1.6 x 1.6 x 1
# # 2D ZSSR --> 1 x 1 x 1

# #Bilinear Interpolation
# #ZSSR
# #Residual Network

# import sys
# sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
# sys.path.append('./LFsim')
# sys.path.append('./src')
# import os
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import zoom
# from typing import Dict, Tuple
# from SRR_SS_Mon.data_read import PairedMRI
# from ZSSR_master import configs, configs_2, ZSSR

# from nilearn import plotting
# from nibabel.viewers import OrthoSlicer3D
# import tensorflow as tf
# import pydicom
# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on what's installed

# from pydicom.filereader import dcmread

# print(tf.config.list_physical_devices('GPU'))  # Check if GPU is visible
# print(tf.config.list_physical_devices('CPU'))  # Check if CPU is visible

# if tf.config.list_physical_devices('GPU'):
#     print("CUDNN detected!")
# else:
#     print("Display the data using OrthoSlicer3D")
#     OrthoSlicer3D(img.dataobj).show()
#     plotting.plot_anat(img, title="3D TSC Image")
#     plt.show()

# # Note: Undo CUDNN detected!")

# from do_zssr_collage import do_mask_image, do_ZSSR_steps, do_zssr_recon_slices
# from nifti_write import make_nifti
# from LF_simulation_functions import read_nifti
# from colorama import Fore, Back, Style
# import itertools

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# viewing = False
# ds_to_process = 4
# target_resolution_fact = [1.6, 1.6, 1]
# snr_component = False

# max_iters = 3000
# min_iters = 256

# # Define parameter options
# widths = [64]
# depths = [8]
# crop_sizes = [32]
# noise_stds = [0.1]
# # Load dataset
# training_path = "Data/ULC_img enhancement/Training data"

# dataset = PairedMRI(training_path)

# # List subjects
# print(dataset.subjects)

# # Get voxel sizes
# voxel_info = dataset.get_voxel_sizes(dataset.subjects[0])
# print(voxel_info)
# viewing = True

# for i, subject_id in enumerate(dataset.subjects[0:1]):  # take first 5 subjects
#     print(Fore.CYAN + f"\n=== Processing Subject {i+1}: {subject_id} ===" + Style.RESET_ALL)

#     # Get LF & HF images
#     subject_LF = dataset.get_subject_image(subject_id, "T1", 'LF',target_spacing = (1.6, 1.6, 1), visible=False)
#     subject_HF = dataset.get_subject_image(subject_id, "T1", 'HF',target_spacing = (1.6, 1.6, 1), visible=False)

#     # Convert to uint8 NIfTI
#     subject_LF = nib.Nifti1Image(subject_LF.get_fdata().astype(np.uint8),
#                                  subject_LF.affine, subject_LF.header)
#     subject_HF = nib.Nifti1Image(subject_HF.get_fdata().astype(np.uint8),
#                                  subject_HF.affine, subject_HF.header)

#     img_data = subject_LF.get_fdata()

#     if viewing:
#         print("Displaying LF image data using OrthoSlicer3D")
#         OrthoSlicer3D(img_data).show()

#     # Generate unique NIfTI filename per subject
    
#     nifti_file = f"Data/{subject_id}_T1.nii.gz"

#     hdr = subject_LF.header
#     pixdim = hdr['pixdim']

#     # Print info
#     print(Fore.GREEN + 'PROCESSING NIFTI FILE METADATA' + Style.RESET_ALL)
#     print("Dimensions:", hdr.get_data_shape())
#     print("Voxel Sizes:", hdr.get_zooms())
#     print("Data Type:", hdr.get_data_dtype())
#     print("Intent:", hdr.get_intent())

#     # Print current combination
#     # print(Fore.YELLOW + f"Running ZSSR with width={width}, depth={depth}, crop_size={crop_size}, noise_std={noise_std}" + Style.RESET_ALL)

#     tf.keras.backend.clear_session()

#     # change of recon.config
#     recon_config = configs.Config()
#     recon_config.scale_factors = [[np.sqrt(target_resolution_fact[0]), 1]]
#     recon_config.max_iters = max_iters
#     recon_config.min_iters = min_iters
#     recon_config.width = width
#     recon_config.depth = depth
#     recon_config.noise_std = noise_std
#     recon_config.crop_size = crop_size

#     # Run ZSSR
#     print('Passing through ZSSR ..........')
#     im_lf_sim_zssr = do_zssr_recon_slices(
#         img=img_data, fname=nifti_file,
#         do_preprocess=True, do_postprocess=True,
#         padding=False, mask_image=False, reuse_mask=False,
#         target_resolution_fact=target_resolution_fact, recon_conf=recon_config
#     )

#     # # Save output with subject-specific name and config values
#     zssr_fname = (
#         f"Data/Results_ss/{subject_id}_T1_zssrnew_test1{snr_component}.nii.gz"
#     )

#     make_nifti(im_lf_sim_zssr, fname=zssr_fname, mask=False,
#                 res=[pixdim[1], pixdim[2], pixdim[3]], dim_info=[0, 1, 2])
    
#     print(Fore.YELLOW + f"Saved ZSSR output -> {zssr_fname}" + Style.RESET_ALL)
#     if viewing:
#         OrthoSlicer3D(im_lf_sim_zssr).show()