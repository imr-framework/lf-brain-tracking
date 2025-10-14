import sys
sys.path.insert(0, './')

sys.path.append('./data_read_code')
from src_niv.prep_data import data_ops
from src_niv.read_lf5_data import process_subject
from src_niv.utils import srr_generator, display_pred, load_and_preprocess_hf, load_and_preprocess_lf, visualize_hf_slices,padding_LF, visualize_lf_slices,rotate_slices, visualize_resampled, resample_volume, visualize_planes,visualize_pair, normalize_volume
from src_niv.prep_lf import normalize, resize_mri_volume
from src_niv.zssr import  zero_shot_super_resolution, extract_brain, extract_lf_volumes
from src_niv.niv_srr_main_single import train
from src_niv.models.ResUNet import residual_srr_unet, residual_att_unet_3d
from src_niv.models.DenseUNet import build_dense_unet_3d
from src_niv.models.Inception import build_inception_unet_3d
from demo_read_data import read_lf_data
from src_niv.prep_lf import register_to_hf

import os
import cv2
import nibabel as nib
from nilearn import plotting
from nibabel.viewers import OrthoSlicer3D
import tensorflow as tf
import pydicom
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on what's installed
import matplotlib.pyplot as plt
import ants
import nibabel as nib
from sklearn.feature_extraction import image
import os
from skimage.transform import resize  # Required for 3D resizing

# 59228
subjects1 = ['26184'] # 59228

# Define the path to the IRF_3T folder (High Field Data)
nhp_base_path = './Data/IRF_3T'
day_idx = 3
visualize = True
visualize_pairs = True
padding = False
register2_hf = True
augmentation = True

# Training parameters
steps_per_epoch = 30
epochs = 500
batch_size = 1

lf_input_volume_combined = []
hf_input_volume_combined = []
hf_target_volume_combined = []

# Define the path to the IRF_3T folder ( High Field Data)
nhp_base_path = './Data/IRF_3T'

# src_niv/prep_data.py

import sys
sys.path.insert(0, './') 
import os
import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt
# from src_niv.utils import dicom_info

class data_ops:
    def __init__(self, folder_path):
        """
        Initializes the data_ops object with the folder path containing DICOM files.
        Reads and stores volumes and voxel sizes for each time point.
        """
        self.folder_path = folder_path
        self.data, self.voxel_sizes = self.data_read(folder_path)

    def read_volume_from_folder(self, folder_path):
        
        print('Inside me ..............')

        """
        Reads and stacks DICOM slices from the folder into a 3D volume.
        Returns the volume as a NumPy array (shape: [slices, height, width]) and voxel size.
        """
        dicom_files = glob.glob(os.path.join(folder_path, "**", "*.dcm"), recursive=True)
        if not dicom_files:
            print(f" No DICOM files found in: {folder_path}")
            return None, None

        dicom_datasets = []
        for fp in dicom_files:
            try:
                ds = pydicom.dcmread(fp, force=True)

                # dicom_info(ds) # print dicom info

                dicom_datasets.append(ds)
            except Exception as e:
                print(f"Failed to read {fp}: {e}")

        # print(dicom_datasets.shape)
        # Sort by InstanceNumber to maintain slice order
        dicom_datasets.sort(key=lambda ds: int(ds.get("InstanceNumber", 0)))
        # print(dicom_datasets.shape)
        # Stack slices into a 3D numpy array
        volume = np.stack([ds.pixel_array.astype(np.float32) for ds in dicom_datasets], axis=0)
        
        # print(volume.shape)
        # volume = np.transpose(volume, (1, 2, 0)) # Shape (H, W, slices)
        # print(f"Original HF volume shape (after transpose): {volume.shape}")

        # print(volume.shape)
        # --- Get voxel size ---
        try:
            # In-plane spacing [y, x] from PixelSpacing
            pixel_spacing = dicom_datasets[0].PixelSpacing  # [row_spacing, col_spacing]
            y_spacing, x_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])

            # Between-slice spacing (z)
            if hasattr(dicom_datasets[0], 'SpacingBetweenSlices'):
                z_spacing = float(dicom_datasets[0].SpacingBetweenSlices)
            else:
                # Estimate from ImagePositionPatient
                z_positions = [float(ds.ImagePositionPatient[2]) for ds in dicom_datasets]
                z_diffs = np.diff(sorted(z_positions))
                z_spacing = np.mean(z_diffs) if len(z_diffs) > 0 else 1.0  # fallback if only one slice

            voxel_size = [z_spacing, y_spacing, x_spacing]  # [z, y, x]
            print(f" Voxel size: {voxel_size}")
        except Exception as e:
            print(f"⚠️ Could not extract voxel size: {e}")
            voxel_size = [1.0, 1.0, 1.0]  # Default/fallback spacing

        return volume, voxel_size

    def data_read(self, folder_path):
        
        """
        Reads DICOM timepoints (assumed to be 5 folders matching 'T2_n100')
        and stores volumes and voxel sizes in dictionaries.
        """
        data = {}
        voxel_sizes = {}

        print(f"\033[92mScanning: {folder_path}\033[0m")

        # Locate folders with T2w images
        day_folders = [d for d in os.listdir(folder_path)
                       if os.path.isdir(os.path.join(folder_path, d)) and "T2_n100" in d]
        day_folders.sort()

        if len(day_folders) < 5:
            print(f" Only found {len(day_folders)} timepoints (expected 5).")

        for day_idx, day_folder in enumerate(day_folders[:5], start=1):
            full_day_path = os.path.join(folder_path, day_folder)
            print(f" Reading Day {day_idx} from: {full_day_path}")
            volume, voxel_size = self.read_volume_from_folder(full_day_path)
            if volume is not None:
                data[day_idx] = volume
                voxel_sizes[day_idx] = voxel_size
                print(f" Day {day_idx}: Loaded volume with shape {volume.shape}")
            else:
                print(f" Failed to load Day {day_idx} volume.")

        return data, voxel_sizes


def load_and_preprocess_hf(subject, day_idx, visualize=True):
    """
    Loads and preprocesses HF MRI data for a given subject and day index.
    Returns the normalized, resampled HF volume.
    """

    print("Inside Me ........")

    # Initialize data object and load data (LFMRI_data_IRF)
    print(f"===============================\n\nInside utils load_and_preprocess_hf data processing day {day_idx} .............\n")
    print(f"\n====================This Function loads all day data and returns specific day========")
    subjects = os.listdir(nhp_base_path)
    subjects = sorted(subjects)
    print(f"Available subjects: {subjects}")
    print(f"Selected subject: {subject}")

    nhp_data_path = f'{nhp_base_path}/{subject}' #truct the full path to the DICOM folder

    # Initialize data object and load data (IRF_3T)
    data_obj = data_ops(nhp_data_path)

    # Retrieve dictionary of 3D volumes (day1 to day5)
    all_volumes = data_obj.data
    voxel_sizes = data_obj.voxel_sizes
    # Select and visualize Day 1: 10 slices spaced 10 apart

    volume_26184 = all_volumes[day_idx]
    voxel_sizes_26184 = voxel_sizes[day_idx]

    print(f'========================HF-MRI: Day {day_idx} and voxel size {voxel_sizes_26184} ==================')

    if visualize == True:
        
        print(f"Type: {type(volume_26184)}")
        print(f"Shape: {volume_26184.shape}")
        print(f"Dtype: {volume_26184.dtype}")
        print(f"Min: {np.min(volume_26184)}, Max: {np.max(volume_26184)}")
        print(f"Mean: {np.mean(volume_26184):.2f}, Std: {np.std(volume_26184):.2f}")
        visualize_hf_slices(all_volumes)
        # visualize_planes(all_volumes, voxel_sizes, day_idx)

for subject in subjects1:
    # for day_idx in [1, 2]:  # Assuming 0 = Day 1, 1 = Day 2
        print(f"\n=============================== Processing subject: {subject}, Day: {day_idx} ===============================")

        # ----- Load HF data -----
        print(f"\n=============================== HF_MRI data processing started .............")
        resampled_volume_hf_norm = load_and_preprocess_hf(subject, day_idx, visualize)

# # Perform the simulation below with the steps outlined above
# hf_path = 'Data/data_sim_check/3T'  # Replace with your HF DICOM series folder
# lf_noise_path = 'stored'  # Replace with your LF .3d file or folder - with repeated noise acquisitions
# lf_reference_path = 'Data/data_sim_check/47mT'  # Replace with your LF reference image path if available - to get target SNR
# output_folder = hf_path + '_1simulated_LF'

# Halbach_sim_worker = LF_sim_worker(hf_path, lf_noise_path, lf_reference_path, target_resolution=[1, 1, 2], output_folder=output_folder)
# Halbach_sim_worker.load_data()
# Halbach_sim_worker.resize_hf_to_target_resolution()
# Halbach_sim_worker.compute_noise(visualization=False)
# Halbach_sim_worker.add_noise_to_hf_image(alpha = 0.5, max_iterations=800, snr_tolerance=0.1, visualization=False)
# Halbach_sim_worker.resize_noisy_sim_to_hf_resolution() # resample back to target resolution specified - usually 1 x 1 x 2 mm_cubed for NiV
# [hf_og, x_train, y_train] = Halbach_sim_worker.save_data_for_training()  # saves the x_train and y_train as .npy files in the output folder
# print(Fore.GREEN + f'Saved training data in {Halbach_sim_worker.output_folder}' + Style.RESET_ALL)
# print(Fore.CYAN + f'Original HF image shape: {hf_og.shape}, Simulated LF image shape: {x_train.shape}, Resized HF image shape: {y_train.shape}, Target resolution: {Halbach_sim_worker.target_resolution}' + Style.RESET_ALL)
# Halbach_sim_worker.visualize_central_slices()