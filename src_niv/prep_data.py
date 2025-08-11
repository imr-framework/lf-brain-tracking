# src_niv/prep_data.py

import sys
sys.path.insert(0, './') 
import os
import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from src_niv.utils import dicom_info

class data_ops:
    def __init__(self, folder_path):
        """
        Initializes the data_ops object with the folder path containing DICOM files.
        Reads and stores volumes and voxel sizes for each time point.
        """
        self.folder_path = folder_path
        self.data, self.voxel_sizes = self.data_read(folder_path)

    def read_volume_from_folder(self, folder_path):
        
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
                dicom_info(ds)
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

