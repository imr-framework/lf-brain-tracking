
import os
import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt

class data_ops:
    def __init__(self, folder_path):
        """
        Initializes the data_ops object with the folder path containing DICOM files.
        Reads and stores volumes for each time point.
        """
        self.folder_path = folder_path
        self.data = self.data_read(folder_path)

    def read_volume_from_folder(self, folder_path):
        """
        Reads and stacks DICOM slices from the folder into a 3D volume.
        Returns the volume as a NumPy array (shape: [slices, height, width]).
        """
        dicom_files = glob.glob(os.path.join(folder_path, "**", "*.dcm"), recursive=True)
        if not dicom_files:
            print(f" No DICOM files found in: {folder_path}")
            return None

        dicom_datasets = []
        for fp in dicom_files:
            try:
                ds = pydicom.dcmread(fp, force=True)
                dicom_datasets.append(ds)
            except Exception as e:
                print(f"Failed to read {fp}: {e}")

        # Sort by InstanceNumber to maintain slice order
        dicom_datasets.sort(key=lambda ds: int(ds.get("InstanceNumber", 0)))

        # Stack slices into a 3D numpy array
        volume = np.stack([ds.pixel_array.astype(np.float32) for ds in dicom_datasets], axis=0)
        return volume

    def data_read(self, folder_path):
        """
        Reads DICOM timepoints (assumed to be 5 folders matching 'T2_n100')
        and stores volumes in a dictionary.
        """
        data = {}

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
            volume = self.read_volume_from_folder(full_day_path)
            if volume is not None:
                data[day_idx] = volume
                print(f" Day {day_idx}: Loaded volume with shape {volume.shape}")
            else:
                print(f" Failed to load Day {day_idx} volume.")

        return data

