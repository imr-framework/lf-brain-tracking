import os
import glob
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re

class data_ops:
    def __init__(self, folder_path):
        """
        Initializes the data_ops object with the folder path containing DICOM files.

        Args:
            folder_path (str): Path to the folder containing DICOM files.
        """
        self.folder_path = folder_path
        self.volume = self.data_read(folder_path)

    def data_read(self, folder_path):
        """
        Reads DICOM files from the given folder, filters for T2-weighted (T2w) images,
        and concatenates the slices into a 3D volume.

        Args:
            folder_path (str): Path to the folder containing DICOM files.

        Returns:
            np.ndarray: 3D numpy array representing the T2w volume, or None if not found.
        """
        data = {}
        subject_number = 0
        # 1. Locate subject folders (e.g. 'S01', 'S02', …) ----------------------
        subjects = [d for d in os.listdir(folder_path)
                    if os.path.isdir(os.path.join(folder_path, d))]
        subjects.sort()  
        print(f"Found subjects: {subjects}")

        # 2. Walk through each subject and each day --------------------------------
        for subj in subjects[0:1]:
            subject_number += 1
            print(f"\033[92mStarting processing of subject {subject_number}: {subj}\033[0m")

            subj_dir = os.path.join(folder_path, subj)
            data[subj] = {}

            # 2. Get all day folders (assume random-named, e.g., date strings)
            day_folders = [d for d in os.listdir(subj_dir)
                        if os.path.isdir(os.path.join(subj_dir, d)) and "T2_n100" in d]
            
            day_folders.sort()

            # print(f"Found day folders for subject {subj}: {day_folders}")

            # Iterate over days (1 to 6) and process each corresponding folder
            for day_idx, day_folder in enumerate(day_folders[:6], start=1):
                sequence_path = os.path.join(subj_dir, day_folder)
                dicom_files = glob.glob(os.path.join(sequence_path, "**", "*.dcm"), recursive=True)
                dicom_files.sort()

                dicom_datasets = []
                for fp in dicom_files:
                    # print(f"Reading DICOM file: {fp}")
                    ds = pydicom.dcmread(fp, force=True)
                    dicom_datasets.append(ds)

                dicom_datasets.sort(key=lambda ds: int(ds.get("InstanceNumber", 0)))
                data[subj][day_idx] = dicom_datasets

                print(f"Subject {subj}, Day {day_idx}, Found {len(dicom_datasets)} DICOM files in {sequence_path}")
                
                # # Only proceed if there are enough slices
                # if len(dicom_datasets) >= 100:
                #     slice_indices = [0, 19, 39, 59, 99]
                #     fig, axes = plt.subplots(1, 5, figsize=(15, 5))
                #     for ax, idx in zip(axes, slice_indices):
                #         img = dicom_datasets[idx].pixel_array
                #         ax.imshow(img, cmap='gray')
                #         ax.set_title(f"Slice {idx+1}")
                #         ax.axis('off')
                #     plt.tight_layout()
                #     plt.show()

                # # Only proceed if there are enough slices
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

        return data

    
