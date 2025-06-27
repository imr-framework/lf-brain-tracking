import os
import pydicom
import numpy as np

class data_ops:
    def __init__(self, folder_path):
        """
        Initializes the data_ops object with the folder path containing DICOM files.

        Args:
            folder_path (str): Path to the folder containing DICOM files.
        """
        self.folder_path = folder_path
        self.volume = self.data_read(folder_path)
        print(f"Loaded volume shape: {self.volume.shape if self.volume is not None else 'None'}")

    def data_read(self, folder_path):
        """
        Reads DICOM files from the given folder, filters for T2-weighted (T2w) images,
        and concatenates the slices into a 3D volume.

        Args:
            folder_path (str): Path to the folder containing DICOM files.

        Returns:
            np.ndarray: 3D numpy array representing the T2w volume, or None if not found.
        """

        dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if os.path.isfile(os.path.join(folder_path, f))]
        t2w_slices = []

        for file in dicom_files:
            try:
                ds = pydicom.dcmread(file, stop_before_pixels=False)
                # Check for T2w in SeriesDescription or ImageType
                desc = getattr(ds, 'SeriesDescription', '').lower()
                img_type = [s.lower() for s in getattr(ds, 'ImageType', [])]
                print(f"Processing file: {file}, SeriesDescription: {desc}, ImageType: {img_type}")
                if 't2' in desc or any('t2' in s for s in img_type):
                    t2w_slices.append((ds, ds.InstanceNumber))
            except Exception:
                continue

        if not t2w_slices:
            return None

        # Sort by InstanceNumber to maintain slice order
        t2w_slices.sort(key=lambda x: x[1])
        volume = np.stack([s[0].pixel_array for s in t2w_slices], axis=0)
        return volume
    
    def get_folder_path(self, pattern=None):
        """
        Returns the folder path containing DICOM files.

        Returns:
            str: The folder path.
        """