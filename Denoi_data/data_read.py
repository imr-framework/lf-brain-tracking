import os
from nilearn.plotting import plot_anat
from nilearn.plotting import plot_img
import matplotlib.pyplot as plt

# Path to the main directory
base_dir = 'Data/Deoni_test_retest_'

# Walk through all subfolders
for root, dirs, files in os.walk(base_dir):
    nii_files = [f for f in files if f.lower().endswith('.nii.gz')]
    for nii_file in nii_files:
        nii_path = os.path.join(root, nii_file)
        print(f"Folder: {os.path.basename(root)} - File: {nii_file}")
        plot_anat(nii_path, title=f"{os.path.basename(root)} - {nii_file}")
        plt.show()
        # Display using orthoslicer
        plot_img(nii_path, title=f"Ortho: {os.path.basename(root)} - {nii_file}")
        plt.show()