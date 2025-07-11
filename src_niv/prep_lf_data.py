import sys
sys.path.append('./data_read_code')
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import os
import glob
import pydicom
import re

from demo_read_data import read_lf_data
import json

def process_subject(subject='26184'):
    output_folder = 'Data/LFMRI_DATA_IRF_nifti'
    data_dir = 'Data/IRF_3T'
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    folders.sort()

    Figures_folder = 'Figures'
    Figures = os.path.join(Figures_folder, subject)
    if not os.path.exists(Figures):
        os.makedirs(Figures)

    search_string = subject

    print(f"Searching for subfolders containing '{search_string}' in each IRF folder...")

    irf_folders = [
        f for f in os.listdir('Data/LFMRI_DATA_IRF')
        if 'IRF_071E_2_C1_' in f and os.path.isdir(
            os.path.join('Data/LFMRI_DATA_IRF', f)
        )
    ]

    irf_folders.sort()
    visit = 0

    for irf_folder in irf_folders:
        folder_path = os.path.join('Data/LFMRI_DATA_IRF', irf_folder)
        subfolders = [sf for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))]
        matching_subfolders = [sf for sf in subfolders if search_string in sf]
        if matching_subfolders:
            for sf in matching_subfolders:
                print(f"  {sf}")
                visit += 1
                data_folder = os.path.join(folder_path, sf)
                three_dtse_folders = [f for f in os.listdir(data_folder) if '3DTSE' in f and os.path.isdir(os.path.join(data_folder, f))]
                for t_folder in three_dtse_folders:
                    t_folder_path = os.path.join(data_folder, t_folder)
                    subfolders_3dtse = [subf for subf in os.listdir(t_folder_path) if os.path.isdir(os.path.join(t_folder_path, subf))]
                    subfolders_3dtse.sort()
                    for subf in subfolders_3dtse:
                        print(f"        {subf}")
                        sub_folder = os.path.join(t_folder, subf)
                        Visit_id = f'V{visit:02d}'
                        name = '&'.join([irf_folder, sf, Visit_id, t_folder, subf]) + '.nii'
                        fig_name = '&'.join([irf_folder, sf, Visit_id, t_folder, subf]) + '.png'
                        print(f"  Visit ID: {Visit_id}")

                        im = read_lf_data(data_folder, output_folder, subject, sub_folder, name)

                        config_path = 'lf-mri.json'
                        entry = {
                            "subject": subject,
                            "visit": Visit_id,
                            "data_folder": data_folder,
                            "sub_folder": sub_folder
                        }

                        # Load existing config or create new
                        if os.path.exists(config_path):
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                        else:
                            config = []

                        config.append(entry)

                        with open(config_path, 'w') as f:
                            json.dump(config, f, indent=2)

                        if im is None:
                            print("Skipping due to read error.")
                            continue

                        print(np.max(np.abs(im)))
                        print("Min value:", np.min(np.abs(im)))
                        print("Data type of np.abs(im):", np.abs(im).dtype)
                        print("Shape of im:", im.shape)

                        num_slices = im.shape[2]
                        fig, axes = plt.subplots(2, 8, figsize=(20, 8))
                        fig.suptitle(f'All Axial Slices for {name}\n{subject}\n{Visit_id}\n3DTSE/{subf}', fontsize=16)
                        axes = axes.flatten()

                        for i in range(16):
                            if i < num_slices:
                                slice_img = np.flipud(np.abs(im[:, :, i]).T)
                                axes[i].imshow(slice_img, cmap='gray')
                                axes[i].set_title(f'Slice {i + 1}')
                                axes[i].axis('off')
                            else:
                                axes[i].axis('off')

                        plt.tight_layout()
                        plt.savefig(f'Figures/{subject}/{fig_name}')
                        plt.show()
                        plt.close()

if __name__ == "__main__":
    process_subject()
