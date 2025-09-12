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

# Define a dictionary mapping subject IDs to their corresponding lists
subject_data = {
    # Predefined subfolder sequences for specific subjects
    26184 : ["3DTSE/6", "3DTSE/9", "3DTSE/10", "3DTSE/8", "3DTSE/10"],
    30366 : ["3DTSE/4", "3DTSE/6", "3DTSE/8", "3DTSE/2", "3DTSE/8"],
    59175 : ["3DTSE/2", "3DTSE/4", "3DTSE/1", "3DTSE/9", "3DTSE/6"],
    34507 : ["3DTSE/4", "3DTSE/4", "3DTSE/3", "3DTSE/3", "3DTSE/13"],
    35547 : ["3DTSE/7", "3DTSE/4", "3DTSE/5", "3DTSE/14", "3DTSE/5"],
    59228 : ["3DTSE/8", "3DTSE/13", "3DTSE/12", "3DTSE/9", "3DTSE/13"],
    59877 : ["3DTSE/2", "3DTSE/2", "3DTSE/2", "3DTSE/2", "3DTSE/8"], # Bad quality data
    59081 : ["3DTSE/2", "3DTSE/4", "3DTSE/7", "3DTSE/6", "3DTSE/1"], # 1,2,3, 5 --
    35528 : ["3DTSE/4", "3DTSE/2", "3DTSE/3", "3DTSE/7", "3DTSE/4"],
    59233 : ["3DTSE/10", "3DTSE/6", "3DTSE/2", "3DTSE/10", "3DTSE/13"]
}

# Access using variable

def process_subject(subject='26184', fix_wrap = True, wrap_around = int(2), fov_3T = [140, 140, 70]):
    """
    Process a specific subject's data and return the loaded dataset.
    Args:
        subject (str): Subject ID to process.
        fix_wrap (bool): Whether to fix wrap-around issues.
        wrap_around (int): Number of wrap-around slices to fix.
    Returns:
        lf_dataset (list): List of 3D volumes for the subject.
    """

    sub_list = subject_data[int(subject)]
    
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
    lf_dataset = []

    for irf_folder in irf_folders:
        folder_path = os.path.join('Data/LFMRI_DATA_IRF', irf_folder)
        subfolders = [sf for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))]
        matching_subfolders = [sf for sf in subfolders if search_string in sf]
        if matching_subfolders:
            for sf in matching_subfolders:
                # print(f"  {sf}")
                visit += 1
                data_folder = os.path.join(folder_path, sf)
                three_dtse_folders = [f for f in os.listdir(data_folder) if '3DTSE' in f and os.path.isdir(os.path.join(data_folder, f))]
                for t_folder in three_dtse_folders:
                    t_folder_path = os.path.join(data_folder, t_folder)
                    subfolders_3dtse = [subf for subf in os.listdir(t_folder_path) if os.path.isdir(os.path.join(t_folder_path, subf))]
                    subfolders_3dtse.sort()
                    for subf in subfolders_3dtse:
                        # print(f"        {subf}")
                        sub_folder = os.path.join(t_folder, subf)
                        Visit_id = f'V{visit:02d}'
                        
                        name = '&'.join([irf_folder, sf, Visit_id, t_folder, subf]) + '.nii'
                        fig_name = '&'.join([irf_folder, sf, Visit_id, t_folder, subf]) + '.png'

                        im, im_props = read_lf_data(data_folder, output_folder, subject, sub_folder, name)
                        
                        if im_props is not None:
                            fov_im = [im_props.res_dim1 * im.shape[0],  im_props.res_dim2 * im.shape[1], im_props.res_dim3 * im.shape[2]]
                            if wrap_around > 0 and fix_wrap:
                                # Fix wrap-around slices if needed
                                num_wrap_slices = int(im.shape[2] / wrap_around)
                                im = np.roll(im, -int(num_wrap_slices), axis=2)
                                print(f"Fixed wrap-around for {subject}, {Visit_id}, {sub_folder}")

                            # Check if the field of view is greater than 3T, if so, skip the number of slices to ensure FOV is constant across both fields
                            if  fov_im[2] > fov_3T[2]:
                                num_slices_to_skip = int((fov_im[2] - fov_3T[2]) / im_props.res_dim3)
                                im = im[:, :, num_slices_to_skip:] # skips the lower two slices closer to the throat of the monkey - matching 3T
                                print(f"Skipped {num_slices_to_skip} slices for {subject}, {Visit_id}, {sub_folder}")
                            
                        print(f"Loaded data for {subject}, {Visit_id}, {sub_folder}")
                        print(sub_list[visit-1])

                        if sub_folder == sub_list[visit-1]:
                            # fov_im = [im_props.res_dim1 * im.shape[0],  im_props.res_dim2 * im.shape[1], im_props.res_dim3 * im.shape[2]]
                            # if wrap_around > 0 and fix_wrap:
                            #     # Fix wrap-around slices if needed
                            #     num_wrap_slices = int(im.shape[2] / wrap_around)
                            #     im = np.roll(im, -int(num_wrap_slices), axis=2)
                            #     print(f"Fixed wrap-around for {subject}, {Visit_id}, {sub_folder}")

                            # # Check if the field of view is greater than 3T, if so, skip the number of slices to ensure FOV is constant across both fields
                            # if fov_im[0] > fov_3T[0] or fov_im[1] > fov_3T[1] or fov_im[2] > fov_3T[2]:
                            #     num_slices_to_skip = int((fov_im[2] - fov_3T[2]) / im_props.res_dim3)
                            #     im = im[:, :, :num_slices_to_skip-1:]
                            #     print(f"Skipped {num_slices_to_skip} slices for {subject}, {Visit_id}, {sub_folder}")
                            config_path = 'lf-mri.json'
                            entry = {
                                "subject": subject,
                                "visit": Visit_id,
                                "data_folder": data_folder,
                                "sub_folder": sub_folder
                            }
                            
                            lf_dataset.append(im)
                        
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

                        # print(np.max(np.abs(im)))
                        # print("Min value:", np.min(np.abs(im)))
                        # print("Data type of np.abs(im):", np.abs(im).dtype)
                        print("Shape of im:", im.shape)
                        
                        # num_slices = im.shape[2]
                        # fig, axes = plt.subplots(2, 8, figsize=(20, 8))
                        # fig.suptitle(f'All Axial Slices for {name}\n{subject}\n{Visit_id}\n3DTSE/{subf}', fontsize=16)
                        # axes = axes.flatten()

                        # for i in range(16):
                        #     if i < num_slices:
                        #         slice_img = np.flipud(np.abs(im[:, :, i]).T)
                        #         axes[i].imshow(slice_img, cmap='gray')
                        #         axes[i].set_title(f'Slice {i + 1}')
                        #         axes[i].axis('off')
                        #     else:
                        #         axes[i].axis('off')

                        # plt.tight_layout()
                        # plt.savefig(f'Figures/{subject}/{fig_name}')
                        # plt.show()
                        # plt.close()

    return lf_dataset

if __name__ == "__main__":
    
    subjects = ['26184']

    for subject in subjects:
        
        # subject = '59228'
        lf_dataset = process_subject(subject = subject)  # Returns list of 3D volumes (e.g., 5 timepoints)

        print(f"Total timepoints loaded: {len(lf_dataset)}")
        for i, volume in enumerate(lf_dataset):
            print(f"Timepoint {i+1} shape: {volume.shape}")

        # # Extract the first timepoint
        # first_tp = lf_dataset[0]  # or lf_dataset[:1] to keep it as a list

        # # Print shape
        # print("Shape:", first_tp.shape)

        # # Print number of slices along each dimension
        # h, w, d = first_tp.shape
        # print(f"Height (H): {h}, Width (W): {w}, Depth (D): {d}")

        # # Or if shape is (D, H, W), change unpacking accordingly:
        # # d, h, w = first_tp.shape

        # # Print individual slices (you can choose how many to print)
        # print("\nExample slice (middle of volume):")
        # middle_slice = first_tp[:, :, d // 2]  # adjust indexing based on shape
        # print(middle_slice)

        # # Print min and max intensity values
        # print("\nMin intensity:", first_tp.min())
        # print("Max intensity:", first_tp.max())
