import os
import nibabel as nib
import matplotlib.pyplot as plt

# Folder containing the NIfTI files
input_folder = 'SRR_r21/output_nifty'

# List and sort the first three .nii.gz files
nii_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.nii.gz')])[:3]

# Load each file into separate variables
data_list = []
for f in nii_files:
    img = nib.load(os.path.join(input_folder, f))
    data_list.append(img.get_fdata())

# Parameters
start_slice = 10      # first slice to display
num_slices = 10       # number of slices to display
         # step size between slices

orientations = ['Axial', 'Coronal', 'Sagittal']

# Loop through orientations
for ori_idx, ori in enumerate(orientations):
    fig, axes = plt.subplots(num_slices, len(data_list), figsize=(4*len(data_list), 2*num_slices))
    
    for slice_idx in range(num_slices):
        if ori =='Axial':
            step = 2
        else:
            step = 8
        s = start_slice + slice_idx * step  # compute slice index with step
        for file_idx, data in enumerate(data_list):
            ax = axes[slice_idx, file_idx] if num_slices > 1 else axes[file_idx]
            
            if ori == 'Axial':
                ax.imshow(data[:, :, s], cmap='gray')
            elif ori == 'Coronal':
                ax.imshow(data[:, s, :], cmap='gray')
            elif ori == 'Sagittal':
                ax.imshow(data[s, :, :], cmap='gray')
            
            if slice_idx == 0:
                ax.set_title(nii_files[file_idx], fontsize=10)
            ax.axis('off')
        
        axes[slice_idx, 0].set_ylabel(f'{ori} slice {s}', fontsize=10)
    
    plt.suptitle(f'{ori} slices comparison across files (step={step})', fontsize=14)
    plt.tight_layout()
    plt.show()