import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generate_subject_grid_report(file_path, output_folder, num_slices=20):
    """
    Generates and displays high-quality grid reports for Axial, Coronal, and Sagittal views.
    """
    # 1. Load Data
    if file_path.endswith('.npy'):
        data = np.load(file_path)
    else:
        data = nib.load(file_path).get_fdata()
    
    data = np.abs(data) 
    subject_id = os.path.basename(file_path).split('.')[0]
    
    # Mapping for anatomical planes
    views = [
        {"axis": 0, "name": "Sagittal"},
        {"axis": 1, "name": "Coronal"},
        {"axis": 2, "name": "Axial"}
    ]

    # Global contrast normalization to ensure clear anatomy
    vmin, vmax = np.percentile(data, [0.5, 99.5])

    for view in views:
        axis = view["axis"]
        name = view["name"]

        # 3. Slice Selection (Central 20)
        total_slices = data.shape[axis]
        mid = total_slices // 2
        start = max(0, mid - num_slices // 2)
        slice_indices = np.arange(start, start + num_slices)

        # 4. Create Figure with Matplotlib GridSpec
        ncols, nrows = 5, 4
        fig = plt.figure(figsize=(15, 12), facecolor='black')
        
        # GridSpec settings for the 0.01 overlap / no spacing look
        gs = gridspec.GridSpec(nrows, ncols, wspace=-0.01, hspace=-0.01)

        for i in range(nrows * ncols):
            ax = fig.add_subplot(gs[i // ncols, i % ncols])
            
            if i < len(slice_indices):
                idx = slice_indices[i]
                
                # Slicing logic based on axis
                if axis == 0:   s_slice = data[idx, :, :]
                elif axis == 1: s_slice = data[:, idx, :]
                else:           s_slice = data[:, :, idx]

                # Displaying with rot90 for standard anatomical orientation
                ax.imshow(np.rot90(s_slice), cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
                
                # Overlay Slice Index
                # ax.text(10, 25, f"S: {idx}", color='red', fontsize=10, fontweight='bold', 
                #         bbox=dict(facecolor='black', alpha=0.5, lw=0))
            else:
                ax.imshow(np.zeros((64, 64)), cmap='gray')
            
            ax.axis('off')

        # 5. Styling and Interactive Display
        plt.subplots_adjust(left=0, right=1, bottom=0, top=0.93)
        plt.suptitle(f"SUBJECT: {subject_id} | VIEW: {name.upper()}", 
                     color='cyan', fontsize=22, y=0.97, fontweight='bold')

        # Save the file
        save_path = os.path.join(output_folder, f"{subject_id}_{name}_grid.png")
        plt.savefig(save_path, dpi=200, facecolor='black')
        
        # Display in Matplotlib viewer
        plt.show() 
        
        # Note: In some scripts, plt.close() is needed here to free memory 
        # but plt.show() usually handles the block.
        print(f"Finished processing {name} for {subject_id}")

# --- Execution ---
data_folder = 'Data/MW/VLF_invivo/raw'
output_folder = 'SRR_r21/pngs'
os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(data_folder) if f.endswith(('.nii', '.nii.gz', '.npy'))]

for i in range(min(3, len(files))):
    file_path = os.path.join(data_folder, files[i])
    generate_subject_grid_report(file_path, output_folder)