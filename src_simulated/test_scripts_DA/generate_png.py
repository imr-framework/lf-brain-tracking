import os
import math
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# CHANGE THIS TO YOUR BASE FOLDER
# -------------------------------
BASE_DIR = "niv_raw_data/Nipah_IRF_data/Retro_data/Only_T2_NIfTI copy/AGM_08671"

def save_png_from_nifti(nii_path):
    print(f"Processing: {nii_path}")

    # Read NIfTI
    nii = nib.load(nii_path)
    vol = nii.get_fdata().astype(np.float32)

    # Same orientation as your generator
    vol = np.rot90(vol, k=1, axes=(0, 1))

    depth = vol.shape[2]

    # Slice range: 11 -> last slice
    start_slice = 6
    end_slice = 17

    if start_slice > end_slice:
        print("Skipping (too few slices)")
        return

    slice_ids = list(range(start_slice, end_slice + 1))

    n_cols = 4
    n_rows = math.ceil(len(slice_ids) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False
    )

    for i, idx in enumerate(slice_ids):
        r = i // n_cols
        c = i % n_cols

        # EXACT orientation used in your visualize_comparison()
        axes[r, c].imshow(vol[:, :, idx], cmap="gray", origin="lower")
        axes[r, c].set_title(f"Slice {idx}", fontsize=10)
        axes[r, c].axis("off")

    # Hide unused axes
    for j in range(len(slice_ids), n_rows * n_cols):
        r = j // n_cols
        c = j % n_cols
        axes[r, c].axis("off")

    plt.tight_layout()

    # Save in same folder
    base = os.path.basename(nii_path)
    if base.endswith(".nii.gz"):
        png_name = base[:-7] + ".png"
    else:
        png_name = os.path.splitext(base)[0] + ".png"

    png_path = os.path.join(os.path.dirname(nii_path), png_name)

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Saved: {png_path}")


# -------------------------------------------------------
# Walk through all subfolders
# -------------------------------------------------------
for root, dirs, files in os.walk(BASE_DIR):
    dirs.sort()
    files.sort()

    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            save_png_from_nifti(os.path.join(root, file))

print("\nDone.")