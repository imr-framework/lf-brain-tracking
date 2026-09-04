"""
Figure 2 Generator

Creates a publication-quality 3 × 2 figure.
Each panel is completely independent.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# ==========================
# USER SETTINGS
# ==========================

FIGURE_NAME = "Figure2"
OUTPUT_FOLDER = "outputs_figures"

DPI = 600
CMAP = "gray"


# ==========================
# PANEL DEFINITIONS
# ==========================
# Panel order:
#
# [0] [1] [2]
# [3] [4] [5]

# ==========================
# PANEL DEFINITIONS
# ==========================

BASE_FOLDER = r"Data_prospective_study/t2_sim/AFRICAN.GREEN_NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000"


PANELS = [
    {
        "file": os.path.join(
            BASE_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000_LFsim_1x1x2mm.npy"
        ),
        "plane": "axial",
        "slice": 84,
        "title": "1 × 1 × 2 mm"
    },

    {
        "file": os.path.join(
            BASE_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000_LFsim_1.5x1.5x2mm.npy"
        ),
        "plane": "axial",
        "slice": 84,
        "title": "1.5 × 1.5 × 2 mm"
    },

    {
        "file": os.path.join(
            BASE_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000_LFsim_2x2x2mm.npy"
        ),
        "plane": "axial",
        "slice": 84,
        "title": "2 × 2 × 2 mm"
    },

    {
        "file": os.path.join(
            BASE_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000_LFsim_1x1x3mm.npy"
        ),
        "plane": "axial",
        "slice": 84,
        "title": "1 × 1 × 3 mm"
    },

    {
        "file": os.path.join(
            BASE_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000_LFsim_1x1x5mm.npy"
        ),
        "plane": "axial",
        "slice": 84,
        "title": "1 × 1 × 5 mm"
    },

    {
        "file": os.path.join(
            BASE_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000_LFsim_2x2x5mm.npy"
        ),
        "plane": "axial",
        "slice": 84,
        "title": "2 × 2 × 5 mm"
    }
]


NROWS = 2
NCOLS = 3


# ==========================
# LOAD IMAGE SLICE
# ==========================

def load_slice(filename, plane, slice_index):

    nii = nib.load(filename)

    data = nii.get_fdata()
    voxel_size = nii.header.get_zooms()

    plane = plane.lower()

    if plane == "axial":
        image = data[:, :, slice_index]
        aspect_ratio = voxel_size[1] / voxel_size[0]

    elif plane == "coronal":
        image = data[:, slice_index, :]
        aspect_ratio = voxel_size[2] / voxel_size[0]

    elif plane == "sagittal":
        image = data[slice_index, :, :]
        aspect_ratio = voxel_size[2] / voxel_size[1]

    else:
        raise ValueError(f"Unknown plane: {plane}")

    return np.rot90(image), aspect_ratio



# ==========================
# AUTOMATIC FIGURE SIZE
# ==========================

max_width = 0
max_height = 0

for panel in PANELS:

    nii = nib.load(panel["file"])
    shape = nii.shape

    if panel["plane"] == "axial":
        width, height = shape[0], shape[1]

    elif panel["plane"] == "coronal":
        width, height = shape[0], shape[2]

    else:
        width, height = shape[1], shape[2]

    max_width = max(max_width, width)
    max_height = max(max_height, height)


panel_width = 3.2
panel_height = panel_width * max_height / max_width


fig, axes = plt.subplots(
    NROWS,
    NCOLS,
    figsize=(
        NCOLS * panel_width,
        NROWS * panel_height
    )
)

axes = np.ravel(axes)


# ==========================
# DRAW PANELS
# ==========================

for index, panel in enumerate(PANELS):

    ax = axes[index]

    image, aspect_ratio = load_slice(
        panel["file"],
        panel["plane"],
        panel["slice"]
    )

    ax.imshow(
        image,
        cmap=CMAP,
        aspect=aspect_ratio,
        origin="lower",
        interpolation="nearest"
    )

    ax.set_title(
        panel["title"],
        fontsize=11,
        pad=3
    )

    ax.axis("off")


# Remove ALL gaps between panels
plt.subplots_adjust(
    left=0,
    right=1,
    bottom=0,
    top=1,
    wspace=0,
    hspace=0
)


# ==========================
# SAVE OUTPUT
# ==========================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

png_file = os.path.join(
    OUTPUT_FOLDER,
    FIGURE_NAME + ".png"
)

pdf_file = os.path.join(
    OUTPUT_FOLDER,
    FIGURE_NAME + ".pdf"
)


plt.savefig(
    png_file,
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0
)

plt.savefig(
    pdf_file,
    bbox_inches="tight",
    pad_inches=0
)

plt.close(fig)


print("Saved:", png_file)
print("Saved:", pdf_file)