"""
=========================================================================
Figure 1 Generator
-------------------------------------------------------------------------
Creates a 2-row publication-quality figure.

Row 1 : T2-FLAIR images
Row 2 : T2-weighted images

Features
--------
✓ Any number of visits (columns determined automatically)
✓ Individual or automatic slice selection for every image
✓ Preserves anatomical aspect ratio
✓ No whitespace between panels
✓ High-resolution output
✓ Saves to outputs_figures/
=========================================================================
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

##########################################################################
# USER SETTINGS
##########################################################################

FIGURE_NAME = "Figure2"

OUTPUT_FOLDER = "Data_prospective_study/make_figures/outputs_figures"

DATA_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Only_T2_NIfTI_copy",
    "AGM_08078",
    "2018-06__Studies"
)

PLANE = "axial"      # axial, coronal, sagittal
CMAP = "gray"

DPI = 600

##########################################################################
# EDIT ONLY THESE LISTS
##########################################################################

FLAIR = [

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN^NA_08078_MR_2018-06-08_075057_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000",
            "AFRICAN.GREEN^NA_08078_MR_2018-06-08_075057_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000.nii.gz"
        ),
        "slice": 17,
        "title": "2018-06-08"
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN^NA_08078_MR_2018-06-11_074512_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000",
            "AFRICAN.GREEN^NA_08078_MR_2018-06-11_074512_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000.nii.gz"
        ),
        "slice": 17,
        "title": "2018-06-11"
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN^NA_08078_MR_2018-06-14_074753_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000",
            "AFRICAN.GREEN^NA_08078_MR_2018-06-14_074753_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000.nii.gz"
        ),
        "slice": 16,
        "title": "2018-06-14"
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-17_070805_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30__00000",
            "AFRICAN.GREEN_NA_08078_MR_2018-06-17_070805_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30__00000.nii.gz"
        ),
        "slice": 18,
        "title": "2018-06-17"
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30_",
            "AFRICAN.GREEN_NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30_.nii.gz"
        ),
        "slice": 16,
        "title": "2018-06-20"
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-22_090645_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30__",
            "AFRICAN.GREEN_NA_08078_MR_2018-06-22_090645_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30__.nii.gz"
        ),
        "slice": 17,
        "title": "2018-06-22"
    },

]

T2W = [

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN^NA_08078_MR_2018-06-08_075057_MR.BRAIN.WOWCONTRAST_T2W.TSE_n30__00000",
            "AFRICAN.GREEN^NA_08078_MR_2018-06-08_075057_MR.BRAIN.WOWCONTRAST_T2W.TSE_n30__00000.nii.gz"
        ),
        "slice": 17
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN^NA_08078_MR_2018-06-11_074512_MR.BRAIN.WOWCONTRAST_T2W.TSE_n30__00000",
            "AFRICAN.GREEN^NA_08078_MR_2018-06-11_074512_MR.BRAIN.WOWCONTRAST_T2W.TSE_n30__00000.nii.gz"
        ),
        "slice": 17
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN^NA_08078_MR_2018-06-14_074753_MR.BRAIN.WOWCONTRAST_T2W.TSE_n30__00000",
            "AFRICAN.GREEN^NA_08078_MR_2018-06-14_074753_MR.BRAIN.WOWCONTRAST_T2W.TSE_n30__00000.nii.gz"
        ),
        "slice": 16
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-17_070805_MR.BRAIN.WCONTRAST_T2W.TSE_n30__00000",
            "AFRICAN.GREEN_NA_08078_MR_2018-06-17_070805_MR.BRAIN.WCONTRAST_T2W.TSE_n30__00000.nii.gz"
        ),
        "slice": 18
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000",
            "AFRICAN.GREEN_NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000.nii.gz"
        ),
        "slice": 16
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-22_090645_MR.BRAIN.WCONTRAST_T2W.TSE_n30__0000",
            "AFRICAN.GREEN_NA_08078_MR_2018-06-22_090645_MR.BRAIN.WCONTRAST_T2W.TSE_n30__0000.nii.gz"
        ),
        "slice": 17
    },

]

# for figure 2
FLAIR = [

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN^NA_08078_MR_2018-06-08_075057_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000",
            "AFRICAN.GREEN^NA_08078_MR_2018-06-08_075057_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000.nii.gz"
        ),
        "slice": 18,
        "title": "2018-06-08"
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN^NA_08078_MR_2018-06-11_074512_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000",
            "AFRICAN.GREEN^NA_08078_MR_2018-06-11_074512_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000.nii.gz"
        ),
        "slice": 19,
        "title": "2018-06-11"
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN^NA_08078_MR_2018-06-14_074753_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000",
            "AFRICAN.GREEN^NA_08078_MR_2018-06-14_074753_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000.nii.gz"
        ),
        "slice": 18,
        "title": "2018-06-14"
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-17_070805_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30__00000",
            "AFRICAN.GREEN_NA_08078_MR_2018-06-17_070805_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30__00000.nii.gz"
        ),
        "slice": 19,
        "title": "2018-06-17"
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30_",
            "AFRICAN.GREEN_NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30_.nii.gz"
        ),
        "slice": 18,
        "title": "2018-06-20"
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-22_090645_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30__",
            "AFRICAN.GREEN_NA_08078_MR_2018-06-22_090645_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30__.nii.gz"
        ),
        "slice": 18,
        "title": "2018-06-22"
    },

]

T2W = [

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN^NA_08078_MR_2018-06-08_075057_MR.BRAIN.WOWCONTRAST_T2W.TSE_n30__00000",
            "AFRICAN.GREEN^NA_08078_MR_2018-06-08_075057_MR.BRAIN.WOWCONTRAST_T2W.TSE_n30__00000.nii.gz"
        ),
        "slice": 18
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN^NA_08078_MR_2018-06-11_074512_MR.BRAIN.WOWCONTRAST_T2W.TSE_n30__00000",
            "AFRICAN.GREEN^NA_08078_MR_2018-06-11_074512_MR.BRAIN.WOWCONTRAST_T2W.TSE_n30__00000.nii.gz"
        ),
        "slice": 19
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN^NA_08078_MR_2018-06-14_074753_MR.BRAIN.WOWCONTRAST_T2W.TSE_n30__00000",
            "AFRICAN.GREEN^NA_08078_MR_2018-06-14_074753_MR.BRAIN.WOWCONTRAST_T2W.TSE_n30__00000.nii.gz"
        ),
        "slice": 18
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-17_070805_MR.BRAIN.WCONTRAST_T2W.TSE_n30__00000",
            "AFRICAN.GREEN_NA_08078_MR_2018-06-17_070805_MR.BRAIN.WCONTRAST_T2W.TSE_n30__00000.nii.gz"
        ),
        "slice": 19
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000",
            "AFRICAN.GREEN_NA_08078_MR_2018-06-20_072600_MR.BRAIN.WCONTRAST_T2W.TSE_n30__000.nii.gz"
        ),
        "slice": 18
    },

    {
        "file": os.path.join(
            DATA_FOLDER,
            "AFRICAN.GREEN NA_08078_MR_2018-06-22_090645_MR.BRAIN.WCONTRAST_T2W.TSE_n30__0000",
            "AFRICAN.GREEN_NA_08078_MR_2018-06-22_090645_MR.BRAIN.WCONTRAST_T2W.TSE_n30__0000.nii.gz"
        ),
        "slice": 18
    },

]
##########################################################################
# DO NOT MODIFY BELOW
##########################################################################

assert len(FLAIR) == len(T2W), "FLAIR and T2W must have the same number of visits."

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

N_COLS = len(FLAIR)
N_ROWS = 2


def select_best_slice(data, plane):

    plane = plane.lower()
    if plane == "axial":
        stats = np.sum(np.abs(data), axis=(0, 1))
    elif plane == "coronal":
        stats = np.sum(np.abs(data), axis=(0, 2))
    elif plane == "sagittal":
        stats = np.sum(np.abs(data), axis=(1, 2))
    else:
        raise ValueError("Plane must be axial/coronal/sagittal")

    return int(np.nanargmax(stats))


def load_slice(filename, plane, slice_number=None):

    img = nib.load(filename)

    data = img.get_fdata()

    if slice_number is None:
        slice_number = select_best_slice(data, plane)

    zoom = img.header.get_zooms()

    plane = plane.lower()

    if plane == "axial":
        image = data[:, :, slice_number]
        aspect = zoom[1] / zoom[0]

    elif plane == "coronal":
        image = data[:, slice_number, :]
        aspect = zoom[2] / zoom[0]

    elif plane == "sagittal":
        image = data[slice_number, :, :]
        aspect = zoom[2] / zoom[1]

    else:
        raise ValueError("Plane must be axial/coronal/sagittal")

    image = np.rot90(image)

    return image, aspect


##########################################################################
# Automatically determine figure size
##########################################################################

largest_x = 0
largest_y = 0

for panel in FLAIR:

    img = nib.load(panel["file"])

    data = img.get_fdata()

    if PLANE == "axial":
        sx, sy = data.shape[0], data.shape[1]

    elif PLANE == "coronal":
        sx, sy = data.shape[0], data.shape[2]

    else:
        sx, sy = data.shape[1], data.shape[2]

    largest_x = max(largest_x, sx)
    largest_y = max(largest_y, sy)

panel_width = 3.0
panel_height = panel_width * largest_y / largest_x

fig = plt.figure(
    figsize=(panel_width * N_COLS,
             panel_height * N_ROWS)
)

##########################################################################
# Plot Row 1 (FLAIR)
##########################################################################

for i, panel in enumerate(FLAIR):

    ax = plt.subplot(N_ROWS, N_COLS, i + 1)

    image, aspect = load_slice(
        panel["file"],
        PLANE,
        panel.get("slice")
    )

    ax.imshow(
        image,
        cmap=CMAP,
        origin="lower",
        aspect=aspect,
        interpolation="nearest"
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    if "title" in panel:
        title_text = panel["title"]
        if panel.get("slice") is not None:
            title_text += f" (slice {panel['slice']})"
        ax.set_title(title_text, fontsize=12, pad=6, fontweight="bold")


##########################################################################
# Plot Row 2 (T2W)
##########################################################################

for i, panel in enumerate(T2W):

    ax = plt.subplot(N_ROWS, N_COLS, N_COLS + i + 1)

    image, aspect = load_slice(
        panel["file"],
        PLANE,
        panel.get("slice")
    )

    ax.imshow(
        image,
        cmap=CMAP,
        origin="lower",
        aspect=aspect,
        interpolation="nearest"
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

##########################################################################
# Remove ALL spacing
##########################################################################

plt.subplots_adjust(
    left=0,
    right=1,
    bottom=0,
    top=1,
    wspace=0,
    hspace=-0.08
)

##########################################################################
# Save
##########################################################################

png_name = os.path.join(
    OUTPUT_FOLDER,
    FIGURE_NAME + ".png"
)

pdf_name = os.path.join(
    OUTPUT_FOLDER,
    FIGURE_NAME + ".pdf"
)

plt.savefig(
    png_name,
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0
)

plt.savefig(
    pdf_name,
    bbox_inches="tight",
    pad_inches=0
)

plt.close()

print()
print("=" * 60)
print("Figure saved successfully.")
print(png_name)
print(pdf_name)
print("=" * 60)