"""
=========================================================================
Figure 3

Top row    : Original T2-weighted images
Bottom row : Simulated 1×1×2 mm images

Each column represents one subject.

No titles
No labels
No spacing
Publication-quality output

Save:
outputs_figures/Figure3.png
outputs_figures/Figure3.pdf
=========================================================================
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

##########################################################################
# USER SETTINGS
##########################################################################

FIGURE_NAME = "Figure3"

OUTPUT_FOLDER = "outputs_figures"

PLANE = "axial"

CMAP = "gray"

DPI = 600

##########################################################################
# EDIT ONLY THIS SECTION
##########################################################################

ORIGINAL = [

    {
        "file": r"subject1_original.nii.gz",
        "slice": 82,
    },

    {
        "file": r"subject2_original.nii.gz",
        "slice": 74,
    },

    {
        "file": r"subject3_original.nii.gz",
        "slice": 90,
    },

    {
        "file": r"subject4_original.nii.gz",
        "slice": 81,
    },

]

SIMULATED = [

    {
        "file": r"subject1_simulated.nii.gz",
        "slice": 80,
    },

    {
        "file": r"subject2_simulated.nii.gz",
        "slice": 72,
    },

    {
        "file": r"subject3_simulated.nii.gz",
        "slice": 88,
    },

    {
        "file": r"subject4_simulated.nii.gz",
        "slice": 79,
    },

]

##########################################################################
# DO NOT MODIFY BELOW
##########################################################################

assert len(ORIGINAL) == len(SIMULATED)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

NCOLS = len(ORIGINAL)
NROWS = 2


def load_slice(fname, plane, sl):

    img = nib.load(fname)

    data = img.get_fdata()

    zoom = img.header.get_zooms()

    plane = plane.lower()

    if plane == "axial":
        image = data[:, :, sl]
        aspect = zoom[1] / zoom[0]

    elif plane == "coronal":
        image = data[:, sl, :]
        aspect = zoom[2] / zoom[0]

    elif plane == "sagittal":
        image = data[sl, :, :]
        aspect = zoom[2] / zoom[1]

    else:
        raise ValueError("Unknown plane")

    return np.rot90(image), aspect


##########################################################################
# AUTOMATIC FIGURE SIZE
##########################################################################

largest_x = 0
largest_y = 0

for panel in ORIGINAL:

    img = nib.load(panel["file"])

    shape = img.shape

    if PLANE == "axial":
        sx, sy = shape[0], shape[1]

    elif PLANE == "coronal":
        sx, sy = shape[0], shape[2]

    else:
        sx, sy = shape[1], shape[2]

    largest_x = max(largest_x, sx)
    largest_y = max(largest_y, sy)

panel_width = 3.0
panel_height = panel_width * largest_y / largest_x

fig = plt.figure(
    figsize=(
        panel_width * NCOLS,
        panel_height * NROWS
    )
)

##########################################################################
# FIRST ROW
##########################################################################

for i, panel in enumerate(ORIGINAL):

    ax = plt.subplot(NROWS, NCOLS, i + 1)

    image, aspect = load_slice(
        panel["file"],
        PLANE,
        panel["slice"]
    )

    ax.imshow(
        image,
        cmap=CMAP,
        origin="lower",
        aspect=aspect,
        interpolation="nearest"
    )

    ax.axis("off")

##########################################################################
# SECOND ROW
##########################################################################

for i, panel in enumerate(SIMULATED):

    ax = plt.subplot(
        NROWS,
        NCOLS,
        NCOLS + i + 1
    )

    image, aspect = load_slice(
        panel["file"],
        PLANE,
        panel["slice"]
    )

    ax.imshow(
        image,
        cmap=CMAP,
        origin="lower",
        aspect=aspect,
        interpolation="nearest"
    )

    ax.axis("off")

##########################################################################
# REMOVE SPACING
##########################################################################

plt.subplots_adjust(
    left=0,
    right=1,
    bottom=0,
    top=1,
    wspace=0,
    hspace=0
)

##########################################################################
# SAVE
##########################################################################

png = os.path.join(
    OUTPUT_FOLDER,
    FIGURE_NAME + ".png"
)

pdf = os.path.join(
    OUTPUT_FOLDER,
    FIGURE_NAME + ".pdf"
)

plt.savefig(
    png,
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0
)

plt.savefig(
    pdf,
    bbox_inches="tight",
    pad_inches=0
)

plt.close()

print("======================================")
print("Saved:", png)
print("Saved:", pdf)
print("======================================")