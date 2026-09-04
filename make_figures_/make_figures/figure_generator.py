"""
=========================================================================
Universal NIfTI Figure Generator

Features:
---------
- Multiple NIfTI files
- Multiple visits
- Different slice numbers
- Different resolutions
- Any grid size
- Controlled overlap
- Publication-quality output

=========================================================================
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


##########################################################################
# USER SETTINGS
##########################################################################

FIGURE_NAME = "Figure3"

OUTPUT_FOLDER = "outputs_figures"

# Grid
NROWS = 2
NCOLS = 4


# Image plane
PLANE = "axial"
# options:
# axial
# coronal
# sagittal


# Display
CMAP = "gray"

DPI = 600


# Show titles above panels
SHOW_TITLES = True


# Panel overlap control
#
# 0.00 = touching
# positive = more separation
# negative = overlap
#
# Example:
# -0.05 gives slight overlap

HORIZONTAL_OVERLAP = -0.03

VERTICAL_OVERLAP = -0.03


##########################################################################
# DEFINE PANELS
#
# Enter panels row-by-row
#
# Example:
#
# Row 1:
# panel1 panel2 panel3 panel4
#
# Row 2:
# panel5 panel6 panel7 panel8
#
##########################################################################


PANELS = [

    # ---------------- ROW 1 ----------------

    {
        "file": r"visit1_original.nii.gz",
        "slice": 80,
        "title": "Visit 1"
    },

    {
        "file": r"visit2_original.nii.gz",
        "slice": 82,
        "title": "Visit 2"
    },

    {
        "file": r"visit3_original.nii.gz",
        "slice": 84,
        "title": "Visit 3"
    },

    {
        "file": r"visit4_original.nii.gz",
        "slice": 86,
        "title": "Visit 4"
    },


    # ---------------- ROW 2 ----------------


    {
        "file": r"visit1_simulated.nii.gz",
        "slice": 78,
        "title": ""
    },

    {
        "file": r"visit2_simulated.nii.gz",
        "slice": 80,
        "title": ""
    },

    {
        "file": r"visit3_simulated.nii.gz",
        "slice": 82,
        "title": ""
    },

    {
        "file": r"visit4_simulated.nii.gz",
        "slice": 84,
        "title": ""
    },

]


##########################################################################
# CHECK
##########################################################################

if len(PANELS) != NROWS * NCOLS:
    raise ValueError(
        "Number of panels must equal NROWS × NCOLS"
    )


os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


##########################################################################
# FUNCTION TO EXTRACT SLICE
##########################################################################

def get_slice(filename, plane, sl):

    img = nib.load(filename)

    data = img.get_fdata()

    zoom = img.header.get_zooms()


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

        raise ValueError(
            "Plane must be axial, coronal, or sagittal"
        )


    image = np.rot90(image)


    return image, aspect



##########################################################################
# AUTOMATIC FIGURE SIZE
##########################################################################

max_x = 0
max_y = 0


for p in PANELS:

    img = nib.load(
        p["file"]
    )

    shape = img.shape


    if PLANE == "axial":

        x, y = shape[0], shape[1]


    elif PLANE == "coronal":

        x, y = shape[0], shape[2]


    else:

        x, y = shape[1], shape[2]


    max_x = max(max_x, x)

    max_y = max(max_y, y)



panel_width = 3.0

panel_height = (
    panel_width *
    max_y /
    max_x
)



##########################################################################
# CREATE FIGURE
##########################################################################

fig = plt.figure(
    figsize=(
        panel_width * NCOLS,
        panel_height * NROWS
    ),
    facecolor="white"
)



gs = gridspec.GridSpec(

    NROWS,
    NCOLS,

    figure=fig,

    wspace=HORIZONTAL_OVERLAP,

    hspace=VERTICAL_OVERLAP

)



##########################################################################
# PLOT PANELS
##########################################################################

for i, panel in enumerate(PANELS):


    row = i // NCOLS

    col = i % NCOLS


    ax = fig.add_subplot(
        gs[row, col]
    )


    image, aspect = get_slice(

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



    if SHOW_TITLES:

        ax.set_title(

            panel["title"],

            fontsize=11,

            pad=3

        )



##########################################################################
# SAVE
##########################################################################

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



plt.close()



print("--------------------------------")
print("Figure saved:")
print(png_file)
print(pdf_file)
print("--------------------------------")