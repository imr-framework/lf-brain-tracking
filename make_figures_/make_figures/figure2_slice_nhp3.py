"""
Figure 2 Generator

Creates a publication-quality 2 × 3 figure.
Each panel is independent.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ==========================
# USER SETTINGS
# ==========================

DATASET_FOLDER = "t2_sim_300"  # change to "t2_sim_300" as needed
FIGURE_NAME = f"Figure2_slice11_nhp3_{DATASET_FOLDER}"
OUTPUT_FOLDER = "Data_prospective_study/make_figures/outputs_figure2"

DPI = 600
CMAP = "gray"

# =================
# PANEL DEFINITIONS
# =================

BASE_FOLDER = os.path.join(
    "Data_prospective_study",
    DATASET_FOLDER,
    "AFRICAN.GREEN^NA_08671_MR_2018-06-28_065129_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000",
)
FILE_PREFIX = "AFRICAN.GREEN^NA_08671_MR_2018-06-28_065129_MR.BRAIN.WOWCONTRAST_T2.2D.FLAIR_n30__00000"
# Edit this one list for panel order, title, and slice values.
PANEL_CONFIG = [
    {
        "letter": "a)",
        "title": "0.5 × 0.5 × 2.0 mm",
        "key": "hf_original",
        "suffix": "1x1x2mm",
        "slice": 10,
    },
    {
        "letter": "b)",
        "title": "1.0 × 1.0 × 3.0 mm",
        "key": "lf_simulated_upsampled",
        "suffix": "1x1x3mm",
        "slice":6,
    },
    {
        "letter": "c)",
        "title": "1.0 × 1.0 × 5.0 mm",
        "key": "lf_simulated_upsampled",
        "suffix": "1x1x5mm",
        "slice": 4,
    },
    {
        "letter": "d)",
        "title": "1.0 × 1.0 × 2.0 mm",
        "key": "lf_simulated_upsampled",
        "suffix": "1x1x2mm",
        "slice": 10,
    },
    {
        "letter": "e)",
        "title": "1.5 × 1.5 × 2.0 mm",
        "key": "lf_simulated_upsampled",
        "suffix": "1.5x1.5x2mm",
        "slice": 10,
    },
    {
        "letter": "f)",
        "title": "2.0 × 2.0 × 2.0 mm",
        "key": "lf_simulated_upsampled",
        "suffix": "2x2x2mm",
        "slice": 10,
    },
]


def panel_file(suffix):
    return os.path.join(BASE_FOLDER, f"{FILE_PREFIX}_LFsim_{suffix}.npy")


PANELS = [
    {
        "file": panel_file(cfg["suffix"]),
        "key": cfg["key"],
        "plane": "axial",
        "slice": cfg["slice"],
        "title": cfg["title"],
        "letter": cfg["letter"],
    }
    for cfg in PANEL_CONFIG
]


NROWS = 2
NCOLS = 3
REFERENCE_DEPTH = 30


# ==========================
# LOAD IMAGE SLICE
# ==========================

def load_slice(filename, image_key, plane, slice_index):

    obj = np.load(
        filename,
        allow_pickle=True
    ).item()


    data = obj[image_key]


    plane = plane.lower()


    def resolve_slice_index(depth, requested_index):

        if requested_index is None:
            return depth // 2

        if 0 <= requested_index < depth:
            return requested_index

        if depth <= 1:
            return 0

        scaled_index = round(
            requested_index * (depth - 1) / (REFERENCE_DEPTH - 1)
        )

        return max(0, min(depth - 1, scaled_index))


    if plane == "axial":

        slice_index = resolve_slice_index(data.shape[2], slice_index)

        image = data[:, :, slice_index]


    elif plane == "coronal":

        slice_index = resolve_slice_index(data.shape[1], slice_index)

        image = data[:, slice_index, :]


    elif plane == "sagittal":

        slice_index = resolve_slice_index(data.shape[0], slice_index)

        image = data[slice_index, :, :]


    else:

        raise ValueError(
            f"Unknown plane: {plane}"
        )

    # rotate the image for correct orientation; K = -1 rotates 90 degrees clockwise (0,1)

    # image = np.rot90(image, k=1, axes=(0, 1))
    # image = np.rot90(image)

    return image, slice_index



# ==========================
# CREATE FIGURE
# ==========================

fig, axes = plt.subplots(
    NROWS,
    NCOLS,
    figsize=(NCOLS * 3.2, NROWS * 3.2),
    constrained_layout=False
)


axes = np.ravel(axes)



# ==========================
# DRAW PANELS
# ==========================

for i, panel in enumerate(PANELS):

    ax = axes[i]


    if panel["file"] is None:

        ax.axis("off")
        continue


    image, used_slice = load_slice(
        panel["file"],
        panel["key"],
        panel["plane"],
        panel["slice"]
    )


    ax.imshow(
        image,
        cmap=CMAP,
        origin="upper",
        interpolation="nearest"
    )


    ax.text(
        0.02,
        0.98,
        panel["letter"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="white",
    )


    if panel["title"]:
        title_with_slice = f"{panel['title']} {used_slice}"
        ax.text(
            0.5,
            0.98,
            title_with_slice,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="white",
            bbox={
                "facecolor": "black",
                "alpha": 0.55,
                "edgecolor": "none",
                "pad": 1.5,
            },
        )


    ax.axis("off")



# ==========================
# REMOVE ALL SPACING
# ==========================

fig.subplots_adjust(
    left=0,
    right=1,
    bottom=0,
    top=1,
    wspace=0,
    hspace=-0.14
)



# ==========================
# SAVE FIGURE
# ==========================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


png_file = os.path.join(
    OUTPUT_FOLDER,
    FIGURE_NAME + ".png"
)


plt.savefig(
    png_file,
    dpi=DPI,
    pad_inches=0
)


plt.close(fig)


print("Saved:", png_file)