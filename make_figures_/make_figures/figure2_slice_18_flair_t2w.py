"""
Combined Figure 2 panel for FLAIR and T2W at slice 18.
Top row: FLAIR
Bottom row: T2W
Same a-e sequence as the single-modality slice-18 panels, excluding the 1x1x5mm volume.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


DATASET_FOLDER = "t2_sim_300"
FIGURE_NAME = f"Figure2_slice18_flair_t2w_{DATASET_FOLDER}"
OUTPUT_FOLDER = "Data_prospective_study/make_figures/outputs_figure2"
DPI = 600
CMAP = "gray"
REFERENCE_DEPTH = 30

FLAIR_BASE_FOLDER = os.path.join(
    "Data_prospective_study",
    DATASET_FOLDER,
    "AFRICAN.GREEN NA_08078_MR_2018-06-22_090645_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30__",
)
FLAIR_FILE_PREFIX = "AFRICAN.GREEN NA_08078_MR_2018-06-22_090645_MR.BRAIN.WCONTRAST_T2.2D.FLAIR_n30__"

T2W_BASE_FOLDER = os.path.join(
    "Data_prospective_study",
    DATASET_FOLDER,
    "AFRICAN.GREEN NA_08078_MR_2018-06-22_090645_MR.BRAIN.WCONTRAST_T2W.TSE_n30__0000",
)
T2W_FILE_PREFIX = "AFRICAN.GREEN NA_08078_MR_2018-06-22_090645_MR.BRAIN.WCONTRAST_T2W.TSE_n30__0000"


PANEL_SEQUENCE = [
    {
        "letter": "a)",
        "title": "0.5 × 0.5 × 2.0",
        "suffix": "1x1x2mm",
        "slice_flair": 18,
        "slice_t2w": 18,
        "key": "hf_original",
    },
    {
        "letter": "b)",
        "title": "1.0 × 1.0 × 2.0",
        "suffix": "1x1x2mm",
        "slice_flair": 18,
        "slice_t2w": 19,
        "key": "lf_simulated_upsampled",
    },
    {
        "letter": "c)",
        "title": "1.0 × 1.0 × 3.0",
        "suffix": "1x1x3mm",
        "slice_flair": 12,
        "slice_t2w": 13,
        "key": "lf_simulated_upsampled",
    },
    {
        "letter": "d)",
        "title": "1.5 × 1.5 × 2.0",
        "suffix": "1.5x1.5x2mm",
        "slice_flair": 18,
        "slice_t2w": 19,
        "key": "lf_simulated_upsampled",
    },
    {
        "letter": "e)",
        "title": "2.0 × 2.0 × 2.0",
        "suffix": "2x2x2mm",
        "slice_flair": 18,
        "slice_t2w": 19,
        "key": "lf_simulated_upsampled",
    },
]


def panel_file(base_folder, file_prefix, suffix):
    return os.path.join(base_folder, f"{file_prefix}_LFsim_{suffix}.npy")


def build_panels():
    panels = []
    for cfg in PANEL_SEQUENCE:
        panels.append(
            {
                "modality": "FLAIR",
                "file": panel_file(FLAIR_BASE_FOLDER, FLAIR_FILE_PREFIX, cfg["suffix"]),
                "slice": cfg["slice_flair"],
                **cfg,
            }
        )
        panels.append(
            {
                "modality": "T2W",
                "file": panel_file(T2W_BASE_FOLDER, T2W_FILE_PREFIX, cfg["suffix"]),
                "slice": cfg["slice_t2w"],
                **cfg,
            }
        )
    return panels


PANELS = build_panels()


def resolve_slice_index(depth, requested_index):
    if requested_index is None:
        return depth // 2
    if 0 <= requested_index < depth:
        return requested_index
    if depth <= 1:
        return 0
    scaled_index = round(requested_index * (depth - 1) / (REFERENCE_DEPTH - 1))
    return max(0, min(depth - 1, scaled_index))


def load_slice(filename, image_key, plane, slice_index):
    obj = np.load(filename, allow_pickle=True).item()
    data = obj[image_key]
    plane = plane.lower()

    if plane == "axial":
        slice_idx = resolve_slice_index(data.shape[2], slice_index)
        image = data[:, :, slice_idx]
    elif plane == "coronal":
        slice_idx = resolve_slice_index(data.shape[1], slice_index)
        image = data[:, slice_idx, :]
    elif plane == "sagittal":
        slice_idx = resolve_slice_index(data.shape[0], slice_index)
        image = data[slice_idx, :, :]
    else:
        raise ValueError(f"Unknown plane: {plane}")

    return image, slice_idx


fig, axes = plt.subplots(2, 5, figsize=(15.0, 6.0), constrained_layout=False)
axes = np.asarray(axes).reshape(2, 5)

for row_idx, modality in enumerate(["FLAIR", "T2W"]):
    for col_idx, panel in enumerate(PANEL_SEQUENCE):
        ax = axes[row_idx, col_idx]
        modal_panel = next(
            p for p in PANELS if p["modality"] == modality and p["letter"] == panel["letter"]
        )

        image, _ = load_slice(
            modal_panel["file"],
            modal_panel["key"],
            "axial",
            modal_panel["slice"],
        )

        ax.imshow(image, cmap=CMAP, origin="upper", interpolation="nearest")
        ax.text(
            0.02,
            0.98,
            panel["letter"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
            color="white",
        )
        ax.text(
            0.5,
            0.98,
            panel["title"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.5},
        )
        ax.axis("off")

for ax in axes.flat:
    ax.set_aspect("equal")

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=-0.030, hspace=-0.15)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
png_path = os.path.join(OUTPUT_FOLDER, f"{FIGURE_NAME}.png")
plt.savefig(png_path, dpi=DPI, pad_inches=0)
plt.close(fig)

print("Saved:", png_path)
