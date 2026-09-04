"""
Retro evaluation figure generator.

Builds a 3-row figure from subject/date/slice cases.
Rows are fixed modalities:
- Row 1: volume_hf
- Row 2: Synthetic_LF
- Row 3: Synthetic_HF
"""

import os
import argparse
import glob

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# ==========================
# USER SETTINGS
# ==========================

RETRO_ROOT = "Data_prospective_study/Retro_Evaluator_t2w_new_700-500"

FIGURE_NAME = "retro_evaluation"
OUTPUT_FOLDER = "Data_prospective_study/make_figures/outputs_figure2"

DPI = 600
CMAP = "gray"
PLANE = "axial"  # axial, coronal, sagittal
ROTATE_K = 1  # 90-degree rotation (np.rot90)
APPLY_H4_BIAS_CORRECTION_ROW3 = False  # Set False to disable for third row.
H4_SIGMA_PIXELS = 24.0
SOFTEN_THIRD_ROW = False
SOFTEN_SIGMA_PIXELS = 0.9

# Set to 0.0 for fully touching panels.
# Set to 0.06 for slight separation.
WSPACE = -0.12
HSPACE = -0.08


# =====================================================
# CASE CONFIGURATION (edit subject/date/slice here)
# =====================================================

CASES = [
    {
        "subject": "AGM_08078",
        "date_token": "2018-06-20",
        "slice": 17,
        "label": "AGM_08078 20",
    },
    {
        "subject": "AGM_08078",
        "date_token": "2018-06-22",
        "slice": 17,
        "label": "AGM_08078 22 s17",
    },
    {
        "subject": "AGM_08078",
        "date_token": "2018-06-22",
        "slice": 18,
        "label": "AGM_08078 22 s18",
    },
    {
        "subject": "AGM_08112",
        "date_token": "2018-06-20",
        "slice": 16,
        "label": "AGM_08112 20",
    },
    {
        "subject": "AGM_08671",
        "date_token": "2018-06-28",
        "slice": 10,
        "label": "AGM_08671 28-06",
    },
]


ROW_DEFS = [
    ("volume_hf", "HF"),
    ("Synthetic_LF", "Synthetic LF"),
    ("Synthetic_HF", "Synthetic HF"),
]


def choose_preferred_file(candidates):
    preferred = [
        f for f in candidates
        if "T2W.TSE_n30" in os.path.basename(f)
        and "Resliced" not in os.path.basename(f)
        and "Masked" not in os.path.basename(f)
    ]
    if preferred:
        return sorted(preferred)[0]

    non_resliced = [
        f for f in candidates
        if "Resliced" not in os.path.basename(f)
        and "Masked" not in os.path.basename(f)
    ]
    if non_resliced:
        return sorted(non_resliced)[0]

    return sorted(candidates)[0]


def find_case_file(subject, subfolder, date_token):
    subject_root = os.path.join(RETRO_ROOT, subject, subfolder)
    pattern = os.path.join(subject_root, "*.nii.gz")
    candidates = [p for p in glob.glob(pattern) if date_token in os.path.basename(p)]

    if not candidates:
        raise FileNotFoundError(
            f"No file found for subject={subject}, folder={subfolder}, date={date_token}"
        )

    return choose_preferred_file(candidates)


def build_panel_rows(cases):
    rows = []
    for subfolder, row_title in ROW_DEFS:
        row = []
        for case in cases:
            resolved_file = find_case_file(
                case["subject"],
                subfolder,
                case["date_token"],
            )
            row.append(
                {
                    "file": resolved_file,
                    "slice": case["slice"],
                    "title": f"{row_title} {case['label']}",
                }
            )
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate retro evaluation figure for any subject folder."
    )
    parser.add_argument(
        "--figure-name",
        default=FIGURE_NAME,
        help="Base output figure name",
    )
    return parser.parse_args()


def load_volume(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    ext = path.lower()
    if ext.endswith(".nii") or ext.endswith(".nii.gz"):
        return np.asarray(nib.load(path).get_fdata())

    if ext.endswith(".npy"):
        return np.asarray(np.load(path))

    return np.asarray(plt.imread(path))


def resolve_slice(shape, plane, requested):
    if len(shape) < 3:
        return 0

    if plane == "axial":
        depth = shape[2]
    elif plane == "coronal":
        depth = shape[1]
    elif plane == "sagittal":
        depth = shape[0]
    else:
        raise ValueError(f"Unknown plane: {plane}")

    if requested is None:
        return depth // 2

    return int(max(0, min(depth - 1, requested)))


def extract_slice(volume, plane, slice_index):
    if volume.ndim == 2:
        return volume, 0

    if volume.ndim == 3:
        idx = resolve_slice(volume.shape, plane, slice_index)
        if plane == "axial":
            return volume[:, :, idx], idx
        if plane == "coronal":
            return volume[:, idx, :], idx
        return volume[idx, :, :], idx

    # RGB/RGBA image fallback
    return volume, 0


def normalize_for_display(image):
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)

    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)

    data = image[finite]
    lo = np.percentile(data, 1)
    hi = np.percentile(data, 99)

    if hi <= lo:
        lo = float(np.min(data))
        hi = float(np.max(data))
        if hi <= lo:
            return np.zeros_like(image, dtype=np.float32)

    out = (image - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.0
    return out


def orient_for_display(image):
    return np.rot90(image, k=ROTATE_K)


def h4_bias_correct_2d(image, sigma_pixels=24.0):
    """Apply a lightweight low-frequency bias-field correction to one 2D slice."""
    img = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=np.float32)

    out = np.zeros_like(img, dtype=np.float32)
    valid = img[finite]

    # Shift to positive range for stable log-domain correction.
    eps = 1e-6
    shifted = img.copy()
    shifted[finite] = shifted[finite] - float(np.min(valid)) + eps
    shifted[~finite] = eps

    log_img = np.log(shifted)
    h, w = log_img.shape

    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius2 = fx * fx + fy * fy
    gaussian_lp = np.exp(-2.0 * (np.pi ** 2) * (sigma_pixels ** 2) * radius2)

    bias_log = np.fft.ifft2(np.fft.fft2(log_img) * gaussian_lp).real
    corrected = np.exp(log_img - bias_log)

    scale = float(np.median(valid))
    out[finite] = corrected[finite] * scale
    out[~finite] = 0.0
    return out


def gaussian_blur_2d(image, sigma_pixels=0.9):
    """Small Gaussian blur to reduce over-sharpened appearance."""
    img = np.asarray(image, dtype=np.float32)
    if sigma_pixels <= 0:
        return img

    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=np.float32)

    fill_value = float(np.median(img[finite]))
    work = img.copy()
    work[~finite] = fill_value

    h, w = work.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius2 = fx * fx + fy * fy
    gaussian_lp = np.exp(-2.0 * (np.pi ** 2) * (sigma_pixels ** 2) * radius2)

    blurred = np.fft.ifft2(np.fft.fft2(work) * gaussian_lp).real.astype(np.float32)
    blurred[~finite] = 0.0
    return blurred


def main():
    args = parse_args()
    panel_rows = build_panel_rows(CASES)

    nrows = len(panel_rows)
    ncols = max(len(row) for row in panel_rows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.2, nrows * 3.2),
        constrained_layout=False,
    )

    axes = np.atleast_2d(axes)

    for r, row in enumerate(panel_rows):
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(row):
                ax.axis("off")
                continue

            panel = row[c]
            volume = load_volume(panel["file"])
            image, used_slice = extract_slice(volume, PLANE.lower(), panel.get("slice"))

            if r == 2 and APPLY_H4_BIAS_CORRECTION_ROW3 and image.ndim == 2:
                image = h4_bias_correct_2d(image, sigma_pixels=H4_SIGMA_PIXELS)

            if r == 2 and SOFTEN_THIRD_ROW and image.ndim == 2:
                image = gaussian_blur_2d(image, sigma_pixels=SOFTEN_SIGMA_PIXELS)

            image = orient_for_display(image)

            if image.ndim == 2:
                image = normalize_for_display(image)
                ax.imshow(image, cmap=CMAP, origin="lower", interpolation="nearest")
            else:
                ax.imshow(image, origin="lower", interpolation="nearest")

            ax.axis("off")

    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_name = f"{args.figure_name}_multi_subject.png"
    output_png = os.path.join(OUTPUT_FOLDER, output_name)

    plt.savefig(output_png, dpi=DPI, pad_inches=0)
    plt.close(fig)
    print("Saved:", output_png)


if __name__ == "__main__":
    main()
