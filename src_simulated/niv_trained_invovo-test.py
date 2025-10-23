# Load models
# Load lf-data
# evaluate functions
# recombine

import sys
sys.path.insert(0, './')
sys.path.insert(0, './data_read_code')

import argparse
import sys
import logging
from datetime import datetime
import time
import pydicom
from glob import glob
import LFsim.keaDataProcessing as keaProc
from LFsim.utils_sim import *
import os
from read_kea3d import kea3d

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from src_niv.metrics import psnr, ssim, mse, composite_loss
from src_niv.utils import visualize_pair

# ------------------------------------------------
# Sliding Window Inference
# ------------------------------------------------
def predict_volume(model, lf_volume, patch_size=(64,64,32), overlap=0.5):
    """
    Sliding-window 3D prediction on LF volume.
    Returns predicted enhanced volume of same shape.
    """
    H, W, D = lf_volume.shape
    px, py, pz = patch_size

    # Strides
    sx = int(px * (1 - overlap))
    sy = int(py * (1 - overlap))
    sz = int(pz * (1 - overlap))

    pad_x = (px - H % px) if H % px != 0 else 0
    pad_y = (py - W % py) if W % py != 0 else 0
    pad_z = (pz - D % pz) if D % pz != 0 else 0

    lf_padded = np.pad(lf_volume, ((0,pad_x),(0,pad_y),(0,pad_z)), mode='reflect')
    H_pad, W_pad, D_pad = lf_padded.shape

    pred_volume = np.zeros_like(lf_padded, dtype=np.float32)
    count_volume = np.zeros_like(lf_padded, dtype=np.float32)

    for x in range(0, H_pad - px + 1, sx):
        for y in range(0, W_pad - py + 1, sy):
            for z in range(0, D_pad - pz + 1, sz):
                patch = lf_padded[x:x+px, y:y+py, z:z+pz]
                patch_input = np.expand_dims(patch, axis=(0,-1))
                pred_patch = model.predict(patch_input, verbose=0)
                pred_patch = np.squeeze(pred_patch)
                pred_volume[x:x+px, y:y+py, z:z+pz] += pred_patch
                count_volume[x:x+px, y:y+py, z:z+pz] += 1.0

    pred_volume /= np.maximum(count_volume, 1e-8)
    return pred_volume[:H, :W, :D]

import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def visualize_and_save_results(model_path, lf, hf, pred, pred2=None, slices=[20, 40, 60],
                               title_prefix="", out_dir="Output_patch/results"):
    """
    Displays specified LF, HF, predicted slices, and saves both the entire volumes (NIfTI)
    and each displayed figure as PNGs named by slice number.
    """
    os.makedirs(out_dir, exist_ok=True)  # Create output folder if it doesn't exist

    # --- Display and Save specified slices ---
    for s in slices:
        plt.figure(figsize=(16, 5))
        num_subplots = 4 if pred2 is not None else 3

        # LF
        plt.subplot(1, num_subplots, 1)
        plt.imshow(lf[:, :, s], cmap='gray')
        plt.title(f"LF slice {s}")
        plt.axis('off')

        # Predicted (1-pass)
        plt.subplot(1, num_subplots, 2)
        plt.imshow(pred[:, :, s], cmap='gray')
        plt.title("Predicted (1-pass)")
        plt.axis('off')

        # Pred2 if provided
        if pred2 is not None:
            plt.subplot(1, num_subplots, 3)
            plt.imshow(pred2[:, :, s], cmap='gray')
            plt.title("Re-enhanced (2-pass)")
            plt.axis('off')

        # HF Ground Truth
        plt.subplot(1, num_subplots, num_subplots)
        plt.imshow(hf[:, :, s], cmap='gray')
        plt.title("HF Ground Truth")
        plt.axis('off')

        plt.suptitle(f"{title_prefix} Slice {s}\nModel: {model_path}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # --- Save the figure ---
        fig_path = os.path.join('Output_patch/results', f"{title_prefix}_slice{s}.png")
        # plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"✅ Saved figure for slice {s}: {fig_path}")

        plt.show()
        plt.close()

        # # --- Save entire volumes as NIfTI ---
        # affine = np.eye(4)  # Use identity if no affine is provided
        # # nib.save(nib.Nifti1Image(lf, affine), os.path.join(out_dir, f"{title_prefix}_LF.nii.gz"))
        # nib.save(nib.Nifti1Image(hf, affine), os.path.join(out_dir, f"{title_prefix}_HF.nii.gz"))
        # # nib.save(nib.Nifti1Image(pred, affine), os.path.join(out_dir, f"{title_prefix}_Pred.nii.gz"))

        # if pred2 is not None:
        #     nib.save(nib.Nifti1Image(pred2, affine), os.path.join(out_dir, f"{title_prefix}_Pred2.nii.gz"))

        # print(f"💾 Saved full 3D volumes to {out_dir}")

# ------------------------------------------------
# Iterative Enhancement Pipeline
# ------------------------------------------------
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

def evaluate_model(folder_path, model_name, X_test, y_test,
                   patch_size=(64, 64, 32), overlap=0.5,
                   visualize_slices=[17]):
    """
    Evaluate a two-stage (base + retrained) SRR or denoising model pipeline.

    Args:
        folder_path (str): Path containing the model checkpoints.
        model_name (str): Base model name (without suffixes).
        X_test, y_test: Test LF and HF volumes (numpy arrays).
        patch_size (tuple): Patch size for 3D inference.
        overlap (float): Overlap ratio for sliding window inference.
        visualize_slices (list): Slice indices for visualization.

    Example:
        evaluate_model("models/", "LFHF_SRR_Model", X_test, y_test)
        -> loads:
            models/LFHF_SRR_Model_checkpoint.keras
            models/LFHF_SRR_Model_retrained_checkpoint.keras
    """

    # ------------------------------------------------------------
    # 🔹 Define model paths
    # ------------------------------------------------------------
    model_path_train = os.path.join(folder_path, f"{model_name}_checkpoint.keras")
    model_path_retrained = os.path.join(folder_path, f"{model_name}_retrained_checkpoint.keras")

    if not os.path.exists(model_path_train):
        raise FileNotFoundError(f"Base model not found: {model_path_train}")
    if not os.path.exists(model_path_retrained):
        raise FileNotFoundError(f"Retrained model not found: {model_path_retrained}")

    print(f"🔹 Base model: {model_path_train}")
    print(f"🔹 Retrained model: {model_path_retrained}")

    # ------------------------------------------------------------
    # 🔹 Load models
    # ------------------------------------------------------------
    model1 = load_model(model_path_train,
                        custom_objects={'psnr': psnr, 'ssim': ssim,
                                        'mse': mse, 'composite_loss': composite_loss},
                        compile=False)
    model2 = load_model(model_path_retrained,
                        custom_objects={'psnr': psnr, 'ssim': ssim,
                                        'mse': mse, 'composite_loss': composite_loss},
                        compile=False)
    print("✅ Both models loaded successfully.")

    results = []

    # ------------------------------------------------------------
    # 🔹 Evaluate each subject
    # ------------------------------------------------------------
    for i in range(len(X_test)):
        print(f"\n🧠 Evaluating subject {i+1}/{len(X_test)} ...")
        lf = X_test[i]
        hf = y_test[i]

        # ---- Stage 1 Prediction ----
        pred1 = predict_volume(model1, lf, patch_size=patch_size, overlap=overlap)

        # ---- Stage 2 Refinement ----
        pred2 = predict_volume(model2, pred1, patch_size=patch_size, overlap=overlap)

        # ---- Compute Metrics ----
        psnr1 = psnr(tf.convert_to_tensor(hf[np.newaxis, ..., np.newaxis]),
                     tf.convert_to_tensor(pred1[np.newaxis, ..., np.newaxis])).numpy()
        ssim1 = ssim(tf.convert_to_tensor(hf[np.newaxis, ..., np.newaxis]),
                     tf.convert_to_tensor(pred1[np.newaxis, ..., np.newaxis])).numpy()

        psnr2 = psnr(tf.convert_to_tensor(hf[np.newaxis, ..., np.newaxis]),
                     tf.convert_to_tensor(pred2[np.newaxis, ..., np.newaxis])).numpy()
        ssim2 = ssim(tf.convert_to_tensor(hf[np.newaxis, ..., np.newaxis]),
                     tf.convert_to_tensor(pred2[np.newaxis, ..., np.newaxis])).numpy()

        print(f"📈 Stage1 → PSNR: {psnr1:.3f}, SSIM: {ssim1:.4f}")
        print(f"📈 Stage2 → PSNR: {psnr2:.3f}, SSIM: {ssim2:.4f}")

        # # ---- Visualization ----
        visualize_and_save_results(model_name, lf, hf, pred1, pred2,
                          slices=visualize_slices, title_prefix=f"Subject_day5 {i}")

        # ---- Collect results ----
        results.append({
            'subject': i,
            'psnr_stage1': psnr1,
            'ssim_stage1': ssim1,
            'psnr_stage2': psnr2,
            'ssim_stage2': ssim2
        })

    print("\n✅ Evaluation complete.")
    return results


# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize_volume(vol, method='minmax'):
    if method=='minmax':
        vol_min, vol_max = vol.min(), vol.max()
        if vol_max - vol_min > 0:
            vol = (vol - vol_min) / (vol_max - vol_min)
        else:
            vol = np.zeros_like(vol)
    elif method=='zscore':
        mean, std = vol.mean(), vol.std()
        if std>0:
            vol = (vol - mean) / std
        else:
            vol = np.zeros_like(vol)
    return vol

def normalize_dataset(X, y, method='minmax'):
    X_norm = np.array([normalize_volume(vol, method) for vol in X])
    y_norm = np.array([normalize_volume(vol, method) for vol in y])
    return X_norm, y_norm

# ------------------------------------------------
# Example Usage
# ------------------------------------------------
def load_lf_3d_file(lf_path):
    
    # Assuming .3d is a simple binary float32 file with known dimensions
    # You may need to adapt this to your .3d file format

    sub_folder = os.path.basename(lf_path)
    data_folder = os.path.dirname(lf_path)
    sample_data = kea3d(data_folder=data_folder, sub_folder=sub_folder)
    kspace = sample_data.kspace_gauss_filter
    im = np.abs(np.fft.fftshift(np.fft.fftn((np.fft.fftshift(kspace)))))

    acqu_path = lf_path + '/acqu.par'
    image_path_LF = lf_path + '/data.3d'
    ImageScanParams = keaProc.readPar(acqu_path)
        
    # self.LF_ref_kSpace = keaProc.readKSpace(image_path_LF)
    # LF_acq = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(self.LF_ref_kSpace)))
    
    LF_ref_kSpace = kspace
    LF_acq = im
    
    LF_ref_im = np.abs(LF_acq)
    fov_LF_ref_acq = ImageScanParams.get('FOV')
    matrix_LF_ref_acq = LF_acq.shape
    res_LF_ref_acq = np.divide(fov_LF_ref_acq, matrix_LF_ref_acq)
    print(Fore.CYAN + 'Matrix size of acquired Low Field image: ', matrix_LF_ref_acq, Style.RESET_ALL)
    print(Fore.CYAN + 'FOV of acquired LF: ', fov_LF_ref_acq, Style.RESET_ALL)
    print(Fore.CYAN + 'Resolution of acquired LF: ', res_LF_ref_acq, Style.RESET_ALL)

    num_slices = LF_acq.shape[2]

    # fig, axes = plt.subplots(2, 8, figsize=(20, 8))
    # # fig.suptitle(f'All Axial Slices for {name}\n{subject}\n{Visit_id}\n3DTSE/{subf}', fontsize=16)
    # axes = axes.flatten()

    # for i in range(16):
    #     if i < num_slices:
    #         slice_img = np.flipud(np.abs(LF_acq[:, :, i]).T)
    #         axes[i].imshow(slice_img, cmap='gray')
    #         axes[i].set_title(f'Slice {i + 1}')
    #         axes[i].axis('off')
    #     else:
    #         axes[i].axis('off')

    # plt.tight_layout()
    # # plt.savefig(f'Figures/{subject}/{fig_name}')
    # plt.show()
    # plt.close()

    return im

import numpy as np
from scipy.ndimage import zoom

def resample_volume_numpy(im, current_spacing=(2.0, 2.0, 5.0), new_spacing=(1.0, 1.0, 2.0), order=3):
    """
    Resample a 3D numpy volume to the desired voxel spacing.
    
    Args:
        im (np.ndarray): 3D MRI volume (Z, Y, X)
        current_spacing (tuple): Current voxel spacing in mm (z, y, x)
        new_spacing (tuple): Desired voxel spacing in mm (z, y, x)
        order (int): Interpolation order (0=nearest, 1=linear, 3=cubic)
    
    Returns:
        np.ndarray: Resampled 3D volume
    """
    zoom_factors = [current_spacing[i] / new_spacing[i] for i in range(3)]
    print(f"Zoom factors (z, y, x): {zoom_factors}")
    
    resampled_im = zoom(im, zoom_factors, order=order)
    print(f"Original shape: {im.shape} → New shape: {resampled_im.shape}")
    
    return resampled_im

import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 🔹 Helper functions for padding/unpadding
# ------------------------------------------------------------
def pad_volume_to_multiple(volume, multiple=8):
    """
    Pads a 3D or 4D (batched) volume so that each spatial dimension is 
    a multiple of `multiple`.

    Args:
        volume (np.ndarray): (H, W, D) or (B, H, W, D)
        multiple (int): The multiple to pad to (default=8)

    Returns:
        padded_volume (np.ndarray)
        pad_info (list): [(before_h, after_h), (before_w, after_w), (before_d, after_d)]
    """
    if volume.ndim == 4:
        _, h, w, d = volume.shape
    elif volume.ndim == 3:
        h, w, d = volume.shape
    else:
        raise ValueError(f"Unsupported shape: {volume.shape}")

    def pad_for_dim(size):
        remainder = size % multiple
        if remainder == 0:
            return (0, 0)
        total_pad = multiple - remainder
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        return (pad_before, pad_after)

    pad_h = pad_for_dim(h)
    pad_w = pad_for_dim(w)
    pad_d = pad_for_dim(d)
    pad_info = [pad_h, pad_w, pad_d]

    if volume.ndim == 4:
        padded = np.pad(volume, ((0, 0), pad_h, pad_w, pad_d), mode='constant')
    else:
        padded = np.pad(volume, (pad_h, pad_w, pad_d), mode='constant')

    return padded, pad_info


def unpad_volume(volume, pad_info):
    """
    Removes padding using pad_info from pad_volume_to_multiple().
    """
    pad_h, pad_w, pad_d = pad_info
    if volume.ndim == 4:
        _, h, w, d = volume.shape
        return volume[:, 
                      pad_h[0]:h - pad_h[1] if pad_h[1] > 0 else h,
                      pad_w[0]:w - pad_w[1] if pad_w[1] > 0 else w,
                      pad_d[0]:d - pad_d[1] if pad_d[1] > 0 else d]
    elif volume.ndim == 3:
        h, w, d = volume.shape
        return volume[
            pad_h[0]:h - pad_h[1] if pad_h[1] > 0 else h,
            pad_w[0]:w - pad_w[1] if pad_w[1] > 0 else w,
            pad_d[0]:d - pad_d[1] if pad_d[1] > 0 else d
        ]
    else:
        raise ValueError(f"Unsupported input shape: {volume.shape}")

# ------------------------------------------------------------
# 🔹 Main script
# ------------------------------------------------------------
if __name__ == "__main__":
    model_name = 'residual_srr_unet_l1_l2_ssim_mse_ssim_edge'
    folder_path = "Output_patch"
    data_folder = "Data/data_sim_check/3T_1simulated_LF/train_test"
    subjects = ["26184"]
    test_days = [3]

    # ------------------------------------------------------------
    # Load subject data
    # ------------------------------------------------------------
    def load_subject_day_data(subject, day):
        path = os.path.join(data_folder, f"{subject}_day{day}_train_data.npy")
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            return None
        data = np.load(path, allow_pickle=True).item()
        return data["x_train"].astype(np.float32), data["y_train"].astype(np.float32)

    X_test, y_test = [], []
    for subj in subjects:
        for day in test_days:
            d = load_subject_day_data(subj, day)
            if d is not None:
                X_test.append(d[0])
                y_test.append(d[1])

    X_test, y_test = np.array(X_test), np.array(y_test)
    print(f"📦 Loaded {len(X_test)} test volumes.")

    # Normalize and pad test data
    X_test, _ = pad_volume_to_multiple(X_test, multiple=8)
    y_test, pad_info_y = pad_volume_to_multiple(y_test, multiple=8)
    X_test, y_test = normalize_dataset(X_test, y_test, method='minmax')

    print(f"X_test shape: {X_test.shape}, range=({X_test.min():.3f}, {X_test.max():.3f})")
    print(f"y_test shape: {y_test.shape}, range=({y_test.min():.3f}, {y_test.max():.3f})")

    # ------------------------------------------------------------
    # Load and resample LF volume
    # ------------------------------------------------------------
    import os
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt

    # -----------------------------
    # Helper functions
    # -----------------------------
    def normalize_volume(volume, method='minmax'):
        if method == 'minmax':
            vmin, vmax = np.min(volume), np.max(volume)
            if vmax - vmin > 0:
                return (volume - vmin) / (vmax - vmin)
            else:
                return np.zeros_like(volume)
        elif method == 'zscore':
            mean, std = np.mean(volume), np.std(volume)
            return (volume - mean) / (std + 1e-8)
        else:
            raise ValueError("Unknown normalization method.")

    # -----------------------------
    # Load LF MRI data
    # -----------------------------
    data_folder = 'Data/MW/VLF_invivo/raw'

    # Find the first .nii, .nii.gz, or .npy file
    files = [f for f in os.listdir(data_folder) if f.endswith(('.nii', '.nii.gz', '.npy'))]
    if not files:
        raise FileNotFoundError("No NIfTI or NPY files found in the specified folder.")

    file_path = os.path.join(data_folder, files[0])
    print(f"📂 Loading file: {file_path}")

    # Load image
    if file_path.endswith('.npy'):
        im = np.load(file_path)
        affine = np.eye(4)
    else:
        img = nib.load(file_path)
        im = img.get_fdata()
        affine = img.affine

    print(f"Subject data shape: {im.shape}")

    # Normalize
    im = np.abs(im)
    im = normalize_volume(im, method='minmax')
    print(f"✅ Normalized volume shape: {im.shape}, range=({im.min():.4f}, {im.max():.4f})")
    
    # -----------------------------
    # Prepare data for model
    # -----------------------------
    y_test = im.copy()  # Ground truth same shape as input
    im = np.expand_dims(im, axis=(0))   # (1, H, W, D, 1)
    y_test = np.expand_dims(y_test, axis=(0))  # (1, H, W, D, 1)

    print(f"🧠 Final LF input shape: {im.shape}, y_test shape: {y_test.shape}")

    im = im.astype(np.float32)
    y_test = y_test.astype(np.float32)
    # -----------------------------
    # Visualization
    # -----------------------------
    midz = im.shape[3] // 2
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(im[0, :, :, midz], cmap='gray')
    plt.title("Input LF Volume")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(y_test[0, :, :, midz], cmap='gray')
    plt.title("Ground Truth (same as input)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Evaluate model
    # -----------------------------
    results = evaluate_model(
        folder_path=folder_path,
        model_name=model_name,
        X_test=im,
        y_test=y_test,
        patch_size=(64, 64, 32),
        overlap=0.5,
        visualize_slices=[midz]
    )

    print("\n✅ Model evaluation complete.")
