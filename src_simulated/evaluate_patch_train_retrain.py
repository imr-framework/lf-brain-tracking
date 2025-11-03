import sys
sys.path.insert(0, './') 
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

    # model_path_train = 'Output_patch_noise/residual_srr_unet_l1_l2_ssim_l2_ssim_edge_checkpoint.keras'
    # model_path_retrained = 'Output_patch/residual_srr_unet_l1_l2_ssim_mse_ssim_edge_retrained_checkpoint.keras'

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

if __name__ == "__main__":
    # ------------------------------------------------------------
    # 🔹 Model list and paths
    # ------------------------------------------------------------
    model_names = [
        'residual_srr_unet_l1_l2_ssim_l2_ssim_edge',
        # 'residual_srr_unet_l2_ssim_mse_ssim_edge',
        # 'residual_srr_unet_l1_l2_ssim_l2_ssim',
        # 'residual_srr_unet_l2_ssim_l2_ssim'
        # 'residual_srr_unet_l2_ssim_mse_ssim_edge'
    ]
    folder_path = "Output_patch_noise"

    # ------------------------------------------------------------
    # 🔹 Load test data once
    # ------------------------------------------------------------
    data_folder = "Data/data_sim_check/35528simulated_LF/train_test"
    subjects = ["26184", "30366", "35528", "34507", "35547", "59228", "59877", "59233"]
    test_days = [5]

    def load_subject_day_data(subject, day):
        path = os.path.join(data_folder, f"{subject}_day{day}_train_data.npy")
        if not os.path.exists(path):
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
    X_test, y_test = normalize_dataset(X_test, y_test, method='minmax')

    #gaussian smoothing to reduce noise X_test
    from scipy.ndimage import gaussian_filter
    X_test = np.array([gaussian_filter(vol, sigma=1) for vol in X_test])
    print("✅ Test data normalized.")

    # ------------------------------------------------------------
    # 🔹 Evaluate each model sequentially
    # ------------------------------------------------------------
    for i, model_name in enumerate(model_names, start=1):
        print(f"\n{'='*70}")
        print(f"🚀 Evaluating Model {i}/{len(model_names)} → {model_name}")
        print(f"{'='*70}")

        results = evaluate_model(
            folder_path=folder_path,
            model_name=model_name,
            X_test=X_test,       # evaluate on subset for speed
            y_test=y_test,
            patch_size=(64, 64, 32),
            overlap=0.5,
            visualize_slices=[15]
        )

        # Optional: save results as CSV
        import pandas as pd
        results_path = os.path.join('Output_patch/results', f"results_test_day{test_days}_{model_name}.csv")
        pd.DataFrame(results).to_csv(results_path, index=False)
        print(f"💾 Results saved: {results_path}")

    print("\n✅ All models evaluated successfully.")

        # # Print summary
        # avg_psnr = np.mean([r['psnr'] for r in results])
        # avg_ssim = np.mean([r['ssim'] for r in results])
        # print(f"\n✅ Evaluation Complete. Mean PSNR: {avg_psnr:.3f}, Mean SSIM: {avg_ssim:.4f}")