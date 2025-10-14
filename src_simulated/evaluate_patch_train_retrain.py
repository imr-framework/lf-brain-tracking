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


# ------------------------------------------------
# Visualization Utility
# ------------------------------------------------
import matplotlib.pyplot as plt

def visualize_results(model_path, lf, hf, pred, pred2=None, slices=[20, 40, 60], title_prefix=""):
    for s in slices:
        plt.figure(figsize=(16, 5))
        num_subplots = 4 if pred2 is not None else 3

        plt.subplot(1, num_subplots, 1)
        plt.imshow(lf[:, :, s], cmap='gray')
        plt.title(f"LF slice {s}")

        plt.subplot(1, num_subplots, 2)
        plt.imshow(hf[:, :, s], cmap='gray')
        plt.title("HF Ground Truth")

        plt.subplot(1, num_subplots, 3)
        plt.imshow(pred[:, :, s], cmap='gray')
        plt.title("Predicted (1-pass)")

        if pred2 is not None:
            plt.subplot(1, num_subplots, 4)
            plt.imshow(pred2[:, :, s], cmap='gray')
            plt.title("Re-enhanced (2-pass)")

        # Add model_path to the title for context
        plt.suptitle(f"{title_prefix} Slice {s}\nModel: {model_path}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        plt.show()


# ------------------------------------------------
# Iterative Enhancement Pipeline
# ------------------------------------------------
def evaluate_model(model_path, X_test, y_test,
                   patch_size=(64,64,32), overlap=0.5,
                   iterative_refine=False, visualize_slices=[20,40,60]):

    # Load model
    print(f"🔹 Loading model: {model_path}")
    model = load_model(model_path,
                       custom_objects={'psnr': psnr, 'ssim': ssim,
                                       'mse': mse, 'composite_loss': composite_loss},
                       compile=False)
    print("✅ Model loaded.")

    results = []
    for i in range(len(X_test)):
        print(f"\n🧠 Evaluating subject {i+1}/{len(X_test)} ...")
        lf = X_test[i]
        hf = y_test[i]

        # 1st pass prediction
        pred = predict_volume(model, lf, patch_size=patch_size, overlap=overlap)

        # Optional iterative refinement
        if iterative_refine:
            print("🔁 Performing second-pass refinement...")
            pred2 = predict_volume(model, pred, patch_size=patch_size, overlap=overlap)
        else:
            pred2 = None

        # Compute metrics
        psnr_val = psnr(tf.convert_to_tensor(hf[np.newaxis,...,np.newaxis]),
                        tf.convert_to_tensor(pred[np.newaxis,...,np.newaxis])).numpy()
        ssim_val = ssim(tf.convert_to_tensor(hf[np.newaxis,...,np.newaxis]),
                        tf.convert_to_tensor(pred[np.newaxis,...,np.newaxis])).numpy()
        print(f"📈 PSNR: {psnr_val:.3f}, SSIM: {ssim_val:.4f}")

        # Visualization
        visualize_results(model_path, lf, hf, pred, pred2, slices=visualize_slices, title_prefix=f"Subject {i}")

        results.append({
            'subject': i,
            'psnr': psnr_val,
            'ssim': ssim_val
        })

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
import os
import numpy as np

# ------------------------------------------------
if __name__ == "__main__":
    # Paths
    folder_path = "Output_patch"

    # Loop through all items in the folder
    for item in os.listdir(folder_path):
        if item.endswith(".csv"):
            continue  # 🚫 Skip CSV files

        if "retrained" in item.lower():
            continue  # 🚫 Skip CSV files and retrained items
        
        model_path = os.path.join(folder_path, item)
        print(model_path)

        # Load test data
        data_folder = "Data/data_sim_check/3T_1simulated_LF/train_test"
        subjects = ["26184", "30366", "35528", "34507", "35547", "59228", "59877", "59233"]
        test_days = [3]

        def load_subject_day_data(subject, day):
            path = os.path.join(data_folder, f"{subject}_day{day}_train_data.npy")
            if not os.path.exists(path): return None
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

        results = evaluate_model(
            model_path=model_path,
            X_test=X_test,
            y_test=y_test,
            patch_size=(64,64,32),
            overlap=0.0,
            iterative_refine=False,
            visualize_slices=[15]
        )

        # Print summary
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        print(f"\n✅ Evaluation Complete. Mean PSNR: {avg_psnr:.3f}, Mean SSIM: {avg_ssim:.4f}")