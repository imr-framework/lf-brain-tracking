import sys
sys.path.insert(0, './')
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from src_niv.metrics import psnr, ssim, mse, composite_loss
from src_niv.utils import visualize_pair
from src_simulated.train_scripts.niv_srr_simulated_training_patch import (
    load_data_for_days,
    normalize_dataset,
    srr_batch_generator,
    build_or_load_model,
    compile_model,
    train_model
)

# -----------------------------
# PARAMETERS
# -----------------------------
data_folder = "niv_raw_data/Nipah_IRF_data/data_niv/Low_field_data_DA"
subjects = ["26184", "30366", "35528", "34507", "35547", "59228", "59877", "59233"]
train_day = 1
val_day = 2
test_days = [3, 4, 5]
output_path = "Output"
checkpoint_path = os.path.join(output_path, "residual_srr_unet_patch_checkpoint.keras")
refined_model_name = "residual_srr_unet_patch_checkpoint"
#residual_srr_unet_patch_checkpoint_checkpoint -- for patch
os.makedirs(output_path, exist_ok=True)

batch_size = 32
patch_xy, patch_z = 64, 32
steps_per_epoch = 24
epochs = 1000  # fine-tuning for fewer epochs

# -----------------------------
# LOAD PREVIOUS MODEL
# -----------------------------
print(f"✅ Loading pretrained model from checkpoint: {checkpoint_path}")
base_model = load_model(checkpoint_path, custom_objects={
    "psnr": psnr,
    "ssim": ssim,
    "mse": mse,
    "composite_loss": composite_loss
})

# -----------------------------
# LOAD DATA
# -----------------------------
X_train, y_train = load_data_for_days(subjects, [train_day])
X_val, y_val = load_data_for_days(subjects, [val_day])

X_train, y_train = normalize_dataset(X_train, y_train)
X_val, y_val = normalize_dataset(X_val, y_val)

print(f"📦 Training volumes: {len(X_train)} | Validation volumes: {len(X_val)}")

# -----------------------------
# GENERATE PSEUDO-ENHANCED INPUTS
# -----------------------------
def predict_refined_inputs(model, lf_volumes, patch_size=(64,64,32), overlap=0.5):
    """Apply model prediction volume-wise to generate refined inputs."""
    refined_vols = []
    for i, lf in enumerate(lf_volumes):
        print(f"🔁 Predicting refined input for volume {i+1}/{len(lf_volumes)} ...")
        H, W, D = lf.shape
        px, py, pz = patch_size

        sx = max(1, int(px*(1-overlap)))
        sy = max(1, int(py*(1-overlap)))
        sz = max(1, int(pz*(1-overlap)))

        lf_padded = np.pad(lf, ((0, px), (0, py), (0, pz)), mode='reflect')
        pred_volume = np.zeros_like(lf_padded)
        count_volume = np.zeros_like(lf_padded)

        for x in range(0, lf_padded.shape[0] - px + 1, sx):
            for y in range(0, lf_padded.shape[1] - py + 1, sy):
                for z in range(0, lf_padded.shape[2] - pz + 1, sz):
                    patch = lf_padded[x:x+px, y:y+py, z:z+pz]
                    patch_in = np.expand_dims(patch, axis=(0,-1))
                    pred_patch = model.predict(patch_in, verbose=0)
                    pred_patch = np.squeeze(pred_patch)
                    pred_volume[x:x+px, y:y+py, z:z+pz] += pred_patch
                    count_volume[x:x+px, y:y+py, z:z+pz] += 1.0

        pred_volume /= np.maximum(count_volume, 1e-5)
        refined_vols.append(pred_volume[:H, :W, :D])

    return np.array(refined_vols, dtype=np.float32)

# 🔁 Generate refined training and validation inputs
lf_train_refined = predict_refined_inputs(base_model, X_train)
lf_val_refined = predict_refined_inputs(base_model, X_val)

import matplotlib.pyplot as plt
import random

def show_refinement_slices(originals, refined, n=3, axis=2):
    """
    Visualize random slices comparing LF original vs. refined model output.
    Args:
        originals: np.array of LF input volumes (X_train)
        refined: np.array of refined volumes (lf_train_refined)
        n: number of samples to display
        axis: slicing axis (0=axial, 1=coronal, 2=sagittal)
    """
    indices = random.sample(range(len(originals)), n)
    plt.figure(figsize=(10, 4 * n))

    for i, idx in enumerate(indices):
        orig = originals[idx]
        refn = refined[idx]

        # pick a central slice along the given axis
        slice_idx = orig.shape[axis] // 2
        orig_slice = np.take(orig, slice_idx, axis=axis)
        refn_slice = np.take(refn, slice_idx, axis=axis)

        vmin = min(orig_slice.min(), refn_slice.min())
        vmax = max(orig_slice.max(), refn_slice.max())

        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(orig_slice, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f"Original LF Volume {idx}")
        plt.axis('off')

        plt.subplot(n, 2, 2*i + 2)
        plt.imshow(refn_slice, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f"Refined (Model Output) Volume {idx}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# 🔍 Show side-by-side comparison
show_refinement_slices(X_train, lf_train_refined, n=3, axis=2)

# -----------------------------
# BUILD & COMPILE NEW MODEL
# -----------------------------
refined_checkpoint_path = os.path.join(output_path, f"{refined_model_name}_checkpoint.keras")
refined_model = build_or_load_model(lambda input_shape: base_model, refined_checkpoint_path)
refined_model = compile_model(refined_model, lr=1e-4)

# -----------------------------
# GENERATORS
# -----------------------------
train_gen = srr_batch_generator(
    lf_volumes=lf_train_refined,
    hf_volumes=y_train,
    batch_size=batch_size,
    patch_xy=patch_xy,
    patch_z=patch_z,
    augment=True
)

val_gen = srr_batch_generator(
    lf_volumes=lf_val_refined,
    hf_volumes=y_val,
    batch_size=8,
    patch_xy=patch_xy,
    patch_z=patch_z,
    augment=False
)

# -----------------------------
# TRAIN REFINEMENT MODEL
# -----------------------------
print("🚀 Starting refinement (second-pass) training ...")

history = train_model(
    train_gen,
    val_gen,
    refined_model,
    refined_checkpoint_path,
    steps_per_epoch=steps_per_epoch,
    validation_steps=max(1, len(X_val)),
    epochs=epochs
)

final_refined_path = os.path.join(output_path, f"{refined_model_name}_final.keras")
refined_model.save(final_refined_path)
print(f"✅ Refined model saved at: {final_refined_path}")
