import sys
sys.path.insert(0, './') 

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.keras.metrics import MeanSquaredError
from src_niv.models.ResUNet import residual_srr_unet
from src_niv.models.DenseUNet import build_dense_unet_3d
from src_niv.models.Inception import build_res_inception_unet_3d
from src_niv.metrics import psnr, ssim, mse, composite_loss
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from src_niv.utils import visualize_pair
from src_simulated.utils_simulated import *
from src_simulated.losses import *

# l1_loss
# l2_loss
# ssim_loss
# l1_l2_ssim_loss
# l2_ssim_loss
# l1_ssim_loss
# mse_ssim_edge_loss

# -----------------------------
# PARAMETERS
# -----------------------------
from config import config

print(config.model_name)

data_folder = "Data/data_sim_check/3T_1simulated_LF/train_test"
subjects = ["26184", "30366", "35528","34507", "35547", "59228", "59877","59233"]
train_day = 1
val_day = 2
test_days = [3, 4, 5]
model_type = residual_srr_unet
model_name = "residual_srr_unet"
output_path = "Output_patch"
os.makedirs(output_path, exist_ok=True)
batch_size = 32
patch_z, patch_xy = 32, 64
visualize = False
angles = [10, 20, 25, -10, -20, -25]

# -----------------------------
# DATA LOADING
# -----------------------------
def load_subject_day_data(subject, day, folder=data_folder):
    file_path = os.path.join(folder, f"{subject}_day{day}_train_data.npy")
    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True).item()
        X = data["x_train"].astype(np.float32)[np.newaxis, ...]
        y = data["y_train"].astype(np.float32)[np.newaxis, ...]
        return X, y
    else:
        raise FileNotFoundError(f"{file_path} does not exist!")

def load_data_for_days(subjects, days):
    X_list, y_list = [], []
    for subject in subjects:
        for day in days:
            try:
                X_sub, y_sub = load_subject_day_data(subject, day)
                X_list.append(X_sub)
                y_list.append(y_sub)
            except FileNotFoundError:
                print(f"[WARNING] Missing data for subject {subject}, day {day}")
    if X_list:
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y
    else:
        return None, None

# -----------------------------
# PADDING UTILITIES
# -----------------------------
def pad_volume_to_multiple(vol, multiple=8):
    h, w, d = vol.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_d = (multiple - d % multiple) % multiple

    pad_before = (pad_h//2, pad_w//2, pad_d//2)
    pad_after = (pad_h - pad_before[0], pad_w - pad_before[1], pad_d - pad_before[2])

    padded_vol = np.pad(vol, ((pad_before[0], pad_after[0]),
                              (pad_before[1], pad_after[1]),
                              (pad_before[2], pad_after[2])),
                        mode='constant', constant_values=0)
    pad_widths = (pad_before, pad_after)
    return padded_vol, pad_widths

def unpad_volume(vol, pad_widths):
    (h_before, w_before, d_before), (h_after, w_after, d_after) = pad_widths
    return vol[h_before:vol.shape[0]-h_after,
               w_before:vol.shape[1]-w_after,
               d_before:vol.shape[2]-d_after]

# -----------------------------
# ROTATION AUGMENTATION
# -----------------------------
def rotation_3d(volume, angle, axes=(0,1)):
    return rotate(volume, angle, axes=axes, reshape=False, mode='reflect')

# -----------------------------
# CUSTOM BATCH GENERATOR
# -----------------------------
import numpy as np, random

def srr_batch_generator(
        lf_volumes, hf_volumes,
        batch_size=32,
        patch_xy=64, patch_z=32,
        patches_per_volume=8,
        augment=True,
        augmentations_per_patch=3,
        angles=(0, 5, 10, 15, 20),
        jitter=5,
        intensity_aug=False,
        noise_std=0.01
    ):
    """
    3D SRR Patch Generator for patch-based training.

    Parameters:
    -----------
    lf_volumes, hf_volumes : np.ndarray
        Arrays of shape (N, H, W, D).
    batch_size : int
        Number of total samples per batch.
    patches_per_volume : int
        Random patches extracted per volume.
    augmentations_per_patch : int
        Number of augmentations per patch.
    """
    n_samples = lf_volumes.shape[0]
    volume_order = np.random.permutation(n_samples)
    vol_idx = 0

    while True:
        x_batch, y_batch = [], []

        while len(x_batch) < batch_size:
            idx = volume_order[vol_idx]
            lf = lf_volumes[idx]
            hf = hf_volumes[idx]

            vol_idx = (vol_idx + 1) % n_samples
            if vol_idx == 0:
                volume_order = np.random.permutation(n_samples)

            lf, _ = pad_volume_to_multiple(lf, multiple=8)
            hf, _ = pad_volume_to_multiple(hf, multiple=8)
            H, W, D = lf.shape

            for _ in range(patches_per_volume):
                # Extract random patch
                z_start = np.random.randint(0, max(1, D - patch_z))
                y_start = np.random.randint(0, max(1, H - patch_xy))
                x_start = np.random.randint(0, max(1, W - patch_xy))

                lf_patch = lf[y_start:y_start + patch_xy,
                              x_start:x_start + patch_xy,
                              z_start:z_start + patch_z]
                hf_patch = hf[y_start:y_start + patch_xy,
                              x_start:x_start + patch_xy,
                              z_start:z_start + patch_z]

                # Add original patch
                x_batch.append(np.expand_dims(lf_patch, axis=-1))
                y_batch.append(np.expand_dims(hf_patch, axis=-1))

                # Add multiple augmentations per patch
                if augment:
                    for _ in range(augmentations_per_patch):
                        aug_lf, aug_hf = augment_patch_pair(
                            lf_patch.copy(), hf_patch.copy(),
                            angles, jitter, intensity_aug, noise_std
                        )
                        x_batch.append(np.expand_dims(aug_lf, axis=-1))
                        y_batch.append(np.expand_dims(aug_hf, axis=-1))

        # Stack and trim
        x_batch = np.stack(x_batch[:batch_size], axis=0)
        y_batch = np.stack(y_batch[:batch_size], axis=0)
        yield x_batch.astype(np.float32), y_batch.astype(np.float32)


# Helper for augmentations
def augment_patch_pair(lf_patch, hf_patch, angles, jitter, intensity_aug, noise_std):
    base_angle = random.choice(angles)
    sign = random.choice([-1, 1])
    angle = sign * (base_angle + random.uniform(0, jitter))
    axes = random.choice([(0, 1), (0, 2), (1, 2)])

    aug_lf = rotation_3d(lf_patch, angle, axes)
    aug_hf = rotation_3d(hf_patch, angle, axes)

    if random.random() < 0.5:
        flip_axis = random.choice([0, 1, 2])
        aug_lf = np.flip(aug_lf, axis=flip_axis)
        aug_hf = np.flip(aug_hf, axis=flip_axis)

    if intensity_aug:
        scale = 1.0 + 0.2 * (np.random.rand() - 0.5)
        aug_lf *= scale
        aug_hf *= scale

        mean_lf = aug_lf.mean()
        mean_hf = aug_hf.mean()
        contrast_factor = 1.0 + 0.2 * (np.random.rand() - 0.5)
        aug_lf = (aug_lf - mean_lf) * contrast_factor + mean_lf
        aug_hf = (aug_hf - mean_hf) * contrast_factor + mean_hf

        aug_lf += np.random.normal(0, noise_std, aug_lf.shape)
        aug_hf += np.random.normal(0, noise_std, aug_hf.shape)

        aug_lf = np.clip(aug_lf, 0, np.max(aug_lf))
        aug_hf = np.clip(aug_hf, 0, np.max(aug_hf))

    return aug_lf, aug_hf

# -----------------------------
# MODEL BUILD / COMPILE / TRAIN
# -----------------------------
def build_or_load_model(model_type, checkpoint_path, input_shape=(64,64,32,1)):
    if os.path.exists(checkpoint_path):
        print(f"✅ Loading model from checkpoint: {checkpoint_path}")
        model = load_model(checkpoint_path, compile=False)
    else:
        print("⚠️ Checkpoint not found. Building new model...")
        model = model_type(config.input_shape)
    return model

# Define custom PSNR and SSIM metrics
def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def compile_model(model, lr=0.001, loss_type='l1_l2_ssim'):
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    mse_metric = MeanSquaredError(name='mse')

    # --- Select Loss ---
    if loss_type == 'l1':
        loss_fn = l1_loss
    elif loss_type == 'l2':
        loss_fn = l2_loss
    elif loss_type == 'ssim':
        loss_fn = ssim_loss
    elif loss_type == 'l1_l2_ssim':
        loss_fn = l1_l2_ssim_loss
    elif loss_type == 'l2_ssim':
        loss_fn = l2_ssim_loss
    elif loss_type == 'l1_ssim':
        loss_fn = l1_ssim_loss
    elif loss_type == 'mse_ssim_edge':
        loss_fn = mse_ssim_edge_loss
    else:
        raise ValueError("❌ Invalid loss_type. Choose from: ['l1', 'l2', 'ssim', 'l1_ssim', 'l2_ssim', 'l1_l2_ssim', 'mse_ssim_edge'].")

    # --- Compile ---
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[mse_metric,
                 psnr_metric,
                 ssim_metric]
    )

    print(f"✅ Model compiled with {loss_type.upper()} loss and lr={lr}")
    return model


def train_model(train_gen, val_gen, model, checkpoint_path,
                steps_per_epoch=30, validation_steps=8, epochs=500):
    """
    Train model with checkpointing, learning rate scheduling, early stopping, and CSV logging.
    """
    # 🔹 Define CSV log path (same name as model checkpoint)
    csv_log_path = os.path.join(output_path, os.path.splitext(os.path.basename(checkpoint_path))[0] + "_training_log.csv")

    # 🔹 Callbacks
    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True
    )

    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=30,
        mode='min',
        verbose=1,
        min_lr=1e-8
    )

    early_stop_cb = EarlyStopping(
        monitor='val_ssim',
        patience=60,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    csv_logger_cb = CSVLogger(csv_log_path, append=False)

    # 🔹 Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb, csv_logger_cb]
    )

    print(f"📊 Training log saved at: {csv_log_path}")
    return history


# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_on_sample(model, lf_volume, hf_volume,
                       patch_size=(64,64,32),
                       overlap=0.5,
                       slice_indices=None):
    """
    Evaluate a 3D model on any LF volume using sliding-window patch inference.

    Args:
        model: Trained 3D model
        lf_volume: Low-field input volume (H,W,D)
        hf_volume: High-field ground truth volume (H,W,D)
        patch_size: Tuple (px, py, pz) for network input
        overlap: Fractional overlap between patches (0-1)
        slice_indices: List of slice indices along Z-axis to visualize

    Returns:
        pred_volume: Predicted SRR volume, same shape as lf_volume
    """
    import numpy as np
    import matplotlib.pyplot as plt

    lf_volume = lf_volume.astype(np.float32)
    hf_volume = hf_volume.astype(np.float32)
    H, W, D = lf_volume.shape
    px, py, pz = patch_size

    # Compute strides
    sx = max(1, int(px*(1-overlap)))
    sy = max(1, int(py*(1-overlap)))
    sz = max(1, int(pz*(1-overlap)))

    # Pad volume to fit integer number of patches
    pad_x = (px - H % px) if H % px != 0 else 0
    pad_y = (py - W % py) if W % py != 0 else 0
    pad_z = (pz - D % pz) if D % pz != 0 else 0

    lf_padded = np.pad(lf_volume, ((0,pad_x),(0,pad_y),(0,pad_z)), mode='reflect')
    hf_padded = np.pad(hf_volume, ((0,pad_x),(0,pad_y),(0,pad_z)), mode='reflect')

    H_pad, W_pad, D_pad = lf_padded.shape
    pred_volume = np.zeros_like(lf_padded, dtype=np.float32)
    count_volume = np.zeros_like(lf_padded, dtype=np.float32)

    # Sliding window inference
    for x in range(0, H_pad - px + 1, sx):
        for y in range(0, W_pad - py + 1, sy):
            for z in range(0, D_pad - pz + 1, sz):
                patch = lf_padded[x:x+px, y:y+py, z:z+pz]
                patch_input = np.expand_dims(patch, axis=(0,-1))  # (1, px, py, pz, 1)
                pred_patch = model.predict(patch_input)
                pred_patch = np.squeeze(pred_patch)

                pred_volume[x:x+px, y:y+py, z:z+pz] += pred_patch
                count_volume[x:x+px, y:y+py, z:z+pz] += 1.0

    # Average overlapping regions
    pred_volume /= count_volume

    # Crop back to original size
    pred_volume = pred_volume[:H, :W, :D]

    # Visualization
    if slice_indices is None:
        slice_indices = [D//2]

    for s in slice_indices:
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(lf_volume[:,:,s], cmap='gray')
        plt.title(f"LF slice {s}")
        plt.subplot(1,3,2)
        plt.imshow(hf_volume[:,:,s], cmap='gray')
        plt.title(f"HF slice {s}")
        plt.subplot(1,3,3)
        plt.imshow(pred_volume[:,:,s], cmap='gray')
        plt.title(f"Predicted SRR slice {s}")
        plt.show()

    return pred_volume

# -----------------------------
# Training function
# -----------------------------
def run_training(lf_train, hf_train, lf_val, hf_val,
                 output_path, model_type=residual_srr_unet, model_name = 'ResUNet',
                 loss_type = 'l1_l2_ssim', patch_xy=64, patch_z=32,
                 batch_size=32, steps_per_epoch=48, epochs=500):
    
    import os
    # checkpoint_path = os.path.join(output_path, f"{model_name}_checkpoint.keras")
    model = build_or_load_model(model_type, config.checkpoint_path)
    model = compile_model(model, lr=0.001, loss_type=loss_type)
    
    # Generators
    train_gen = srr_batch_generator(
        lf_volumes=lf_train, 
        hf_volumes=hf_train,
        batch_size=batch_size,
        patch_xy=patch_xy,
        patch_z=patch_z,
        augment=config.train_augment
    )

    val_gen = srr_batch_generator(
        lf_volumes=lf_val,
        hf_volumes=hf_val,
        batch_size=8,         # validation smaller
        patch_xy=patch_xy,
        patch_z=patch_z,
        augment=config.val_augment
    )

    # Steps
    # steps_per_epoch = max(1, len(lf_train) * 8 // batch_size)  # 4 patches per volume
    validation_steps = max(1, len(lf_val) // 1)

    # Train
    history = train_model(
        train_gen,
        val_gen,
        model,
        config.checkpoint_path,
        steps_per_epoch=steps_per_epoch,
        validation_steps = validation_steps,
        epochs=epochs
    )

    # Save final model
    final_model_path = os.path.join(output_path, f"{model_name}_final.keras")
    model.save(final_model_path)
    print(f"✅ Final model saved at: {final_model_path}")

    return model, history

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

def show_refinement_slices(originals, refined, n=3, axis=2):
    """Visualize random slices comparing LF original vs. refined model output."""
    indices = random.sample(range(len(originals)), n)
    plt.figure(figsize=(10, 4 * n))

    for i, idx in enumerate(indices):
        orig = originals[idx]
        refn = refined[idx]
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
        plt.title(f"Refined Output Volume {idx}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def run_retraining(checkpoint_path,
                   refined_model_name,
                   X_train, y_train, 
                   X_val, y_val, 
                   output_path,
                   batch_size=32,
                   loss_type = 'l2_ssim_edge',
                   patch_xy=64,
                   patch_z=32,
                   steps_per_epoch=34,
                   epochs=1000,
                   visualize=False):
    
    """
    Run second-pass retraining with refined LF inputs using a loaded base model checkpoint.
    """
    print(f"📥 Loading base model from checkpoint: {checkpoint_path}")
    base_model = tf.keras.models.load_model(checkpoint_path, compile=False)

    print("🔧 Generating pseudo-enhanced training and validation inputs ...")
    lf_train_refined = predict_refined_inputs(base_model, X_train)
    lf_val_refined   = predict_refined_inputs(base_model, X_val)

    if visualize:
        print(" Visualizing refinement results ...")
        show_refinement_slices(X_train, lf_train_refined, n=3, axis=2)

    
    refined_checkpoint_path = os.path.join(output_path, f"{refined_model_name}_checkpoint.keras")

    print(" Building and compiling refined model ...")
    refined_model = build_or_load_model(lambda input_shape: base_model, refined_checkpoint_path)
    refined_model = compile_model(refined_model, lr=1e-3,loss_type = loss_type)

    print(" Creating data generators ...")
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

    return refined_model, history, lf_train_refined, lf_val_refined

def compare_four_volumes(vol1, vol2, vol3, vol4,
                         labels=('Input LF', 'Refined LF', 'Refined Output', 'Target HF'),
                         axis=2, n_slices=3, normalize=True):
    """
checkpoint    Display corresponding slices from four 3D volumes in a grid for comparison.

    Args:
        vol1, vol2, vol3, vol4: 3D numpy arrays of same shape (H, W, D).
        labels: tuple of labels for each volume.
        axis: axis to slice along (0=axial, 1=coronal, 2=sagittal).
        n_slices: number of slices to display.
        normalize: if True, min-max normalize slices for display.
    """
    assert vol1.shape == vol2.shape == vol3.shape == vol4.shape, "All volumes must have same shape."

    depth = vol1.shape[axis]
    slice_indices = np.linspace(depth // 4, 3 * depth // 4, n_slices, dtype=int)

    fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4 * n_slices))
    if n_slices == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, s in enumerate(slice_indices):
        # Extract slices along selected axis
        slices = [np.take(v, s, axis=axis) for v in [vol1, vol2, vol3, vol4]]

        # Normalize if requested
        if normalize:
            slices = [(sl - np.min(sl)) / (np.max(sl) - np.min(sl) + 1e-8) for sl in slices]

        vmin = min(sl.min() for sl in slices)
        vmax = max(sl.max() for sl in slices)

        # Plot each volume slice
        for j, (sl, lbl) in enumerate(zip(slices, labels)):
            ax = axes[i, j]
            ax.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"{lbl}\nSlice {s}", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    
    # Load data
    # 🎯 Define selected combinations (denoise → retrain)
    selected_combinations = [
        ("l1_l2_ssim", "mse_ssim_edge"),
        ("l2_ssim", "l2_ssim"),
        ("l1_l2_ssim", "l2_ssim"),
        ("l2_ssim", "mse_ssim_edge")
        ]
    
    # 🌀 Iterate over chosen combinations
    for loss_denoise, loss_retrain in selected_combinations:
        # Update config
        config.loss_type_denoise = loss_denoise
        config.retrain_loss_type = loss_retrain
        # config.loss_weights_denoise = loss_weights[loss_denoise]
        # config.loss_weights_retrain = loss_weights[loss_retrain]
        
        # Combine dynamically
        config.model_name = f"{model_name}_{config.loss_type_denoise}_{config.retrain_loss_type}"
        config.checkpoint_path = os.path.join(config.output_path, f"{config.model_name}_checkpoint.keras")
        config.refined_model_name = f"{config.model_name}_retrained"
        print("------------------------------------------------------\n")

        # 🪶 Print summary
        print(f"🚀 Configuration:")
        print(f"   Model Name: {config.model_name}")
        print(f"   Denoise Loss:  {loss_denoise} ")
        print(f"   Retrain Loss:  {loss_retrain} ")
        print(f"   checkpoint_path:  {config.checkpoint_path} ")
        print("------------------------------------------------------\n")


        # Load Data
        X_train, y_train = load_data_for_days(config.subjects, [config.train_day])
        X_val, y_val     = load_data_for_days(config.subjects, [config.val_day])
        X_test, y_test   = load_data_for_days(config.subjects, config.test_days)

        X_train, y_train = normalize_dataset(X_train, y_train)
        X_val, y_val     = normalize_dataset(X_val, y_val)
        X_test, y_test   = normalize_dataset(X_test, y_test)

        # -----------------------------
        # Print shapes for confirmation
        # -----------------------------

        print("\n✅ Dataset normalization complete.")
        print(f"🧩 X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"🧩 X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
        print(f"🧩 X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
        print(f"📦 Total training volumes: {len(X_train)}")
        print(f"📦 Total validation volumes: {len(X_val)}")
        print(f"📦 Total testing volumes: {len(X_test)}\n")

        # Train model
        trained_model, history = run_training(
            lf_train=X_train,
            hf_train=y_train,
            lf_val=X_val,
            hf_val=y_val,
            output_path=config.output_path,
            model_type=residual_srr_unet,
            model_name=config.model_name,
            loss_type=config.loss_type_denoise,
            patch_xy=config.patch_xy,
            patch_z=config.patch_z,
            batch_size=config.batch_size,
            steps_per_epoch=config.steps_per_epoch,
            epochs=config.epochs
        )

        # Refinement (2nd pass)
        refined_model, history, lf_train_refined, lf_val_refined = run_retraining(
            checkpoint_path=config.checkpoint_path,
            refined_model_name=config.refined_model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_path=config.output_path,
            batch_size=config.retrain_batch_size,
            loss_type=config.retrain_loss_type,
            patch_xy=config.patch_xy,
            patch_z=config.patch_z,
            steps_per_epoch=config.retrain_steps_per_epoch,
            epochs=config.retrain_epochs,
            visualize=config.visualize
        )
        
        refined_model_name = config.refined_model_name
        refined_checkpoint_path = os.path.join(config.output_path, f"{refined_model_name}_checkpoint.keras")
        print(f"📥 Loading base model from checkpoint: {refined_checkpoint_path}")
        base_model = tf.keras.models.load_model(refined_checkpoint_path, compile=False)

        print("🔧 Generating pseudo-enhanced training and validation inputs ...")
        lf_train_refined_output = predict_refined_inputs(base_model, lf_train_refined)
        
        # # # Visual comparison across 4 stages for one subject
        # # compare_four_volumes(
        # #     X_train[0],
        # #     lf_train_refined[0],
        # #     lf_train_refined_output[0],
        # #     y_train[0],
        # #     labels=('LF Input', 'Denoise Output', 'SRR Output', 'HF Target'),
        # #     axis=2,
        # #     n_slices=3
        # # )
        
        # # # Evaluate example on a single LF volume
        # # pred_volume = evaluate_on_sample(
        # #     model=trained_model,
        # #     lf_volume=X_test[0],
        # #     hf_volume=y_test[0],
        # #     patch_size=(64,64,32),
        # #     overlap=0.5,
        # #     slice_indices=[10,20,30]
        # # )