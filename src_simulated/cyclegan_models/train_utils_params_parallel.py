import sys
sys.path.insert(0, './')

import os
import random
from random import random as rand_func
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, gridspec

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

# Image loading and preprocessing
from keras.preprocessing.image import img_to_array, load_img

# Medical image processing
import nibabel as nib
from scipy.ndimage import zoom

# Load config and losses
from src_simulated.cyclegan_models.config import config
from src_simulated.cyclegan_models.losses_cyclegan import *
from src_simulated.cyclegan_models.models import *

# -----------------------------
# Config Parameters
# -----------------------------
EPOCHS = config.EPOCHS
TEST = config.TEST

# Discriminator parameters
DISC_LOSS = config.DISC_LOSS
DISC_LEARNING_RATE = config.DISC_LEARNING_RATE
DISC_BETA_1 = config.DISC_BETA_1
DISC_LOSS_WEIGHTS = config.DISC_LOSS_WEIGHTS

# Generator parameters
GEN_LOSS_1 = config.GEN_LOSS_1
GEN_LOSS_2 = config.GEN_LOSS_2
GEN_LOSS_3 = config.GEN_LOSS_3
GEN_LOSS_4 = config.GEN_LOSS_4
GEN_LEARNING_RATE = config.GEN_LEARNING_RATE
GEN_BETA_1 = config.GEN_BETA_1
GEN_LOSS_WEIGHTS = config.GEN_LOSS_WEIGHTS

# -----------------------------
# Data Loading
# -----------------------------
def load_real_samples(filename):
    """
    Load training dataset (A and B) from npz file and scale to [-1,1].
    """
    data = np.load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


# -----------------------------
# Model Saving
# -----------------------------
def save_models(step, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B, output_path):
    """
    Save GAN models periodically and as 'latest'.
    """
    os.makedirs(output_path, exist_ok=True)
    models = {"g_AtoB": g_model_AtoB, "g_BtoA": g_model_BtoA, "d_A": d_model_A, "d_B": d_model_B}
    for name, model in models.items():
        model.save(os.path.join(output_path, f"{name}_{step+1:06d}.keras"))
        model.save(os.path.join(output_path, f"{name}_latest.keras"))
    print(f"> Saved models for step {step+1}")


def generate_real_samples_a(dataset_a, dataset_b, context_a, context_b, n_samples, patch_shape):
    
    # independent sampling
    ix_A = np.random.randint(0, dataset_a.shape[0], n_samples)
    ix_B = np.random.randint(0, dataset_b.shape[0], n_samples)
    
    # images
    X = dataset_a[ix_A]   # domain A
    Y = dataset_b[ix_B]   # domain B
    
    # correct context alignment
    contextA = context_a[ix_A]
    contextB = context_b[ix_B]
    
    # labels
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    
    return X, Y, contextA, contextB, y

def generate_fake_samples_a(g_model, dataset, context, patch_shape):
    
    # sanity check
    assert dataset.shape[0] == context.shape[0], "Batch mismatch!"
    
    # generate fake images
    X = g_model.predict([dataset, context], verbose=0)
    
    # fake labels
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    
    return X, y

# -----------------------------
# Performance Visualization
# -----------------------------
def summarize_performance(step, g_model, g_inverse, gen_A, gen_B, name, n_samples=5, output_path=None):
    """
    Visualize input, generated, reconstructed, and error images with metrics.
    """
    os.makedirs(output_path, exist_ok=True)

    # Sample n_samples from each domain
    X_A, c_A, _, _ = generate_real_samples_from_gen(gen_A, n_slices=n_samples)
    X_B, c_B, _, _ = generate_real_samples_from_gen(gen_B, n_slices=n_samples)

    # Depending on the direction (A->B or B->A)
    
    X_in, Y_in, context_in, context_inv = X_A, X_B, c_A, c_B

    # Generate fake and reconstructed images
    X_out, _ = generate_fake_samples_from_gen(g_model, X_in, context_in)
    X_rec, _ = generate_fake_samples_from_gen(g_inverse, X_out, context_inv)

    # Scale [-1,1] -> [0,1]
    X_in = (X_in + 1) / 2.0
    Y_in = (Y_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    X_rec = (X_rec + 1) / 2.0

    diff_Y = np.abs(Y_in - X_out)
    diff_X = np.abs(X_in - X_rec)

    fig = plt.figure(figsize=(n_samples*2, 9))
    gs = gridspec.GridSpec(6, n_samples, wspace=-0.01, hspace=-0.01)

    def plot_img(row, col, img, cmap='gray', vmin=None, vmax=None):
        ax = plt.subplot(gs[row, col])
        ax.axis('off')
        ax.imshow(img.squeeze(), cmap=cmap, vmin=vmin, vmax=vmax)
        return ax

    for i in range(n_samples):
        plot_img(0, i, X_in[i])
        plot_img(1, i, Y_in[i])
        plot_img(2, i, X_out[i])
        ax = plot_img(3, i, diff_Y[i], cmap='hot', vmin=0, vmax=diff_Y[i].max())
        plot_img(4, i, X_rec[i])
        ax2 = plot_img(5, i, diff_X[i], cmap='hot', vmin=0, vmax=diff_X[i].max())

    filename = os.path.join(output_path, f"{name}_summary_{step+1:06d}.png")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"> Saved summary plot: {filename}")

# -----------------------------
# Training Metrics
# -----------------------------
def plot_training_metrics(folder_path, filename='training_metrics.csv', save_plot=True):
    csv_path = os.path.join(folder_path, filename)
    if not os.path.exists(csv_path):
        print(f"[Warning] CSV not found: {csv_path}")
        return 0
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[Warning] CSV empty: {csv_path}")
        return 0

    def smooth(vals, window=50):
        return vals.rolling(window=window, min_periods=1).mean()

    fig, axes = plt.subplots(2, 2, figsize=(16,10))
    axes[0,0].plot(df['iteration'], smooth(df['g_A2B_cycA']), label='Cyc A'); axes[0,0].plot(df['iteration'], smooth(df['g_B2A_cycB']), label='Cyc B')
    axes[0,0].set_title('Cycle Loss'); axes[0,0].legend(); axes[0,0].grid(True, linestyle='--', alpha=0.6)

    axes[0,1].plot(df['iteration'], smooth(df['g_A2B_gan']), label='Gen A2B'); axes[0,1].plot(df['iteration'], smooth(df['g_B2A_gan']), label='Gen B2A')
    axes[0,1].set_title('Generator GAN Loss'); axes[0,1].legend(); axes[0,1].grid(True, linestyle='--', alpha=0.6)

    avg_dA = (df['dA_loss_real'] + df['dA_loss_fake']) / 2
    avg_dB = (df['dB_loss_real'] + df['dB_loss_fake']) / 2
    axes[1,0].plot(df['iteration'], smooth(avg_dA), label='Disc A'); axes[1,0].plot(df['iteration'], smooth(avg_dB), label='Disc B')
    axes[1,0].set_title('Discriminator Loss'); axes[1,0].legend(); axes[1,0].grid(True, linestyle='--', alpha=0.6)

    axes[1,1].plot(df['iteration'], smooth(df['psnr_A']), label='PSNR A'); axes[1,1].plot(df['iteration'], smooth(df['psnr_B']), label='PSNR B')
    axes[1,1].set_title('PSNR'); axes[1,1].legend(); axes[1,1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(f"CycleGAN Training Metrics (Total Iter: {df['iteration'].max()})", fontsize=16)
    if save_plot:
        plt.savefig(os.path.join(folder_path, f'cyclegan_metrics_iter_{int(df["iteration"].max()):06d}.png'), dpi=300)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
    return int(df['iteration'].max())

def generate_real_samples_from_gen(gen, n_slices=1, patch_shape=16):
    """
    Returns N random slices from ONE volume.
    """
    X_vol, c_vol = next(gen)   # (H, W, D) or (1,H,W,D)
    if X_vol.ndim == 4:
        x = X_vol[0]
        c = c_vol[0]
    else:
        x = X_vol
        c = c_vol

    d = x.shape[-1]

    # ---- sample N random slices ----
    if n_slices > d:
        slice_idxs = np.random.choice(d, n_slices, replace=True)
    else:
        slice_idxs = np.random.choice(d, n_slices, replace=False)

    X_batch = np.stack([x[..., idx] for idx in slice_idxs], axis=0)
    X_batch = np.expand_dims(X_batch, axis=-1)  # (N,H,W,1)

    # ---- FIX: context is vector → repeat ----
    c_batch = np.tile(c, (n_slices, 1))  # (N, 9)
    y_batch = np.ones((n_slices, patch_shape, patch_shape, 1))
    
    return X_batch, c_batch, y_batch, slice_idxs

def generate_fake_samples_from_gen(g_model, X_real, c, patch_shape=16):
    """
    Generate fake 2D slices using generator with context.
    """
    X_fake = g_model.predict([X_real, c], verbose=0)
    y = np.zeros((len(X_fake), patch_shape, patch_shape, 1))  # fake labels
    return X_fake, y

def update_image_pool(pool, images, max_size=50):
    """
    Maintain a pool of generated images to stabilize discriminator.
    """
    selected = []
    for img in images:
        if len(pool) < max_size:
            pool.append(img)
            selected.append(img)
        elif rand_func() < 0.5:
            selected.append(img)
        else:
            idx = np.random.randint(0, len(pool))
            selected.append(pool[idx])
            pool[idx] = img
    return np.asarray(selected)

# ------------------
# Main Training Loop
# ------------------
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA,
          c_model_AtoB, c_model_BtoA, gen_small, gen_large, steps_per_epoch, output_dir,
          epochs=300, n_slices=1, initial_lr=0.0002, n_iter=40, n_iter_decay=150, small_domain='A'):

    os.makedirs(output_dir, exist_ok=True)
    poolA, poolB = [], []

    n_batch = 1
    n_patch = d_model_A.output_shape[1]
    bat_per_epo = steps_per_epoch  # min of a or b domain subjects in total
    output_csv = os.path.join(output_dir, "training_metrics.csv")

    # Resume from latest checkpoints if available
    model_files = {
        'g_A2B': os.path.join(output_dir, 'g_AtoB_latest.keras'),
        'g_B2A': os.path.join(output_dir, 'g_BtoA_latest.keras'),
        'd_A': os.path.join(output_dir, 'd_A_latest.keras'),
        'd_B': os.path.join(output_dir, 'd_B_latest.keras')
    }

    start_step = 0
    if all(os.path.exists(f) for f in model_files.values()):
        print(">>> Resuming from latest checkpoints...")
        g_model_AtoB = load_model(model_files['g_A2B'], compile=False)
        g_model_BtoA = load_model(model_files['g_B2A'], compile=False)
        d_model_A = load_model(model_files['d_A'], compile=False)
        d_model_B = load_model(model_files['d_B'], compile=False)
        d_model_A.compile(loss=DISC_LOSS, optimizer=Adam(learning_rate=DISC_LEARNING_RATE, beta_1=DISC_BETA_1), loss_weights=DISC_LOSS_WEIGHTS)
        d_model_B.compile(loss=DISC_LOSS, optimizer=Adam(learning_rate=DISC_LEARNING_RATE, beta_1=DISC_BETA_1), loss_weights=DISC_LOSS_WEIGHTS)
        # Get a sample batch from each generator to determine input shape
        X_sample_A, _ = next(gen_small)
        X_sample_B, _ = next(gen_large)
        c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, X_sample_A.shape, context_dim=5)
        c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, X_sample_B.shape, context_dim=5)

        if os.path.exists(output_csv):
            start_step = int(pd.read_csv(output_csv)['iteration'].max())

    # -----------------------------
    # Training iterations
    # -----------------------------
    for epoch in range(0, epochs):

        print(f"\n===== Epoch {epoch+1}/{epochs} =====")

        # dA_loss_real_epoch, dA_loss_fake_epoch, dB_loss_real_epoch, dB_loss_fake_epoch, \
        # g_A2B_total_epoch, g_A2B_gan_epoch, g_A2B_id_epoch, g_A2B_cycA_epoch, g_A2B_cycB_epoch, \
        # g_B2A_total_epoch, g_B2A_gan_epoch, g_B2A_id_epoch, g_B2A_cycA_epoch, g_B2A_cycB_epoch, total_batches = (0,) * 16

        for step in range(steps_per_epoch):
            # ---- get ONE subject from smaller domain ----
            X_small_slices, c_small_slices, y_small_slices, _ = generate_real_samples_from_gen(gen_small, n_slices=n_slices)
            X_large_slices, c_large_slices, y_large_slices, _ = generate_real_samples_from_gen(gen_large, n_slices=n_slices)

            # ---- loop over slices for CycleGAN style update ----
            for i in range(n_slices):
                print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{steps_per_epoch} | Slice {i+1}/{n_slices}")
                if small_domain == 'A':
                    X_realA = X_small_slices[i:i+1]; cA = c_small_slices[i:i+1]; y_realA = y_small_slices[i:i+1]
                    X_realB = X_large_slices[i:i+1]; cB = c_large_slices[i:i+1]; y_realB = y_large_slices[i:i+1]
                else:
                    X_realB = X_small_slices[i:i+1]; cB = c_small_slices[i:i+1]; y_realB = y_small_slices[i:i+1]
                    X_realA = X_large_slices[i:i+1]; cA = c_large_slices[i:i+1]; y_realA = y_large_slices[i:i+1]

                # ---- generate fake ----
                X_fakeA, y_fakeA = generate_fake_samples_from_gen(g_model_BtoA, X_realB, cB)
                X_fakeB, y_fakeB = generate_fake_samples_from_gen(g_model_AtoB, X_realA, cA)

                # ---- update pools ----
                X_fakeA = update_image_pool(poolA, X_fakeA)
                X_fakeB = update_image_pool(poolB, X_fakeB)

                # -------------------- Train Generators --------------------
                g_model_AtoB.trainable = True
                g_model_BtoA.trainable = True
                d_model_A.trainable = False
                d_model_B.trainable = False

                g_loss1, g_adv1, g_id1, g_cyc1a, g_cyc1b = c_model_AtoB.train_on_batch( [X_realA, cA, X_realB], [y_realB, X_realB, X_realA, X_realB])
                g_loss2, g_adv2, g_id2, g_cyc2a, g_cyc2b = c_model_BtoA.train_on_batch([X_realB, cB, X_realA], [y_realA, X_realA, X_realB, X_realA])

                # -------------------- Train Discriminators --------------------
                g_model_AtoB.trainable = False
                g_model_BtoA.trainable = False
                d_model_A.trainable = True
                d_model_B.trainable = True

                dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
                dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
                dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
                dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

                # # -------------------- Accumulate Losses --------------------
                # dA_loss_real_epoch += dA_loss1
                # dA_loss_fake_epoch += dA_loss2
                # dB_loss_real_epoch += dB_loss1
                # dB_loss_fake_epoch += dB_loss2
                # g_A2B_total_epoch += g_loss1
                # g_A2B_gan_epoch += g_adv1
                # g_A2B_id_epoch += g_id1
                # g_A2B_cycA_epoch += g_cyc1a
                # g_A2B_cycB_epoch += g_cyc1b
                # g_B2A_total_epoch += g_loss2
                # g_B2A_gan_epoch += g_adv2
                # g_B2A_id_epoch += g_id2
                # g_B2A_cycA_epoch += g_cyc2a
                # g_B2A_cycB_epoch += g_cyc2b
                # total_batches += 1

    # # -------------------- Logging & CSV --------------------
    # if epoch % 1 == 0:

    #     loss_row = { "iteration": epoch+1, "dA_loss_real": dA_loss_real_epoch/total_batches, "dA_loss_fake": dA_loss_fake_epoch/total_batches, \
    #     "dB_loss_real": dB_loss_real_epoch/total_batches, "dB_loss_fake": dB_loss_fake_epoch/total_batches, \
    #     "g_A2B_total": g_A2B_total_epoch/total_batches, "g_A2B_gan": g_A2B_gan_epoch/total_batches, "g_A2B_id": g_A2B_id_epoch/total_batches, \
    #     "g_A2B_cycA": g_A2B_cycA_epoch/total_batches, "g_A2B_cycB": g_A2B_cycB_epoch/total_batches, \
    #     "g_B2A_total": g_B2A_total_epoch/total_batches, "g_B2A_gan": g_B2A_gan_epoch/total_batches, "g_B2A_id": g_B2A_id_epoch/total_batches, \
    #     "g_B2A_cycA": g_B2A_cycA_epoch/total_batches, "g_B2A_cycB": g_B2A_cycB_epoch/total_batches, "psnr_A": None, "ssim_A": None, "psnr_B": None, "ssim_B": None }
        
        # loss_row = {
        #     "iteration": i + 1,
        #     "dA_loss_real": dA_loss1, "dA_loss_fake": dA_loss2,
        #     "dB_loss_real": dB_loss1, "dB_loss_fake": dB_loss2,
        #     "g_A2B_total": g_loss1, "g_A2B_gan": g_adv1, "g_A2B_id": g_id1,
        #     "g_A2B_cycA": g_cyc1a, "g_A2B_cycB": g_cyc1b,
        #     "g_B2A_total": g_loss2, "g_B2A_gan": g_adv2, "g_B2A_id": g_id2,
        #     "g_B2A_cycB": g_cyc2a, "g_B2A_cycA": g_cyc2b,
        #     "psnr_A": None, "ssim_A": None, "psnr_B": None, "ssim_B": None
        # }

        # write_header = not os.path.exists(output_csv)
        # pd.DataFrame([loss_row]).to_csv(output_csv, mode='a', header=write_header, index=False)
        # print(f"Iter {i+1} | D_A[{dA_loss1:.3f},{dA_loss2:.3f}] D_B[{dB_loss1:.3f},{dB_loss2:.3f}]")

        # -------------------- Periodic Performance & Model Save --------------------
        if (epoch + 1) % 5 == 0:
            summarize_performance(epoch, g_model_AtoB, g_model_BtoA, gen_small, gen_large, 'AtoB', 5, output_dir)
            summarize_performance(epoch, g_model_BtoA, g_model_AtoB, gen_large, gen_small, 'BtoA', 5, output_dir)
            plot_training_metrics(output_dir)

        if (epoch + 1) % 100 == 0:
            save_models(epoch, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B, output_dir)

        # -------------------- Learning Rate Decay --------------------
        # if (epoch + 1) % 100 == 0:
        #     current_epoch = epoch + 1
        #     if current_epoch > n_iter:
        #         lr = max(initial_lr * (1 - (current_epoch - n_iter)/n_iter_decay), 0)
        #         for opt in [c_model_AtoB.optimizer, c_model_BtoA.optimizer, d_model_A.optimizer, d_model_B.optimizer]:
        #             opt.learning_rate.assign(lr)
        #         print(f">>> LR updated to {lr:.6f} at epoch {current_epoch}")