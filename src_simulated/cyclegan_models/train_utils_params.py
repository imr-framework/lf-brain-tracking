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

def generate_real_samples(dataset, context, n_samples, patch_shape):
    """
    Sample real images and context, return with label 1.
    """
    idx = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[idx]
    c = context[idx]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return X, c, y

def generate_fake_samples(g_model, dataset, context, patch_shape):
    """
    Generate fake images using generator with context.
    """
    X = g_model.predict([dataset, context], verbose=0)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

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
def summarize_performance(step, g_model, g_inverse, dataset, name, n_samples=5, output_path=None):
    """
    Visualize input, generated, reconstructed, and error images with metrics.
    """
    os.makedirs(output_path, exist_ok=True)

    X_in, Y_in, contextA, contextB, _ = generate_real_samples_a(dataset[0], dataset[1], dataset[2], dataset[3], n_samples, 0)
    X_out, _ = generate_fake_samples_a(g_model, X_in, contextA, 0)
    X_rec, _ = generate_fake_samples_a(g_inverse, X_out, contextB, 0)

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

# -----------------------------
# Main Training Loop
# -----------------------------
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA,
          c_model_AtoB, c_model_BtoA, dataset, output_dir,
          epochs=300, initial_lr=0.0002, n_iter=40, n_iter_decay=150):
    
    os.makedirs(output_dir, exist_ok=True)
    trainA, trainB, contextA, contextB = dataset
    poolA, poolB = [], []

    n_batch = 1
    n_patch = d_model_A.output_shape[1]
    bat_per_epo = int(len(trainA)/n_batch)
    n_steps = bat_per_epo * epochs
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
        c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, trainA.shape[1:])
        c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, trainB.shape[1:])
        if os.path.exists(output_csv):
            start_step = int(pd.read_csv(output_csv)['iteration'].max())

    # -----------------------------
    # Training iterations
    # -----------------------------
    for i in range(start_step, n_steps):
        # -------------------- Real & Fake samples --------------------
        X_realA, cA, y_realA = generate_real_samples(trainA, contextA, n_batch, n_patch)
        X_realB, cB, y_realB = generate_real_samples(trainB, contextB, n_batch, n_patch)

        g_model_AtoB.trainable = False
        g_model_BtoA.trainable = False
        d_model_A.trainable = True
        d_model_B.trainable = True

        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, cB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, cA, n_patch)

        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)

        # -------------------- Train Generators --------------------
        g_model_AtoB.trainable = True
        g_model_BtoA.trainable = True
        d_model_A.trainable = False
        d_model_B.trainable = False

        g_loss1, g_adv1, g_id1, g_cyc1a, g_cyc1b = c_model_AtoB.train_on_batch( [X_realA, cA, X_realA], [y_realB, X_realB, X_realA, X_realB])
        g_loss2, g_adv2, g_id2, g_cyc2a, g_cyc2b = c_model_BtoA.train_on_batch([X_realB, cB, X_realB], [y_realA, X_realA, X_realB, X_realA])

        # -------------------- Train Discriminators --------------------
        g_model_AtoB.trainable = False
        g_model_BtoA.trainable = False
        d_model_A.trainable = True
        d_model_B.trainable = True

        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        # -------------------- Logging & CSV --------------------
        if (i + 1) % bat_per_epo == 0:
            loss_row = {
                "iteration": i + 1,
                "dA_loss_real": dA_loss1, "dA_loss_fake": dA_loss2,
                "dB_loss_real": dB_loss1, "dB_loss_fake": dB_loss2,
                "g_A2B_total": g_loss1, "g_A2B_gan": g_adv1, "g_A2B_id": g_id1,
                "g_A2B_cycA": g_cyc1a, "g_A2B_cycB": g_cyc1b,
                "g_B2A_total": g_loss2, "g_B2A_gan": g_adv2, "g_B2A_id": g_id2,
                "g_B2A_cycB": g_cyc2a, "g_B2A_cycA": g_cyc2b,
                "psnr_A": None, "ssim_A": None, "psnr_B": None, "ssim_B": None
            }

            write_header = not os.path.exists(output_csv)
            pd.DataFrame([loss_row]).to_csv(output_csv, mode='a', header=write_header, index=False)
            print(f"Iter {i+1} | D_A[{dA_loss1:.3f},{dA_loss2:.3f}] D_B[{dB_loss1:.3f},{dB_loss2:.3f}]")

        # -------------------- Periodic Performance & Model Save --------------------
        if (i + 1) % (bat_per_epo * 5) == 0:
            summarize_performance(i, g_model_AtoB, g_model_BtoA, [trainA, trainB, contextA, contextB], 'AtoB', 5, output_dir)
            summarize_performance(i, g_model_BtoA, g_model_AtoB, [trainB, trainA, contextB, contextA], 'BtoA', 5, output_dir)
            plot_training_metrics(output_dir)

        if (i + 1) % (bat_per_epo * 20) == 0:
            save_models(i, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B, output_dir)

        # -------------------- Learning Rate Decay --------------------
        if (i + 1) % bat_per_epo == 0:
            current_epoch = (i+1)//bat_per_epo
            if current_epoch > n_iter:
                lr = max(initial_lr * (1 - (current_epoch - n_iter)/n_iter_decay), 0)
                for opt in [c_model_AtoB.optimizer, c_model_BtoA.optimizer, d_model_A.optimizer, d_model_B.optimizer]:
                    opt.learning_rate.assign(lr)
                print(f">>> LR updated to {lr:.6f} at epoch {current_epoch}")