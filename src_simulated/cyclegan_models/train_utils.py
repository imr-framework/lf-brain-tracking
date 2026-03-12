import sys
sys.path.insert(0, './')

import os
import random
from random import random as rand_func
from numpy import asarray, load, zeros, ones
from numpy.random import randint
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, LeakyReLU,
    Activation, Concatenate, Add, UpSampling2D, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

# Image loading and preprocessing
from keras.preprocessing.image import img_to_array, load_img

# Nifti / medical image processing
import nibabel as nib
from scipy.ndimage import zoom

# Load config = CycleGANConfig() class from config.py
from src_simulated.cyclegan_models.config import config
# import losses from losses.py
from src_simulated.cyclegan_models.losses_cyclegan import *

# models.py load all models
from src_simulated.cyclegan_models.models import *

EPOCHS = config.EPOCHS
TEST = config.TEST

#parameters for descriminator
DISC_LOSS = config.DISC_LOSS
DISC_LEARNING_RATE = config.DISC_LEARNING_RATE
DISC_BETA_1 = config.DISC_BETA_1
DISC_LOSS_WEIGHTS = config.DISC_LOSS_WEIGHTS
#parameters for generator
GEN_LOSS_1 = config.GEN_LOSS_1
GEN_LOSS_2 = config.GEN_LOSS_2
GEN_LOSS_3 = config.GEN_LOSS_3
GEN_LOSS_4 = config.GEN_LOSS_4
GEN_LEARNING_RATE = config.GEN_LEARNING_RATE
GEN_BETA_1 = config.GEN_BETA_1
GEN_LOSS_WEIGHTS = config.GEN_LOSS_WEIGHTS

# load and prepare training images
def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
#Remember that for real images the label (y) is 1.
def generate_real_samples(dataset_a, dataset_b, n_samples, patch_shape):
	# choose random instances
	ix = randint(0, dataset_a.shape[0], n_samples)
	# retrieve selected images
	X = dataset_a[ix]
	Y = dataset_b[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, Y, y

# generate a batch of images, returns images and targets
#Remember that for fake images the label (y) is 0.
def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake images
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def save_models(step, g_model_AtoB, g_model_BtoA, d_ModelA, d_ModelB, output_path='src_simulated/outputs/cyclegan_t1_t2_upsample10'):
    """
    Saves GAN models at specific intervals and maintains a 'latest' copy.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    models = {
        "g_AtoB": g_model_AtoB,
        "g_BtoA": g_model_BtoA,
        "d_A": d_ModelA,
        "d_B": d_ModelB
    }
    
    for name, model in models.items():
        # Save periodic checkpoint
        ckpt_path = os.path.join(output_path, f"{name}_{step+1:06d}.keras")
        model.save(ckpt_path)

        # save latest as well
        latest_path = os.path.join(output_path, f"{name}_latest.keras")
        model.save(latest_path)
    
    print(f"> Saved checkpoints and latest models for step {step+1}")

# # periodically generate images using the save model and plot input and output images
# from matplotlib import pyplot, gridspec

# def summarize_performance(step, g_model, g_inverse, trainX, trainY, name, n_samples=5,
#                           output_path='src_simulated/outputs/cyclegan_t1_t2_upsample10'):
#     """
#     Visualize input, generated, and reconstructed images in 3 rows.

#     Parameters
#     ----------
#     step : int
#         Current training step for filename.
#     g_model : keras.Model
#         Generator mapping input -> target domain.
#     g_inverse : keras.Model
#         Generator mapping target -> input domain (for reconstruction).
#     trainX : np.array
#         Source images to sample from.
#     name : str
#         Name for saving the plot.
#     n_samples : int
#         Number of images to display.
#     output_path : str
#         Directory to save the figure.
#     """

#     if not os.path.exists(output_path):
#        os.makedirs(output_path)

#     # -----------------------------
#     # 1. Select samples and generate
#     # -----------------------------
#     X_in,Y_in, _ = generate_real_samples(trainX, trainY, n_samples, 0)           # Input
#     X_out, _ = generate_fake_samples(g_model, X_in, 0)              # Generated 
#     X_rec, _ = generate_fake_samples(g_inverse, X_out, 0)           # Reconstructed
    
#     # -----------------------------
#     # 2. Scale from [-1,1] -> [0,1]
#     # -----------------------------
#     X_in = (X_in + 1) / 2.0
#     Y_in = (Y_in + 1) / 2.0
#     X_out = (X_out + 1) / 2.0
#     X_rec = (X_rec + 1) / 2.0

#     # -----------------------------
#     # 3. Create figure
#     # -----------------------------
#     fig = pyplot.figure(figsize=(n_samples * 2, 6))  # 3 rows now
#     gs = gridspec.GridSpec(3, n_samples, wspace=-0.01, hspace=-0.01)

#     # Top row: Input
#     for i in range(n_samples):
#         ax = pyplot.subplot(gs[0, i])
#         ax.axis('off')
#         ax.imshow(X_in[i].squeeze(), cmap='gray', aspect='auto')

#     # Middle row: Generated
#     for i in range(n_samples):
#         ax = pyplot.subplot(gs[1, i])
#         ax.axis('off')
#         ax.imshow(X_out[i].squeeze(), cmap='gray', aspect='auto')

#     # Bottom row: Reconstructed
#     for i in range(n_samples):
#         ax = pyplot.subplot(gs[2, i])
#         ax.axis('off')
#         ax.imshow(X_rec[i].squeeze(), cmap='gray', aspect='auto')

#     # -----------------------------
#     # 4. Save figure
#     # -----------------------------
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     filename = os.path.join(output_path, f'{name}_generated_plot_{step+1:06d}.png')
#     pyplot.savefig(filename, bbox_inches='tight', pad_inches=0)
#     pyplot.close()
#     print(f'> Saved plot: {filename}')

import os
import numpy as np
from matplotlib import pyplot
from matplotlib import gridspec
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def summarize_performance(step, g_model, g_inverse,
                          trainX, trainY,
                          name, n_samples=5,
                          output_path='src_simulated/outputs/cyclegan_t1_t2_upsample10'):

    os.makedirs(output_path, exist_ok=True)

    # -----------------------------
    # 1. Sample data
    # -----------------------------
    X_in, Y_in, _ = generate_real_samples(trainX, trainY, n_samples, 0)
    X_out, _ = generate_fake_samples(g_model, X_in, 0)
    X_rec, _ = generate_fake_samples(g_inverse, X_out, 0)

    # -----------------------------
    # 2. Scale [-1,1] → [0,1]
    # -----------------------------
    X_in  = (X_in  + 1) / 2.0
    Y_in  = (Y_in  + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    X_rec = (X_rec + 1) / 2.0

    # -----------------------------
    # 3. Differences
    # -----------------------------
    diff_Y = np.abs(Y_in - X_out)
    diff_X = np.abs(X_in - X_rec)

    # -----------------------------
    # 4. Figure: NO spacing at all
    # -----------------------------
    fig = pyplot.figure(figsize=(n_samples * 2.0, 9))
    gs = gridspec.GridSpec(
    6, n_samples,
    wspace=-0.01,
    hspace=-0.01
    )

    pyplot.subplots_adjust(
        left=0, right=1,
        top=1, bottom=0,
        wspace=-0.01,
        hspace=-0.01
    )

    def plot_image(row, col, img, cmap='gray', vmin=None, vmax=None):
        ax = pyplot.subplot(gs[row, col])
        ax.axis('off')
        ax.imshow(img.squeeze(), cmap=cmap, vmin=vmin, vmax=vmax)
        return ax

    # -----------------------------
    # 5. Compute metrics
    # -----------------------------
    metrics_Y, metrics_X = [], []

    for i in range(n_samples):
        metrics_Y.append((
            psnr(Y_in[i].squeeze(), X_out[i].squeeze(), data_range=1.0),
            ssim(Y_in[i].squeeze(), X_out[i].squeeze(), data_range=1.0)
        ))
        metrics_X.append((
            psnr(X_in[i].squeeze(), X_rec[i].squeeze(), data_range=1.0),
            ssim(X_in[i].squeeze(), X_rec[i].squeeze(), data_range=1.0)
        ))

    # -----------------------------
    # 6. Plot rows
    # -----------------------------
    for i in range(n_samples):
        # X
        plot_image(0, i, X_in[i])

        # Y
        plot_image(1, i, Y_in[i])

        # G(X)
        plot_image(2, i, X_out[i])

        # |Y - G(X)|  (ERROR MAP)
        ax = plot_image(
            3, i, diff_Y[i],
            cmap='hot',
            vmin=0,
            vmax=diff_Y[i].max()
        )
        ax.text(
            0.02, 0.95,
            f"PSNR={metrics_Y[i][0]:.2f}\nSSIM={metrics_Y[i][1]:.3f}",
            color='blue',
            fontsize=7,
            transform=ax.transAxes,
            verticalalignment='top'
        )

        # F(G(X))
        plot_image(4, i, X_rec[i])

        # |X - F(G(X))|  (CYCLE ERROR)
        ax = plot_image(
            5, i, diff_X[i],
            cmap='hot',
            vmin=0,
            vmax=diff_X[i].max()
        )
        ax.text(
            0.02, 0.95,
            f"PSNR={metrics_X[i][0]:.2f}\nSSIM={metrics_X[i][1]:.3f}",
            color='blue',
            fontsize=7,
            transform=ax.transAxes,
            verticalalignment='top'
        )

    # -----------------------------
    # 7. Save (no padding)
    # -----------------------------
    filename = os.path.join(
        output_path,
        f"{name}_cyclegan_summary_{step+1:06d}.png"
    )
    pyplot.savefig(filename, bbox_inches='tight', pad_inches=0)
    pyplot.close()

    print(f"> Saved CycleGAN summary plot: {filename}")


def plot_training_metrics(folder_path, filename='training_metrics.csv', save_plot=True):
    """
    Plot training loss metrics from CSV and save figure.
    Also call summarize_performance for image visualization.
    Returns max iteration for continuing training.
    """
    csv_path = os.path.join(folder_path, filename)
    
    if not os.path.exists(csv_path):
        print(f"[Warning] Training CSV not found at: {csv_path}")
        return 0
    
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[Warning] Training CSV at {csv_path} is empty")
        return 0
    
    def smooth(values, window=50):
        return values.rolling(window=window, min_periods=1).mean()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    axes[0, 0].plot(df['iteration'], smooth(df['g_A2B_cycA']), label='A→B→A (Cyc A)', color='blue', alpha=0.8)
    axes[0, 0].plot(df['iteration'], smooth(df['g_B2A_cycB']), label='B→A→B (Cyc B)', color='cyan', alpha=0.8)
    axes[0, 0].set_title('Cycle Consistency Loss\n(Lower = Better Shape Preservation)')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    axes[0, 1].plot(df['iteration'], smooth(df['g_A2B_gan']), label='Gen A2B GAN', color='green')
    axes[0, 1].plot(df['iteration'], smooth(df['g_B2A_gan']), label='Gen B2A GAN', color='darkgreen')
    axes[0, 1].set_title('Generator GAN Loss\n(Stability = Healthy Competition)')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    avg_dA = (df['dA_loss_real'] + df['dA_loss_fake']) / 2
    avg_dB = (df['dB_loss_real'] + df['dB_loss_fake']) / 2
    axes[1, 0].plot(df['iteration'], smooth(avg_dA), label='Disc A Loss', color='red')
    axes[1, 0].plot(df['iteration'], smooth(avg_dB), label='Disc B Loss', color='orange')
    axes[1, 0].set_title('Discriminator (Adversarial) Loss\n(Lower = Better Discrimination)')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    axes[1, 1].plot(df['iteration'], smooth(df['psnr_A']), label='PSNR A', color='purple')
    axes[1, 1].plot(df['iteration'], smooth(df['psnr_B']), label='PSNR B', color='magenta')
    axes[1, 1].set_title('PSNR Score\n(Higher = Better Reconstruction)')
    axes[1, 1].set_ylabel('dB')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(f"CycleGAN Training Metrics Dashboard (Total Iterations: {df['iteration'].max()})", fontsize=16)
    
    if save_plot:
        max_iter = int(df['iteration'].max())
        plot_filename = f'cyclegan_performance_report_iter_{max_iter:06d}.png'
        plot_path = os.path.join(folder_path, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"[Info] Saved training metrics plot to {plot_path}")
    else:
        plt.show()
        plt.close(fig)

    # Return max iteration count for training continuation
    return int(df['iteration'].max())

# update image pool for fake images to reduce model oscillation
# update discriminators using a history of generated images
#rather than the ones produced by the latest generators.
#Original paper recommended keeping an image buffer that stores
#the 50 previously created images.

def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif rand_func() < 0.5:  # use the alias imported above
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool)-1)  # safe index
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)

def inspect_domains(A, B, n_samples=5):
	"""
	Prints stats and displays 20 samples (top=A, bottom=B).
	"""
	# ------------------------- Helper: describe -------------------------
	def describe(name, X):
		print(f"\n{name} Dataset:")
		print(f"  Shape         : {X.shape}")
		print(f"  Min pixel     : {X.min():.4f}")
		print(f"  Max pixel     : {X.max():.4f}")
		print(f"  Mean pixel    : {X.mean():.4f}")

	# ------------------------- Print stats -------------------------
	describe("A", A)
	describe("B", B)

	# ------------------------- Visualization -------------------------
	print("\nDisplaying sample images...")

	plt.figure(figsize=(20, 4))
	n = min(n_samples, len(A), len(B))

	for i in range(n):

		# ---------- TOP ROW (A) ----------
		plt.subplot(2, n, 1 + i)
		plt.axis("off")
		imgA = (A[i] + 1) / 2
		if imgA.shape[-1] == 1:
			imgA = imgA[:, :, 0]
			plt.imshow(imgA, cmap="gray")
		else:
			plt.imshow(imgA)

		# ---------- BOTTOM ROW (B) ----------
		plt.subplot(2, n, 1 + n + i)
		plt.axis("off")
		imgB = (B[i] + 1) / 2
		if imgB.shape[-1] == 1:
			imgB = imgB[:, :, 0]
			plt.imshow(imgB, cmap="gray")
		else:
			plt.imshow(imgB)

	plt.tight_layout()
	plt.show()

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA,
          c_model_AtoB, c_model_BtoA, dataset, output_dir='src_simulated/outputs/cyclegan_t1_t2_upsample10',
          epochs=300, initial_lr=0.0002, n_iter=40, n_iter_decay=150):

    # if output directory does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # define properties of the training run
    n_epochs, n_batch = epochs, 1  # batch size fixed to 1
    n_patch = d_model_A.output_shape[1]

    # unpack dataset
    trainA, trainB = dataset

    # prepare image pool for fake images
    poolA, poolB = list(), list()

    # calculate iterations
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs

    output_path = output_dir
    os.makedirs(output_path, exist_ok=True)

    output_csv = os.path.join(output_path, "training_metrics.csv")

    # Paths to "latest" models
    model_files = {
        'g_A2B': os.path.join(output_path, 'g_AtoB_latest.keras'),
        'g_B2A': os.path.join(output_path, 'g_BtoA_latest.keras'),
        'd_A': os.path.join(output_path, 'd_A_latest.keras'),
        'd_B': os.path.join(output_path, 'd_B_latest.keras')
    }

    # Initialize learning rate scheduler
    initial_lr = initial_lr
    n_iter = n_iter        # constant LR epochs
    n_iter_decay = n_iter_decay  # linear decay epochs

    # Resume from latest checkpoints if available
    if all(os.path.exists(f) for f in model_files.values()):
        print(">>> Resuming from latest checkpoints...")
        g_model_AtoB = load_model(model_files['g_A2B'], compile=False)
        g_model_BtoA = load_model(model_files['g_B2A'], compile=False)
        d_model_A = load_model(model_files['d_A'], compile=False)
        d_model_B = load_model(model_files['d_B'], compile=False)
        image_shape = dataset[0].shape[1:]
        # compile discriminators
        d_model_A.compile(
            loss=DISC_LOSS,
            optimizer=Adam(
                learning_rate=DISC_LEARNING_RATE,
                beta_1=DISC_BETA_1
            ),
            loss_weights=DISC_LOSS_WEIGHTS
        )
        d_model_B.compile(
            loss=DISC_LOSS,
            optimizer=Adam(
                learning_rate=DISC_LEARNING_RATE,
                beta_1=DISC_BETA_1
            ),
            loss_weights=DISC_LOSS_WEIGHTS
        )
        # recreate composite models
        c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
        c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
        if os.path.exists(output_csv):
            start_step = int(pd.read_csv(output_csv)['iteration'].max())
        else:
            start_step = 0
    else:
        print(">>> No checkpoints found. Starting training from scratch...")
        start_step = 0

    # -----------------------------
    # Start Training Loop
    # -----------------------------
    for i in range(start_step, n_steps):

        # -----------------------------
        # 1. REAL SAMPLES
        # -----------------------------
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)

        # -----------------------------
        # 2. FAKE SAMPLES
        # -----------------------------
        g_model_AtoB.trainable = False
        g_model_BtoA.trainable = False
        d_model_A.trainable = True
        d_model_B.trainable = True

        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)

        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)

        # -----------------------------
        # 3. TRAIN GENERATORS
        # -----------------------------
        g_model_AtoB.trainable = True
        g_model_BtoA.trainable = True
        d_model_A.trainable = False
        d_model_B.trainable = False

        g_loss1, g_adv1, g_id1, g_cyc1a, g_cyc1b = c_model_AtoB.train_on_batch(
            [X_realA, X_realB],
            [y_realB, X_realB, X_realA, X_realB]
        )

        g_loss2, g_adv2, g_id2, g_cyc2a, g_cyc2b = c_model_BtoA.train_on_batch(
            [X_realB, X_realA],
            [y_realA, X_realA, X_realB, X_realA]
        )

        # -----------------------------
        # 4. TRAIN DISCRIMINATORS
        # -----------------------------
        g_model_AtoB.trainable = False
        g_model_BtoA.trainable = False
        d_model_A.trainable = True
        d_model_B.trainable = True

        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        # -----------------------------
        # 5. LOG PROGRESS & CSV
        # -----------------------------
        if (i + 1) % (bat_per_epo * 1) == 0:
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

            if (i + 1) % (bat_per_epo * 5) == 0:
                fake_B = g_model_AtoB.predict(X_realA, verbose=0)
                rec_A = g_model_BtoA.predict(fake_B, verbose=0)
                psnr_A = tf.reduce_mean(compute_psnr(rec_A, X_realA)).numpy()
                ssim_A = tf.reduce_mean(compute_ssim(rec_A, X_realA)).numpy()
                fake_A = g_model_BtoA.predict(X_realB, verbose=0)
                rec_B = g_model_AtoB.predict(fake_A, verbose=0)
                psnr_B = tf.reduce_mean(compute_psnr(rec_B, X_realB)).numpy()
                ssim_B = tf.reduce_mean(compute_ssim(rec_B, X_realB)).numpy()
                loss_row.update({
                    "psnr_A": psnr_A, "ssim_A": ssim_A,
                    "psnr_B": psnr_B, "ssim_B": ssim_B
                })
                print(f" >> [Metrics Update] PSNR_A: {psnr_A:.2f} | SSIM_A: {ssim_A:.4f}")

            print(
                f"Iter {i+1} | D_A[{dA_loss1:.3f},{dA_loss2:.3f}] "
                f"D_B[{dB_loss1:.3f},{dB_loss2:.3f}] | "
                f"G_A2B[T:{g_loss1:.3f} GAN:{g_adv1:.3f}]"
            )

            write_header = not os.path.exists(output_csv)
            pd.DataFrame([loss_row]).to_csv(output_csv, mode='a', header=write_header, index=False)

        # -----------------------------
        # 6. PERIODIC PERFORMANCE CHECK
        # -----------------------------
        if (i + 1) % (bat_per_epo * 5) == 0:
            summarize_performance(i, g_model_AtoB, g_model_BtoA, trainA,'AtoB', 5, output_path)
            summarize_performance(i, g_model_BtoA, g_model_AtoB, trainB,'BtoA', 5, output_path)

            plot_training_metrics(output_path)

        # -----------------------------
        # 7. PERIODIC MODEL SAVE
        # -----------------------------
        if (i + 1) % (bat_per_epo * 20) == 0:
            save_models(i, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B,
                        output_path=output_path)

        # -----------------------------
        # 8. EPOCH-END SCHEDULER (Linear Decay)
        # -----------------------------
        if (i + 1) % (bat_per_epo * 1) == 0:  # once per epoch
            current_epoch = (i + 1) // bat_per_epo

            # Linear decay
            if current_epoch > n_iter:  # start decay after n_iter
                lr = initial_lr * (1 - (current_epoch - n_iter) / n_iter_decay)
                lr = max(lr, 0)  # just in case
                for opt in [c_model_AtoB.optimizer, c_model_BtoA.optimizer, d_model_A.optimizer, d_model_B.optimizer]:
                    print(type(c_model_AtoB.optimizer.learning_rate))
                    opt.learning_rate.assign(lr)
                print(f">>> Linear decay LR updated to {lr:.6f} at epoch {current_epoch}")