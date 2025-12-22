## Read dataset

data_folder = "Data/data_sim_check/35528simulated_LF/train_test"
subjects = ["26184", "30366", "35528","34507", "35547", "59228", "59877","59233"]
train_day = [1,2,3,4,5]

# monet2photo
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
# # Use the saved cyclegan models for image translation
from keras.models import load_model
from matplotlib import pyplot
from numpy.random import randint
import nibabel as nib

# inference test
import tensorflow as tf
from keras.layers import Layer
from keras.saving import register_keras_serializable

class InstanceNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.gamma = self.add_weight(
            shape=(dim,),
            initializer="ones",
            trainable=True,
            name="gamma"
        )
        self.beta = self.add_weight(
            shape=(dim,),
            initializer="zeros",
            trainable=True,
            name="beta"
        )
        super().build(input_shape)

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.math.sqrt(var + self.epsilon)
        return self.gamma * normalized + self.beta

def crop_or_pad_depth(vol, target_D):
    """
    Only crop/pad depth dimension.
    """
    h, w, d = vol.shape
    out = np.zeros((h, w, target_D), dtype=vol.dtype)

    ds = max((target_D - d) // 2, 0)
    de = ds + min(d, target_D)

    d0 = max((d - target_D) // 2, 0)

    out[:, :, ds:de] = vol[:, :, d0:d0 + (de - ds)]
    return out

def normalize_volume(vol):
    """
    Normalize an MRI volume to [-1, 1] for CycleGAN.
    Args:
        vol: numpy array, shape (H,W,D,C) or (H,W,D) or (N,H,W,D,C)
    Returns:
        normalized volume in [-1,1]
    """
    vol = np.abs(vol)           # remove negative artifacts
    max_val = np.max(vol)
    if max_val > 0:
        vol = vol / max_val     # scale to [0,1]
    vol = (vol - 0.5) * 2       # scale to [-1,1]
    return vol

def crop_or_pad_depth(vol, target_d=35):
    """Crop or pad depth to target size."""
    h, w, d = vol.shape
    if d > target_d:
        start = (d - target_d) // 2
        vol = vol[:, :, start:start+target_d]
    elif d < target_d:
        pad_before = (target_d - d) // 2
        pad_after = target_d - d - pad_before
        vol = np.pad(vol, ((0,0),(0,0),(pad_before, pad_after)), mode='constant')
    return vol

def load_nii_volumes(path, current_spacing=(1,1,2), target_spacing=(1,1,2), add_channel=False):

    """
    Load NIfTI volumes, resample by voxel spacing only,
    ensure H=W=128 or discard, crop/pad D→35, normalize to [-1,1].
    """

    TARGET_H = 128
    TARGET_W = 128
    TARGET_D = 35

    volumes = []

    for fname in os.listdir(path):
        if not fname.endswith((".nii", ".nii.gz")):
            continue

        fpath = os.path.join(path, fname)
        nii = nib.load(fpath)
        vol = nii.get_fdata().astype(np.float32)

        # 0. Fix LF-MRI negative values
        vol = np.abs(vol)

        # -------------------------------------------------------
        # 1. RESAMPLE ONLY USING SPACING (NO SHAPE-BASED RESIZE)
        # -------------------------------------------------------
        zoom_factors = (
            current_spacing[0] / target_spacing[0],
            current_spacing[1] / target_spacing[1],
            current_spacing[2] / target_spacing[2]
        )

        vol_iso = zoom(vol, zoom_factors, order=1)
        vol_iso = np.ascontiguousarray(vol_iso)

        h, w, d = vol_iso.shape
        print(f"[INFO] Volume {fname} resampled → {vol_iso.shape}")

        # -------------------------------------------------------
        # 2. ACCEPT ONLY 128×128 IN-PLANE RESOLUTION
        # -------------------------------------------------------
        if h != TARGET_H or w != TARGET_W:
            print(f"[SKIP] {fname} skipped — incorrect size: {vol_iso.shape}")
            continue

        # -------------------------------------------------------
        # 3. FIX DEPTH TO 35 WITHOUT DISTORTION
        # -------------------------------------------------------
        vol_fixed = crop_or_pad_depth(vol_iso, TARGET_D)

        # -------------------------------------------------------
        # 4. Normalize to [-1,1] (CycleGAN requirement)
        # -------------------------------------------------------
        vol_fixed = normalize_volume(vol_fixed)

        # -------------------------------------------------------
        # 5. Optional channel dimension
        # -------------------------------------------------------
        if add_channel:
            vol_fixed = vol_fixed[..., None]   # (H,W,D,1)

        print(f"[LOAD] {fname}: final shape {vol_fixed.shape}")
        volumes.append(vol_fixed)

    if not volumes:
        raise ValueError("No valid volumes loaded. Check dimensions and input path.")

    return np.stack(volumes, axis=0)

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

# write a function to visualize slices from the random volume
def visualize_slices(volume, n_cols=5):
    """
    Visualize slices from a 3D volume.
    """
    n_slices = volume.shape[2]
    n_rows = (n_slices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_slices):
        axes[i].imshow(volume[:, :, i], cmap="gray")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

# dataset path
path = 'Data/Nipah IRF data/Low_field_data_DA/'

# load dataset A - Monet paintings
dataA_all = load_nii_volumes(path + 'test_da_lf/')
print('Loaded dataA: ', dataA_all.shape)

from sklearn.utils import resample
#To get a subset of all images, for faster training during demonstration
dataA = resample(dataA_all,
                 replace=False,
                 n_samples=40,
                 random_state=42)

dataA = dataA[:, :, :, 3:-2]   # new depth = 30

# visualize_slices(dataA[1, :, :, :])
# visualize_slices(dataA[2, :, :, :])
# visualize_slices(dataA[3, :, :, :])
# visualize_slices(dataA[4, :, :, :])
# visualize_slices(dataA[5, :, :, :])
# visualize_slices(dataA[6, :, :, :])
# visualize_slices(dataA[7, :, :, :])
# visualize_slices(dataA[8, :, :, :])
# visualize_slices(dataA[9, :, :, :])
# visualize_slices(dataA[10, :, :, :])

# display range , min and max
print("DataA range: ", np.min(dataA), np.max(dataA))

# Load Data
X_train, y_train = load_data_for_days(subjects, train_day)

dataB = X_train
print('Loaded dataB: ', dataB.shape)
dataB = np.abs(dataB)
# Normalize dataB to [-1, 1]

# Normalize dataB to [-1, 1] volume-wise
for i in range(dataB.shape[0]):
	dataB[i] = normalize_volume(dataB[i])

# Crop dataB to match dataA height and width if needed
dataB = dataB[:, :dataA.shape[1], :dataA.shape[2], :]

dataB = dataB[:, :, :, :-5]

# visualize_slices(dataB[1, :, :, :])
# visualize_slices(dataB[2, :, :, :])
# visualize_slices(dataB[3, :, :, :])
# visualize_slices(dataB[4, :, :, :])
# visualize_slices(dataB[5, :, :, :])
# visualize_slices(dataB[6, :, :, :])
# visualize_slices(dataB[7, :, :, :])
# visualize_slices(dataB[8, :, :, :])
# visualize_slices(dataB[9, :, :, :])
# visualize_slices(dataB[10, :, :, :])

# display range , min and max
print("DataB range: ", np.min(dataB), np.max(dataB))

# discard first three slices and last 2 slices of dataA and last five slices of dataB to have depth 30

# load image data
data = [dataA, dataB]

print('Loaded', data[0].shape, data[1].shape)

# ----------------------------
# Convert Domain A → 2D (256x256)
# ----------------------------
A_slices = []
for i in range(dataA.shape[0]):          # number of volumes
    for z in range(dataA.shape[3]):      # number of slices
        slice_2d = dataA[i, :, :, z]
        slice_2d = cv2.resize(slice_2d, (128, 128), interpolation=cv2.INTER_LINEAR)
        A_slices.append(slice_2d)

A_2D = np.array(A_slices)
print("A_2D:", A_2D.shape)   # expected → (40*35, 256, 256)

# ----------------------------
# Convert Domain B → 2D (256x256)
# ----------------------------
B_slices = []
for i in range(dataB.shape[0]):
    for z in range(dataB.shape[3]):
        slice_2d = dataB[i, :, :, z]
        slice_2d = cv2.resize(slice_2d, (128, 128), interpolation=cv2.INTER_LINEAR)
        B_slices.append(slice_2d)

B_2D = np.array(B_slices)
print("B_2D:", B_2D.shape)   # expected → (40*35, 256, 256)

# Suppose A_2D.shape = (N, 256, 256)
A_2D = A_2D[..., np.newaxis]   # (N, 256, 256, 1)
# A_2D = np.repeat(A_2D, 3, axis=-1)  # (N, 256, 256, 3)

B_2D = B_2D[..., np.newaxis]
# B_2D = np.repeat(B_2D, 3, axis=-1)  # (N, 256, 256, 3)
print(A_2D.shape, B_2D.shape)

data = [B_2D, A_2D]

#print datatype of each
print("A_2D dtype:", A_2D.dtype)
print("B_2D dtype:", B_2D.dtype)

def inspect_domains(A, B, n_samples=20):
	"""
	Prints stats and displays 20 samples (top=A, bottom=B).
	"""

	# ------------------------- Helper: describe --------------------
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

dataset = data

inspect_domains(dataset[0], dataset[1], n_samples=20)

# from cycleGAN_model import define_generator, define_discriminator, define_composite_model, train
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
print(image_shape)

# select a random sample of images from the dataset
# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    print("Selected indices:", ix)

    # retrieve selected images
    X = dataset[ix]
    return X

# plot the image, its translation, and the reconstruction
from numpy import vstack
from matplotlib import pyplot

def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']

    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0

    for i in range(len(images)):
        pyplot.subplot(1, len(images), 1 + i)
        pyplot.axis('off')

        img = images[i]

        # ✅ handle (H, W, 1) or (1, H, W)
        if img.ndim == 3:
            img = img.squeeze()

        # ✅ force grayscale
        pyplot.imshow(img, cmap='gray', vmin=0, vmax=1)
        pyplot.title(titles[i])

    pyplot.show()

def show_plot_domains(A_real, A_gen, A_rec, B_real, B_gen, B_rec, save_path=None):
    
    """
    Display both A and B domains in a single figure:
    - 2 rows: A domain (top), B domain (bottom)
    - 3 columns: Real / Generated / Reconstructed
    - Column titles on top
    - Row labels on left (black color)
    - Zero spacing
    - Handles 3D/4D/5D inputs
    """

    def get_middle_slice(vol):
        # Convert to 2D slice (H,W) and scale [0,1]
        if vol.ndim == 5:       # (1,H,W,D,C)
            vol = vol[0, ..., vol.shape[3]//2, 0]
        elif vol.ndim == 4:     # (H,W,D,C)
            vol = vol[..., vol.shape[2]//2, 0]
        elif vol.ndim == 3:     # (H,W,D) or (H,W,1)
            vol = vol[..., vol.shape[2]//2] if vol.shape[2] > 1 else vol[..., 0]
        vol = (vol + 1) / 2.0   # scale [-1,1] -> [0,1]
        return vol

    # Extract slices
    A_real_slice = get_middle_slice(A_real[0])
    A_gen_slice  = get_middle_slice(A_gen[0])
    A_rec_slice  = get_middle_slice(A_rec[0])

    B_real_slice = get_middle_slice(B_real[0])
    B_gen_slice  = get_middle_slice(B_gen[0])
    B_rec_slice  = get_middle_slice(B_rec[0])

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    col_titles = ['Real', 'Generated', 'Reconstructed']
    row_labels = ['A Domain', 'B Domain']

    for row_idx, row_imgs in enumerate([[A_real_slice, A_gen_slice, A_rec_slice],
                                        [B_real_slice, B_gen_slice, B_rec_slice]]):
        for col_idx, img in enumerate(row_imgs):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            # Column titles on top row
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=12, color='black')

            # Row labels on first column
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], rotation=90, size='large',
                              labelpad=10, color='black')

    plt.subplots_adjust(wspace=0, hspace=0)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
        print(f"[Saved] {save_path}")

    plt.show()

# load dataset
A_data = resample(dataset[0],
                 replace=False,
                 n_samples=50,
                 random_state=42) # reproducible results

B_data = resample(dataset[1],
                 replace=False,
                 n_samples=50,
                 random_state=42) # reproducible results

# select a sample of images
A_real = select_sample(A_data, 1)
B_real = select_sample(B_data, 1)

#print shapes
print("A_real shape:", A_real.shape)
print("B_real shape:", B_real.shape)

inspect_domains(A_real, B_real, n_samples=1)

# Normalize in range [-1, 1] if not already
# A_real = normalize_volume(A_real)
# B_real = normalize_volume(B_real)

print("After normalization:") 
#min and max
print("A_real min:", np.min(A_real), "max:", np.max(A_real))
print("B_real min:", np.min(B_real), "max:", np.max(B_real))

# load the models
g_model_AtoB = load_model('src_simulated/outputs/cyclegan2BA/g_model_AtoB_latest.keras', custom_objects={'InstanceNormalization': InstanceNormalization})
g_model_BtoA = load_model('src_simulated/outputs/cyclegan2BA/g_model_BtoA_latest.keras', custom_objects={'InstanceNormalization': InstanceNormalization})

# generate images
A_generated = g_model_BtoA.predict(B_real)
B_generated = g_model_AtoB.predict(A_real)

# print shapes
print("A_generated shape:", A_generated.shape)
print("B_generated shape:", B_generated.shape)

# reconstruct images
A_reconstructed = g_model_AtoB.predict(A_generated)
B_reconstructed = g_model_BtoA.predict(B_generated)

# print shapes
print("A_reconstructed shape:", A_reconstructed.shape)
print("B_reconstructed shape:", B_reconstructed.shape)

# plot all results
print("A domain:")
show_plot(A_real, A_generated, A_reconstructed)
print("B domain:")
show_plot(B_real, B_generated, B_reconstructed)

show_plot_domains(A_real, A_generated, A_reconstructed,
                  B_real, B_generated, B_reconstructed,
                  save_path=None)

# Function to convert 3-channel image to grayscale
def to_gray(x):
    # x shape: (H, W, 3) or (N, H, W, 3)
    return np.mean(x, axis=-1, keepdims=True)

# Convert generated and reconstructed images to grayscale
A_generated_gray = to_gray(A_generated)
B_generated_gray = to_gray(B_generated)
A_reconstructed_gray = to_gray(A_reconstructed)
B_reconstructed_gray = to_gray(B_reconstructed)

# Function to display images in grayscale
def show_plot_gray(real, generated, reconstructed):
    plt.figure(figsize=(12,4))
    images = [real, generated, reconstructed]
    titles = ['Real', 'Generated', 'Reconstructed']
    for i, img in enumerate(images):
        plt.subplot(1,3,i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Plot all results in grayscale
print("A domain:")
show_plot_gray(A_real, A_generated_gray, A_reconstructed_gray)
print("B domain:")
show_plot_gray(B_real, B_generated_gray, B_reconstructed_gray)

# Load real images and produce outputs

# # ##########################
# # Load a single custom image
# test_image = load_img('monet2photo/sunset256.jpg')
# test_image = img_to_array(test_image)
# test_image_input = np.array([test_image])  # Convert single image to a batch.
# test_image_input = (test_image_input - 127.5) / 127.5
# print("Test image shape:", test_image_input.shape)
# # plot B->A->B (Photo to Monet to Photo)
# monet_generated  = g_model_BtoA.predict(test_image_input)
# photo_reconstructed = g_model_AtoB.predict(monet_generated)
# show_plot(test_image_input, monet_generated, photo_reconstructed)