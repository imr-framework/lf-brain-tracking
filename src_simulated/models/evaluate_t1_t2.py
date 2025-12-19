# Read dataset

data_folder = "Data/Nipah IRF data/IRF_3T_NIFTI"
subjects = ["26184", "30366","34507", "35547", "59877","59233"]
train_day = [1,2,3,4,5]

# monet2photo
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model

import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

import tensorflow as tf
from tensorflow.keras.layers import Layer

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
        # Compute mean/variance over spatial dimensions (H,W,D)
        axes = [i for i in range(1, len(inputs.shape)) if i != self.axis]
        mean, var = tf.nn.moments(inputs, axes=axes, keepdims=True)
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

import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def load_nii_volumes(path, target_spacing=(1,1,2), add_channel=False, 
                     target_h=140, target_w=140, target_d=35, 
                     substring=None, rotate=False, test=False):
    """
    Load NIfTI volumes, optionally filter by substring in filename,
    resample to target voxel spacing, accept only target in-plane resolution,
    fix depth, normalize, optionally add channel dimension.
    
    Parameters:
    - test : bool : if True, load only 5 volumes to save time
    """
    volumes = []

    # Recursively find NIfTI files
    nii_files = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        filenames.sort()
        for fname in filenames:
            if fname.endswith((".nii", ".nii.gz")) and (substring is None or substring in fname):
                nii_files.append(os.path.join(dirpath, fname))

    nii_files.sort(key=lambda x: (os.path.dirname(x), os.path.basename(x)))

    # Limit number of files if test mode is on
    if test:
        nii_files = nii_files[:5]

    for fpath in nii_files:
        nii = nib.load(fpath)
        vol = np.abs(nii.get_fdata().astype(np.float32))  # fix negatives
        header = nii.header
        current_spacing = header.get_zooms()[:3]  # get voxel size

        # Resample to target voxel size
        zoom_factors = (
            current_spacing[0] / target_spacing[0],
            current_spacing[1] / target_spacing[1],
            current_spacing[2] / target_spacing[2]
        )
        vol_iso = zoom(vol, zoom_factors, order=1)
        vol_iso = np.ascontiguousarray(vol_iso)
        h, w, d = vol_iso.shape
        print(f"[INFO] {os.path.basename(fpath)} resampled → {vol_iso.shape} (target spacing {target_spacing})")

        # Check in-plane resolution
        if h != target_h or w != target_w:
            print(f"[SKIP] {os.path.basename(fpath)} skipped — incorrect in-plane size: {vol_iso.shape}")
            continue

        # Crop/pad depth
        vol_fixed = crop_or_pad_depth(vol_iso, target_d)

        # Normalize to [-1,1]
        vol_fixed = normalize_volume(vol_fixed)

        # Rotate 90° counter-clockwise in-plane (H x W)
        if rotate:
            vol_fixed = np.rot90(vol_fixed, k=3, axes=(0, 1))

        # Optional channel dimension
        if add_channel:
            vol_fixed = vol_fixed[..., None]  # (H,W,D,1)

        print(f"[LOAD] {os.path.basename(fpath)}: final shape {vol_fixed.shape}")
        volumes.append(vol_fixed)

        # Stop early if test mode
        if test and len(volumes) >= 5:
            break

    if not volumes:
        raise ValueError("No valid volumes loaded. Check dimensions, substring, and input path.")
    
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

import numpy as np
from scipy.ndimage import zoom

def resample_volume(volume, current_spacing=(1,1,2), target_spacing=(1,1,1), order=3):
    """
    Resample a 3D volume to new voxel spacing.

    Parameters:
    volume : np.ndarray
        3D volume (D, H, W)
    current_spacing : tuple
        Original voxel spacing (z, y, x)
    target_spacing : tuple
        Desired voxel spacing (z, y, x)
    order : int
        Interpolation order:
        0 = nearest (labels)
        1 = linear
        3 = cubic (recommended for MRI)

    Returns:
    resampled_volume : np.ndarray
    """

    zoom_factors = (
        current_spacing[0] / target_spacing[0],
        current_spacing[1] / target_spacing[1],
        current_spacing[2] / target_spacing[2],
    )

    resampled_volume = zoom(volume, zoom_factors, order=order)
    return resampled_volume

# dataset path

# Example usage
data_path = "Data/Nipah IRF data/IRF_3T_NIFTI"
substring_filter = "T1_n100__00001"

dataA_all = load_nii_volumes(
    path=data_path,
    target_spacing=(1,1,2),
    add_channel=False,
    target_h=140,
    target_w=140,
    target_d=35,
    substring=substring_filter,
    rotate=True,
    test=True
)

print("Loaded volumes shape:", dataA_all.shape)

from sklearn.utils import resample

# take first 40 samples from dataA_all
dataA = dataA_all[:40]
print('Loaded dataA: ', dataA.shape)

# convert to grayscale
# dataA = np.array([cv2.cvtColor(dataA[i], cv2.COLOR_RGB2GRAY) for i in range(len(dataA))])
visualize_slices(dataA[1, :, :, :])
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

# Example usage
substring_filter = "T2_n100"

dataB_all = load_nii_volumes(
    path=data_path,
    target_spacing=(1,1,2),
    add_channel=False,
    target_h=140,
    target_w=140,
    target_d=35,
    substring=substring_filter,
    rotate=True,
    test=True
)

print("Loaded volumes shape:", dataB_all.shape)

from sklearn.utils import resample

# take first 40 samples from dataB_all
dataB = dataB_all[:40]
print('Loaded dataB: ', dataB.shape)

# Load Data

# resampled_dataB = []

# for i in range(dataB.shape[0]):
#     vol = dataB[i]  # shape: (140, 140, 35)
#     vol_resampled = resample_volume(
#         vol,
#         current_spacing=(1,1,2),
#         target_spacing=(1,1,1),
#         order=3
#     )  # shape: (140, 140, 70)
#     resampled_dataB.append(vol_resampled)

# # resampled_dataB is a list of 40 arrays, each 140x140x70
# dataB = np.array(resampled_dataB)

# print('Resampled dataB: ', dataB.shape)
# Normalize dataB to [-1, 1] volume-wise
# for i in range(dataB.shape[0]):
# 	dataB[i] = normalize_volume(dataB[i])

# # Crop dataB to match dataA height and width if needed
# dataB = dataB[:, :dataA.shape[1], :dataA.shape[2], :]

# dataB = dataB[:, :, :, :-10]

visualize_slices(dataB[1, :, :, :])
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

# Pad input depth to nearest multiple of 2^n_downsampling (here n_downsampling=2)
def pad_to_shape(vol, target_shape=(144, 144, 40)):
    """
    Pad or crop a 3D volume to the target shape (h, w, d).
    Pads with zeros if needed, crops centrally if too large.
    """
    h, w, d = vol.shape[:3]
    th, tw, td = target_shape

    # Calculate padding or cropping for each dimension
    pad_h = max(th - h, 0)
    pad_w = max(tw - w, 0)
    pad_d = max(td - d, 0)

    # Pad as needed
    pad_width = (
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
        (pad_d // 2, pad_d - pad_d // 2)
    )
    vol_padded = np.pad(vol, pad_width, mode='constant')

    # Crop centrally if needed
    h2, w2, d2 = vol_padded.shape[:3]
    start_h = (h2 - th) // 2
    start_w = (w2 - tw) // 2
    start_d = (d2 - td) // 2
    vol_cropped = vol_padded[start_h:start_h+th, start_w:start_w+tw, start_d:start_d+td]

    return vol_cropped


# Apply padding
dataA = np.array([pad_to_shape(vol) for vol in dataA])
dataB = np.array([pad_to_shape(vol) for vol in dataB])

print('Padded dataA: ', dataA.shape)
print('Padded dataB: ', dataB.shape)

# Suppose A_2D.shape = (N, 256, 256)
A_2D = dataA[..., np.newaxis]   # (N, 256, 256, 1)
# A_2D = np.repeat(A_2D, 3, axis=-1)  # (N, 256, 256, 3)

B_2D = dataB[..., np.newaxis]
# B_2D = np.repeat(B_2D, 3, axis=-1)  # (N, 256, 256, 3)
print(A_2D.shape, B_2D.shape)

data = [A_2D, B_2D]

#print datatype of each
print("A_2D dtype:", A_2D.dtype)
print("B_2D dtype:", B_2D.dtype)

import numpy as np
import matplotlib.pyplot as plt

def inspect_domains(A, B, n_samples=10):
    """
    Prints stats and displays n_samples subjects.
    For multi-slice data, the middle slice is shown.
    Top row = A, Bottom row = B
    """

    # ------------------------- Helper: describe -------------------------
    def describe(name, X):
        print(f"\n{name} Dataset:")
        print(f"  Shape         : {X.shape}")
        print(f"  Min pixel     : {X.min():.4f}")
        print(f"  Max pixel     : {X.max():.4f}")
        print(f"  Mean pixel    : {X.mean():.4f}")

    describe("A", A)
    describe("B", B)

    # ------------------------- Slice extractor -------------------------
    def get_middle_slice(x):
        """
        x: single sample
        Supports:
        - (H, W)
        - (H, W, C)
        - (H, W, D, C)
        - (D, H, W, C)
        """

        if x.ndim == 2:        # (H, W)
            return x

        if x.ndim == 3:        # (H, W, C)
            return x[..., 0] if x.shape[-1] == 1 else x

        if x.ndim == 4:
            # Decide where depth dimension is
            if x.shape[-1] in [1, 3]:     # (H, W, D, C)
                mid = x.shape[2] // 2
                return x[:, :, mid, 0]
            else:                          # (D, H, W, C)
                mid = x.shape[0] // 2
                return x[mid, :, :, 0]

        raise ValueError(f"Unsupported shape: {x.shape}")

    # ------------------------- Visualization -------------------------
    n = min(n_samples, len(A), len(B))
    print(f"\nDisplaying middle slice of {n} samples...")

    plt.figure(figsize=(2 * n, 4))

    for i in range(n):
        # ---------- A ----------
        plt.subplot(2, n, i + 1)
        plt.axis("off")
        imgA = (A[i] + 1) / 2
        imgA = get_middle_slice(imgA)
        plt.imshow(imgA, cmap="gray")

        # ---------- B ----------
        plt.subplot(2, n, n + i + 1)
        plt.axis("off")
        imgB = (B[i] + 1) / 2
        imgB = get_middle_slice(imgB)
        plt.imshow(imgB, cmap="gray")

    plt.tight_layout()
    plt.show()

dataset = data

inspect_domains(dataset[0], dataset[1], n_samples=20)

# from cycleGAN_model import define_generator, define_discriminator, define_composite_model, train
# define input shape based on the loaded dataset

# Suppose you define a patch size
patch_size = (64, 64, 32)  # (H, W, D)

# Use patch size instead of full image shape
image_shape = patch_size + (1,)  # add channel dimension
print(image_shape)  # -> (64, 64, 32, 1)


#Inference code

# ===================== PATCH-WISE 3D CycleGan INFERENCE =====================

# --------------------- PATCH EXTRACTION ---------------------
def extract_patches_3d(volume, patch_size=(64,64,32), stride=(32,32,16)):
    H, W, D = volume.shape
    ph, pw, pd = patch_size
    sh, sw, sd = stride

    patches = []
    coords = []

    for x in range(0, H - ph + 1, sh):
        for y in range(0, W - pw + 1, sw):
            for z in range(0, D - pd + 1, sd):
                patches.append(volume[x:x+ph, y:y+pw, z:z+pd])
                coords.append((x, y, z))

    patches = np.array(patches)[..., np.newaxis]  # (N,ph,pw,pd,1)
    return patches, coords

# --------------------- PATCH RECONSTRUCTION ---------------------
def reconstruct_volume_3d(patches, coords, volume_shape):
    recon = np.zeros(volume_shape, dtype=np.float32)
    weight = np.zeros(volume_shape, dtype=np.float32)

    for patch, (x,y,z) in zip(patches, coords):
        patch = patch[..., 0]
        ph, pw, pd = patch.shape
        recon[x:x+ph, y:y+pw, z:z+pd] += patch
        weight[x:x+ph, y:y+pw, z:z+pd] += 1.0

    return recon / np.maximum(weight, 1e-6)

# --------------------- FULL VOLUME GENERATION ---------------------
def generate_full_volume(
    g_model,
    volume,
    patch_size=(64,64,32),
    stride=(32,32,16),
    batch_size=1
):
    patches, coords = extract_patches_3d(volume, patch_size, stride)
    gen_patches = g_model.predict(patches, batch_size=batch_size, verbose=0)
    return reconstruct_volume_3d(gen_patches, coords, volume.shape)

# --------------------- VISUALIZATION ---------------------
def show_volume_triplet(real, generated, reconstructed):
    mid = real.shape[2] // 2
    titles = ["Real", "Generated", "Reconstructed"]
    images = [real[:,:,mid], generated[:,:,mid], reconstructed[:,:,mid]]

    plt.figure(figsize=(12,4))
    for i, img in enumerate(images):
        plt.subplot(1,3,i+1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()

# ===================== USAGE EXAMPLE =====================
# Input shape expected: (1, H, W, D, 1)

# load the models
g_model_AtoB = load_model('src_simulated/outputs/cyclegan3d_t1_t2/g_model_AtoB_latest.keras', custom_objects={'InstanceNormalization': InstanceNormalization})
g_model_BtoA = load_model('src_simulated/outputs/cyclegan3d_t1_t2/g_model_BtoA_latest.keras', custom_objects={'InstanceNormalization': InstanceNormalization})

# Remove batch + channel
dataset = data

A_vol = dataset[0][0, ..., 0]
B_vol = dataset[1][0, ..., 0]
print("A_vol shape:", A_vol.shape)
print("B_vol shape:", B_vol.shape)

# Generate translations
A_to_B = generate_full_volume(g_model_AtoB, A_vol)
B_to_A = generate_full_volume(g_model_BtoA, B_vol)

# Cycle reconstruction
A_cycle = generate_full_volume(g_model_BtoA, A_to_B)
B_cycle = generate_full_volume(g_model_AtoB, B_to_A)

# Visualize
show_volume_triplet(A_vol, A_to_B, A_cycle)
show_volume_triplet(B_vol, B_to_A, B_cycle)


