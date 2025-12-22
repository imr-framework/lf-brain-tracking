"""
3D CycleGAN conversion of your 2D implementation.

Assumptions:
- image_shape = (D, H, W, C)  # channels_last
- keras_contrib InstanceNormalization available and works with 5D tensors.
- Batch size remains 1 as in original CycleGAN paper/code.

"""

from random import random
from numpy import load, zeros, ones, asarray
from numpy.random import randint

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, Conv3DTranspose, LeakyReLU,
    Activation, Concatenate, Add
)

import nibabel as nib

# If keras_contrib isn't available, try:
# from tfa.layers import InstanceNormalization
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

from matplotlib import pyplot as plt

# ----------------------------
# Discriminator (PatchGAN) 3D
# ----------------------------
def define_discriminator_3d(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # input
    in_image = Input(shape=image_shape)
    # C64: 4x4x4 kernel stride 2
    d = Conv3D(64, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(negative_slope=0.2)(d)
    # C128
    d = Conv3D(128, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(negative_slope=0.2)(d)
    # C256
    d = Conv3D(256, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(negative_slope=0.2)(d)
    # C512 (optional)
    d = Conv3D(512, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(negative_slope=0.2)(d)
    # second last (stride 1)
    d = Conv3D(512, (4,4,4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(negative_slope=0.2)(d)
    # patch output (single channel)
    patch_out = Conv3D(1, (4,4,4), padding='same', kernel_initializer=init)(d)
    model = Model(in_image, patch_out)
    # compile
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
    model.summary()
    return model

# ----------------------------
# ResNet block for 3D generator
# ----------------------------
def resnet_block_3d(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)
    g = Conv3D(n_filters, (3,3,3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv3D(n_filters, (3,3,3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Add()([g, input_layer])  # residual-ish (concatenate as original code did)
    return g

# ----------------------------
# Generator 3D (Encoder-ResNet-Decoder)
# ----------------------------
def define_generator_3d(image_shape, n_resnet=6):
    # note: original 2D used 9 or 6 blocks - for 3D keep n_resnet relatively small for memory
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    # c7s1-64 -> use 7x7x7 conv
    g = Conv3D(64, (7,7,7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d128
    g = Conv3D(128, (3,3,3), strides=(2,2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256
    g = Conv3D(256, (3,3,3), strides=(2,2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # R256 repeated
    for _ in range(n_resnet):
        g = resnet_block_3d(256, g)
    # u128 (transpose conv)
    g = Conv3DTranspose(128, (3,3,3), strides=(2,2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # u64
    g = Conv3DTranspose(64, (3,3,3), strides=(2,2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # final c7s1 - number of channels preserved from input (C)
    n_channels = image_shape[-1]
    g = Conv3D(n_channels, (7,7,7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
    model = Model(in_image, out_image)
    return model

# ----------------------------
# Composite model for training generators
# ----------------------------
def define_composite_model_3d(g_model_1, d_model, g_model_2, image_shape):
    # make g_model_1 trainable, others frozen
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False

    # inputs
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)

    # identity
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)

    # cycle forward and backward
    output_f = g_model_2(gen1_out)
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'mae', 'mae', 'mae'],
                  loss_weights=[1, 5, 10, 10],
                  optimizer=opt)
    return model

# ----------------------------
# Data helpers (3D)
# ----------------------------

from numpy.random import randint
from numpy import ones, zeros, asarray
import matplotlib.pyplot as plt
import os
from random import random

# -----------------------------
# Load real samples (normalized to [-1,1])
# -----------------------------
def load_real_samples(filename):
    # load the dataset (expects .npz with arr_0, arr_1)
    data = np.load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    # ensure 5D: (B,H,W,D,1)
    if X1.ndim == 4: X1 = X1[..., None]
    if X2.ndim == 4: X2 = X2[..., None]
    return [X1, X2]


# -----------------------------
# Generate real 3D samples with matching PatchGAN labels
# -----------------------------
def generate_real_samples(dataset, n_samples, d_model):
    """
    Selects n_samples from dataset and generates real labels
    matching the 3D PatchGAN output shape of d_model.
    """
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]

    # get discriminator output shape (excluding batch and channels)
    y_shape = d_model.output_shape[1:-1]  # (H_out, W_out, D_out)
    y = ones((n_samples, *y_shape, 1), dtype='float32')
    return X, y


# -----------------------------
# Generate fake 3D samples with matching PatchGAN labels
# -----------------------------
def generate_fake_samples(g_model, dataset, d_model):
    """
    Generates fake images using g_model and matching 3D PatchGAN labels.
    """
    X = g_model.predict(dataset)
    y_shape = d_model.output_shape[1:-1]
    y = zeros((len(X), *y_shape, 1), dtype='float32')
    return X, y


# -----------------------------
# Update image pool for fake 3D images
# -----------------------------
def update_image_pool(pool, images, max_size=50):
    """
    Maintains a pool of previously generated fake images
    to stabilize discriminator training.
    """
    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            selected.append(image)
        else:
            ix = randint(0, len(pool) - 1)
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)

# -----------------------------
# Save models
# -----------------------------
def save_models(step, g_model_AtoB, g_model_BtoA, output_path='outputs/cyclegan3d_t1_t2'):
    
    os.makedirs(output_path, exist_ok=True)
    latest_AtoB = os.path.join(output_path, "g_model_AtoB_latest.keras")
    latest_BtoA = os.path.join(output_path, "g_model_BtoA_latest.keras")
    g_model_AtoB.save(latest_AtoB)
    g_model_BtoA.save(latest_BtoA)
    print(f"> Updated latest models: {latest_AtoB}  AND  {latest_BtoA}")

# -----------------------------
# Summarize performance (3D)
# -----------------------------

def summarize_performance(
    step,
    g_model,
    trainX,
    name,
    patch_size=(64, 64, 32),
    n_patches=5,
    out_dir="outputs/cyclegan3d"
):
    """
    Visualize patches from ONE 5D volume.
    
    trainX shape : (N, H, W, D, C)
    """

    os.makedirs(out_dir, exist_ok=True)

    ph, pw, pd = patch_size

    # ---- pick ONE volume ----
    idx = np.random.randint(len(trainX))
    vol = trainX[idx]        # (H,W,D,C)
    vol = vol[..., 0]        # (H,W,D)

    H, W, D = vol.shape
    mid = pd // 2

    inputs, outputs = [], []

    for _ in range(n_patches):
        # random patch location
        x = np.random.randint(0, H - ph + 1)
        y = np.random.randint(0, W - pw + 1)
        z = np.random.randint(0, D - pd + 1)

        patch = vol[x:x+ph, y:y+pw, z:z+pd]

        # model input
        patch_in = patch[np.newaxis, ..., np.newaxis]
        fake = g_model.predict(patch_in, verbose=0)[0, ..., 0]

        inputs.append(patch[:, :, mid])
        outputs.append(fake[:, :, mid])

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    # ---- Plot (no gaps) ----
    fig, ax = plt.subplots(
        2, n_patches,
        figsize=(n_patches * 3, 6),
        gridspec_kw={"wspace": 0, "hspace": 0}
    )

    for i in range(n_patches):
        ax[0, i].imshow(inputs[i], cmap="gray")
        ax[0, i].axis("off")

        ax[1, i].imshow(outputs[i], cmap="gray")
        ax[1, i].axis("off")

    plt.subplots_adjust(0, 0, 1, 1)

    filename = os.path.join(
        out_dir, f"{name}_single_vol_patches_step_{step+1:06d}.png"
    )
    plt.savefig(filename, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"[Saved] {filename}")

# -----------------------------
# Extract aligned random patch from 5D volumes
# -----------------------------
def extract_random_patch_pair(volA, volB, patch_size):
    """
    volA, volB: 5D arrays (H,W,D,C)
    patch_size: tuple (ph,pw,pd)
    Returns aligned patches of shape (ph,pw,pd,C)
    """
    H, W, D, C = volA.shape
    ph, pw, pd = patch_size

    ph = min(ph, H)
    pw = min(pw, W)
    pd = min(pd, D)

    x = np.random.randint(0, H - ph + 1)
    y = np.random.randint(0, W - pw + 1)
    z = np.random.randint(0, D - pd + 1)

    patchA = volA[x:x+ph, y:y+pw, z:z+pd, :]
    patchB = volB[x:x+ph, y:y+pw, z:z+pd, :]
    return patchA, patchB


# -----------------------------
# Generate batch of random real patches
# -----------------------------
def generate_real_patches(trainA, trainB, n_batch, patch_size):
    """
    Generate batch of aligned patches from lists/arrays of 5D volumes.
    Returns XA, XB of shape (n_batch, ph, pw, pd, C)
    """
    XA, XB = [], []

    for _ in range(n_batch):
        idx = np.random.randint(len(trainA))
        volA = trainA[idx]
        volB = trainB[idx]
        pA, pB = extract_random_patch_pair(volA, volB, patch_size)
        XA.append(pA)
        XB.append(pB)

    XA = np.array(XA)
    XB = np.array(XB)
    return XA, XB


def generate_real_samples(X, n_patch):
    """
    X: batch of patches (B,H,W,D,C)
    n_patch: tuple of PatchGAN output shape (H_out, W_out, D_out)
    Returns X and PatchGAN labels of ones
    """
    y = np.ones((X.shape[0], *n_patch, 1), dtype='float32')
    return X, y

def generate_fake_samples(g_model, X, n_patch):
    """
    X: batch of real patches
    g_model: generator model
    n_patch: tuple of PatchGAN output shape (H_out, W_out, D_out)
    Returns fake patches and PatchGAN labels of zeros
    """
    X_fake = g_model.predict(X, verbose=0)
    y_fake = np.zeros((X_fake.shape[0], *n_patch, 1), dtype='float32')
    return X_fake, y_fake


# ----------------------------
# Training loop (3D)
# ----------------------------
# train cyclegan models
def train_3d(
    d_model_A, d_model_B,
    g_model_AtoB, g_model_BtoA,
    c_model_AtoB, c_model_BtoA,
    dataset,
    patch_size=(64,64,32),
    epochs=500,
    n_batch=1,
    n_samples_vis=3,
    output_path='src_simulated/outputs/cyclegan3d_t1_t2'
):
    
    # define properties of the training run
    n_epochs, n_batch = epochs, 1  # batch size fixed to 1
    n_patch = d_model_A.output_shape[1:-1]  # (H_out, W_out, D_out)

    trainA, trainB = dataset
    poolA, poolB = list(), list()

    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * epochs

    for i in range(n_steps):

        # Real patches
        X_realA, X_realB = generate_real_patches(trainA, trainB, n_batch=1, patch_size=patch_size)

        # PatchGAN labels
        X_realA, y_realA = generate_real_samples(X_realA, n_patch)
        X_realB, y_realB = generate_real_samples(X_realB, n_patch)

        # Freeze generators, train discriminators
        g_model_AtoB.trainable = False
        g_model_BtoA.trainable = False
        d_model_A.trainable = True
        d_model_B.trainable = True

        # Fake patches + PatchGAN labels
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)

        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)

        # -----------------------------
        # 3. TRAIN GENERATORS (composite)
        # -----------------------------
        g_model_AtoB.trainable = True
        g_model_BtoA.trainable = True
        d_model_A.trainable = False
        d_model_B.trainable = False

        g_loss_BtoA = c_model_BtoA.train_on_batch(
            [X_realB, X_realA],
            [y_realA, X_realA, X_realB, X_realA]
        )

        g_loss_AtoB = c_model_AtoB.train_on_batch(
            [X_realA, X_realB],
            [y_realB, X_realB, X_realA, X_realB]
        )

        # -----------------------------
        # 4. TRAIN DISCRIMINATORS
        # -----------------------------
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        # -----------------------------
        # 5. LOG PROGRESS
        # -----------------------------
        if (i + 1) % 10 == 0:
            print(
                f"Iteration>{i+1}, "
                f"dA[{dA_loss1:.3f}, {dA_loss2:.3f}] "
                f"dB[{dB_loss1:.3f}, {dB_loss2:.3f}] "
                f"g[{g_loss_AtoB[0]:.3f}, {g_loss_BtoA[0]:.3f}]"
            )

        # # -----------------------------
        # # 6. PERFORMANCE CHECK
        # # -----------------------------
        # if (i + 1) % (bat_per_epo * 10) == 0:
        #     summarize_performance(
        #         step=i,
        #         g_model=g_model_AtoB,
        #         trainX=trainA,
        #         name='AtoB',
        #         patch_size=patch_size,
        #         output_path=output_path
        #     )
        #     summarize_performance(
        #         step=i,
        #         g_model=g_model_BtoA,
        #         trainX=trainB,
        #         name='BtoA',
        #         patch_size=patch_size,
        #         output_path=output_path
        #     )

        # -----------------------------
        # 7. MODEL SAVE
        # -----------------------------
        if (i + 1) % (bat_per_epo * 5) == 0:
            save_models(i, g_model_AtoB, g_model_BtoA, output_path=output_path)

# ----------------------------
## Read dataset

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

import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

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


# generator: A -> B
g_model_AtoB = define_generator_3d(image_shape)
# generator: B -> A
g_model_BtoA = define_generator_3d(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator_3d(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator_3d(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model_3d(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model_3d(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

from datetime import datetime
start1 = datetime.now()
# train models
train_3d(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=500)

stop1 = datetime.now()
#Execution time of the model
execution_time = stop1-start1
print("Execution time is: ", execution_time)