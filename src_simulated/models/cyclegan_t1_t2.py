"""
cycleGAN model

Based on the code by Jason Brownlee from his blogs on https://machinelearningmastery.com/
I am adapting his code to various applications but original credit goes to Jason.

The model uses instance normalization layer:
Normalize the activations of the previous layer at each step,
i.e. applies a transformation that maintains the mean activation
close to 0 and the activation standard deviation close to 1.
Standardizes values on each output feature map rather than across features in a batch.

Download instance normalization code from here: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py
Or install keras_contrib using guidelines here: https://github.com/keras-team/keras-contrib
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import cv2

from random import random
from numpy import load, zeros, ones, asarray
from numpy.random import randint

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, LeakyReLU,
    Activation, Concatenate, Add
)

from matplotlib import pyplot

# discriminator model (70x70 patchGAN)
# C64-C128-C256-C512
#After the last layer, conv to 1-dimensional output, followed by a Sigmoid function.
# The “axis” argument is set to -1 for instance norm. to ensure that features are normalized per feature map.

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

def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)
	# C64: 4x4 kernel Stride 2x2
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128: 4x4 kernel Stride 2x2
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256: 4x4 kernel Stride 2x2
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512: 4x4 kernel Stride 2x2
    # Not in the original paper. Comment this block if you want.
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer : 4x4 kernel but Stride 1x1
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# define model
	model = Model(in_image, patch_out)
	# compile model
    #The model is trained with a batch size of one image and Adam opt.
    #with a small learning rate and 0.5 beta.
    #The loss for the discriminator is weighted by 50% for each model update.
    #This slows down changes to the discriminator relative to the generator model during training.
	model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
	model.summary()
	return model

# generator a resnet block to be used in the generator
# residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layers.
def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Add()([g, input_layer])
	return g

# define the  generator model - encoder-decoder type architecture

#c7s1-k denote a 7×7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1.
#dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2.
# Rk denotes a residual block that contains two 3 × 3 convolutional layers
# uk denotes a 3 × 3 fractional-strided-Convolution InstanceNorm-ReLU layer with k filters and stride 1/2

#The network with 6 residual blocks consists of:
#c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3

#The network with 9 residual blocks consists of:
#c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128, u64,c7s1-3

def define_generator(image_shape, n_resnet=9):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(1, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	model.summary()
	return model

#We define a composite model that will be used to train each generator separately.
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	# Make the generator of interest trainable as we will be updating these weights.
    #by keeping other models constant.
    #Remember that we use this same function to train both generators,
    #one generator at a time.
	g_model_1.trainable = True
	# mark discriminator and second generator as non-trainable
	d_model.trainable = False
	g_model_2.trainable = False

	# adversarial loss
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# identity loss
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# cycle loss - forward
	output_f = g_model_2(gen1_out)
	# cycle loss - backward
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)

	# define model graph
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])

    # define the optimizer
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
	model.compile(loss=['mse', 'mae', 'mae', 'mae'],
               loss_weights=[1, 5, 10, 10], optimizer=opt)
	model.summary()
	return model

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
def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y

# generate a batch of images, returns images and targets
#Remember that for fake images the label (y) is 0.
def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake images
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y


def save_models(step, g_model_AtoB, g_model_BtoA, output_path='src_simulated/outputs/cyclegan_t1_t2'):
    # create directory if not exists
    os.makedirs(output_path, exist_ok=True)

    # -----------------------------
    # 1. Save checkpoint models
    # -----------------------------
    # if step % 100 == 0:
	# 	ckpt_AtoB = os.path.join(output_path, f"g_model_AtoB_{step+1:09d}.keras")
	# 	ckpt_BtoA = os.path.join(output_path, f"g_model_BtoA_{step+1:09d}.keras")

	# 	g_model_AtoB.save(ckpt_AtoB)
	# 	g_model_BtoA.save(ckpt_BtoA)

    # -----------------------------
    # 2. Save "latest" models (overwrite)
    # -----------------------------
    latest_AtoB = os.path.join(output_path, "g_model_AtoB_latest.keras")
    latest_BtoA = os.path.join(output_path, "g_model_BtoA_latest.keras")

    g_model_AtoB.save(latest_AtoB)
    g_model_BtoA.save(latest_BtoA)

    # print(f"> Saved checkpoint: {ckpt_AtoB}  AND  {ckpt_BtoA}")
    print(f"> Updated latest models: {latest_AtoB}  AND  {latest_BtoA}")

# periodically generate images using the save model and plot input and output images
def summarize_performance(step, g_model, trainX, name, n_samples=5, output_path='src_simulated/outputs/cyclegan_t1_t2'):
	# select a sample of input images
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# generate translated images
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# scale all pixels from [-1,1] to [0,1]
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	# plot real images
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_in[i])
	# plot translated image
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])
	# save plot to file
	filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	filename1 = os.path.join(output_path, filename1)
	pyplot.savefig(filename1)
	pyplot.close()

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
		elif random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)



# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA,
          c_model_AtoB, c_model_BtoA, dataset, epochs=500):

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

    # enumerate epochs
    for i in range(n_steps):

        # -----------------------------
        # 1. REAL SAMPLES
        # -----------------------------
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)

        # -----------------------------
        # 2. FAKE SAMPLES
        # -----------------------------
        # Freeze generators, train discriminators
        g_model_AtoB.trainable = False
        g_model_BtoA.trainable = False
        d_model_A.trainable = True
        d_model_B.trainable = True

        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)

        # update images in pool (buffer of 50 images)
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)

        # -----------------------------
        # 3. TRAIN GENERATORS
        # -----------------------------
        # Freeze discriminators, train generators via composite model
        g_model_AtoB.trainable = True
        g_model_BtoA.trainable = True
        d_model_A.trainable = False
        d_model_B.trainable = False

        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch(
            [X_realB, X_realA],
            [y_realA, X_realA, X_realB, X_realA]
        )

        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch(
            [X_realA, X_realB],
            [y_realB, X_realB, X_realA, X_realB]
        )

        # -----------------------------
        # 4. TRAIN DISCRIMINATORS
        # -----------------------------
        # Freeze generators again
        g_model_AtoB.trainable = False
        g_model_BtoA.trainable = False
        d_model_A.trainable = True
        d_model_B.trainable = True

        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        # -----------------------------
        # 5. LOG PROGRESS
        # -----------------------------
        if (i + 1) % 10 == 0:
            print(
                'Iteration>%d, dA[%.3f, %.3f] dB[%.3f, %.3f] g[%.3f, %.3f]' %
                (i + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2)
            )

        # -----------------------------
        # 6. PERIODIC PERFORMANCE CHECK
        # -----------------------------
        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            summarize_performance(i, g_model_BtoA, trainB, 'BtoA')

        # -----------------------------
        # 7. PERIODIC MODEL SAVE
        # -----------------------------
        if (i + 1) % (bat_per_epo * 5) == 0:
            save_models(i, g_model_AtoB, g_model_BtoA,
                        output_path='src_simulated/outputs/cyclegan_t1_t2')

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

# discard first three slices and last 2 slices of dataA and last five slices of dataB to have depth 30

# load image data
data = [dataA, dataB]

import numpy as np

# Assuming data = [dataA, dataB]
for i in range(len(data)):
    # Rotate 90° counter-clockwise in-plane (H x W)
    data[i] = np.rot90(data[i], k=1, axes=(0, 1))

# Now data contains the rotated volumes
print("Rotated volumes shapes:", [d.shape for d in data])


print('Loaded', data[0].shape, data[1].shape)

# ----------------------------
# Convert Domain A → 2D (256x256)
# ----------------------------
A_slices = []
for i in range(dataA.shape[0]):          # number of volumes
    for z in range(dataA.shape[3]):      # number of slices
        slice_2d = dataA[i, :, :, z]
        slice_2d = cv2.resize(slice_2d, (140, 140), interpolation=cv2.INTER_LINEAR)
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
        slice_2d = cv2.resize(slice_2d, (140, 140), interpolation=cv2.INTER_LINEAR)
        B_slices.append(slice_2d)

B_2D = np.array(B_slices)
print("B_2D:", B_2D.shape)   # expected → (40*35, 256, 256)

# Suppose A_2D.shape = (N, 256, 256)
A_2D = A_2D[..., np.newaxis]   # (N, 256, 256, 1)
# A_2D = np.repeat(A_2D, 3, axis=-1)  # (N, 256, 256, 3)

B_2D = B_2D[..., np.newaxis]
# B_2D = np.repeat(B_2D, 3, axis=-1)  # (N, 256, 256, 3)
print(A_2D.shape, B_2D.shape)

data = [A_2D, B_2D]

#print datatype of each
print("A_2D dtype:", A_2D.dtype)
print("B_2D dtype:", B_2D.dtype)

def inspect_domains(A, B, n_samples=20):
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

dataset = data

inspect_domains(dataset[0], dataset[1], n_samples=20)

# from cycleGAN_model import define_generator, define_discriminator, define_composite_model, train
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
print(image_shape)

# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

from datetime import datetime
start1 = datetime.now()
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=500)

stop1 = datetime.now()
#Execution time of the model
execution_time = stop1-start1
print("Execution time is: ", execution_time)

############################################