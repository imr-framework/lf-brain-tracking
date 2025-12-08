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
    Activation, Concatenate
)

import nibabel as nib

# If keras_contrib isn't available, try:
# from tfa.layers import InstanceNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

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
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv3D(128, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv3D(256, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512 (optional)
    d = Conv3D(512, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last (stride 1)
    d = Conv3D(512, (4,4,4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output (single channel)
    patch_out = Conv3D(1, (4,4,4), padding='same', kernel_initializer=init)(d)
    model = Model(in_image, patch_out)
    # compile
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
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
    g = Concatenate()([g, input_layer])  # residual-ish (concatenate as original code did)
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
def load_real_samples_3d(filename):
    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    # scale to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

def generate_real_samples_3d(dataset, n_samples, patch_shape):
    # dataset: (n_samples_total, D, H, W, C)
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    # patch_shape is a tuple (pd, ph, pw)
    pd, ph, pw = patch_shape
    y = ones((n_samples, pd, ph, pw, 1))
    return X, y

def generate_fake_samples_3d(g_model, dataset, patch_shape):
    X = g_model.predict(dataset)
    pd, ph, pw = patch_shape
    y = zeros((len(X), pd, ph, pw, 1))
    return X, y

# ----------------------------
# Image pool update (same logic)
# ----------------------------
def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            selected.append(image)
        else:
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)

# ----------------------------
# Save models
# ----------------------------
def save_models_3d(step, g_model_AtoB, g_model_BtoA):
    filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
    filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
    g_model_AtoB.save(filename1)
    g_model_BtoA.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

# ----------------------------
# Summarize performance (volume -> show central axial slices)
# ----------------------------
def summarize_performance_3d(step, g_model, trainX, name, n_samples=3):
    # choose some samples
    X_in, _ = generate_real_samples_3d(trainX, n_samples, (0,0,0))
    X_out, _ = generate_fake_samples_3d(g_model, X_in, (0,0,0))
    # scale from [-1,1] to [0,1] for display
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    # plot middle slice along depth (axis 1)
    fig, axs = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))
    for i in range(n_samples):
        vol_in = X_in[i]
        vol_out = X_out[i]
        d = vol_in.shape[0] // 2
        # if multi-channel, display first channel
        ch = 0
        axs[0, i].imshow(np.squeeze(vol_in[d, :, :, ch]), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title('input')
        axs[1, i].imshow(np.squeeze(vol_out[d, :, :, ch]), cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title('translated')
    filename = f'{name}_generated_plot_{(step+1):06d}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# ----------------------------
# Training loop (3D)
# ----------------------------
def train_3d(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA,
             c_model_AtoB, c_model_BtoA, dataset, epochs=1):
    n_epochs, n_batch = epochs, 1
    # discriminator output spatial shape (3D)
    out_shape = d_model_A.output_shape[1:4]  # (pd, ph, pw)
    # unpack dataset
    trainA, trainB = dataset
    poolA, poolB = list(), list()
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs

    for i in range(n_steps):
        # real samples
        X_realA, y_realA = generate_real_samples_3d(trainA, n_batch, out_shape)
        X_realB, y_realB = generate_real_samples_3d(trainB, n_batch, out_shape)
        # fake samples
        X_fakeA, y_fakeA = generate_fake_samples_3d(g_model_BtoA, X_realB, out_shape)
        X_fakeB, y_fakeB = generate_fake_samples_3d(g_model_AtoB, X_realA, out_shape)
        # update pools
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)

        # train generators (via composite models)
        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        # update discriminator A on real and fake
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        print('Iteration>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' %
              (i+1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))

        # summarize and save periodically
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance_3d(i, g_model_AtoB, trainA, 'AtoB')
            summarize_performance_3d(i, g_model_BtoA, trainB, 'BtoA')
        if (i+1) % (bat_per_epo * 5) == 0:
            save_models_3d(i, g_model_AtoB, g_model_BtoA)