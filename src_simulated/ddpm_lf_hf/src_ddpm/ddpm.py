import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Paths
# -----------------------------
HF_PATH = "Data/HF-T2/T2/"
CHECKPOINT_DIR = "checkpoints/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------
# Hyperparameters
# -----------------------------
H, W = 128, 128
CHANNELS = 3
BATCH_SIZE = 4
EPOCHS = 100
T = 1000
LEARNING_RATE = 1e-4
SIGMA_NOISE = 0.08
GAUSS_SIGMA = 1.5

# -----------------------------
# Diffusion schedule
# -----------------------------
beta = tf.constant(np.linspace(1e-4, 0.02, T), dtype=tf.float32)
alpha = 1.0 - beta
alpha_bar = tf.math.cumprod(alpha)

# -----------------------------
# Sinusoidal timestep embedding
# -----------------------------
def timestep_embedding(timesteps, dim=128):
    half = dim // 2
    freqs = np.exp(-np.log(10000) * np.arange(0, half) / half)
    args = tf.cast(timesteps, tf.float32)[:, None] * freqs[None, :]
    embedding = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
    return embedding

# -----------------------------
# LF simulation
# -----------------------------
def normalize(vol):
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
    return vol * 2 - 1

def degrade_to_lf(vol, sigma=GAUSS_SIGMA, noise_std=SIGMA_NOISE):
    vol = gaussian_filter(vol, sigma=sigma)
    vol = zoom(vol, 0.5, order=1)
    vol = zoom(vol, 2.0, order=1)
    noise = np.random.normal(0, noise_std, vol.shape)
    return vol + noise

def resample_to_1mm(vol, original_voxel_size, target_voxel_size=(1,1,2)):
    zoom_factors = tuple(o / t for o, t in zip(original_voxel_size, target_voxel_size))
    return zoom(vol, zoom_factors, order=1)

def crop_center(vol, target_shape=(128,128,35)):
    h, w, d = vol.shape
    th, tw, td = target_shape
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    start_d = (d - td) // 2
    return vol[start_h:start_h+th, start_w:start_w+tw, start_d:start_d+td]

# -----------------------------
# Dataset generator
# -----------------------------
def load_volumes():
    hf_files = sorted(os.listdir(HF_PATH))
    for f in hf_files:
        hf = nib.load(HF_PATH + f).get_fdata()
        hdr = nib.load(HF_PATH + f).header
        original_voxel_size = hdr.get_zooms()
        hf = normalize(hf)
        hf = np.rot90(hf, k=3, axes=(0,1))
        hf = resample_to_1mm(hf, original_voxel_size)
        lf = degrade_to_lf(hf)
        hf = crop_center(hf)
        lf = crop_center(lf)
        yield hf, lf

def slice_2p5d(vol, idx):
    return np.stack([vol[:,:,idx-1], vol[:,:,idx], vol[:,:,idx+1]], axis=-1)

def dataset_generator():
    for hf, lf in load_volumes():
        for i in range(1, hf.shape[2]-1):
            yield slice_2p5d(hf, i), slice_2p5d(lf, i)

# -----------------------------
# DDPM forward noise addition
# -----------------------------
def add_noise(x0, t):
    eps = tf.random.normal(tf.shape(x0), dtype=tf.float32)
    a = tf.sqrt(tf.gather(alpha_bar, t))
    b = tf.sqrt(1.0 - tf.gather(alpha_bar, t))
    a = tf.reshape(a, [-1,1,1,1])
    b = tf.reshape(b, [-1,1,1,1])
    xt = a * x0 + b * eps
    return xt, eps

# -----------------------------
# UNet with residual blocks and timestep embedding
# -----------------------------
def ResBlock(x, ch):
    h = Conv2D(ch,3,padding='same')(x)
    h = BatchNormalization()(h)
    h = Activation('swish')(h)
    h = Conv2D(ch,3,padding='same')(h)
    h = BatchNormalization()(h)
    if x.shape[-1] != ch:
        x = Conv2D(ch,1,padding='same')(x)
    return Activation('swish')(x+h)

def Down(x, ch):
    x = Conv2D(ch,3,strides=2,padding='same')(x)
    return Activation('swish')(x)

def Up(x, ch):
    x = UpSampling2D()(x)
    x = Conv2D(ch,3,padding='same')(x)
    return Activation('swish')(x)

def DiffusionUNet(shape, time_embed_dim=128):
    xt = Input(shape)
    lf = Input(shape)
    t = Input((time_embed_dim,))
    # expand t embedding
    t_emb = Dense(np.prod(shape))(t)
    t_emb = Reshape(shape)(t_emb)
    x = Concatenate()([xt, lf, t_emb])

    d1 = ResBlock(x,64)
    d1d = Down(d1,128)
    d2 = ResBlock(d1d,128)
    d2d = Down(d2,256)
    d3 = ResBlock(d2d,256)
    d3d = Down(d3,512)
    mid = ResBlock(d3d,512)
    u2 = Up(mid,256)
    u2 = ResBlock(Concatenate()([u2,d3]),256)
    u1 = Up(u2,128)
    u1 = ResBlock(Concatenate()([u1,d2]),128)
    u0 = Up(u1,64)
    u0 = ResBlock(Concatenate()([u0,d1]),64)
    out = Conv2D(3,1)(u0)
    return Model([xt, lf, t], out)

# -----------------------------
# Dataset
# -----------------------------
ds = tf.data.Dataset.from_generator(
    dataset_generator,
    output_signature=(
        tf.TensorSpec((H,W,CHANNELS), tf.float32),
        tf.TensorSpec((H,W,CHANNELS), tf.float32)
    )
).shuffle(50).batch(BATCH_SIZE)

# -----------------------------
# Model
# -----------------------------
model = DiffusionUNet((H,W,CHANNELS))
model.compile(optimizer=Adam(LEARNING_RATE), loss='mse')

# -----------------------------
# Training loop (true DDPM)
# -----------------------------
for epoch in range(EPOCHS):
    for hf, lf in ds:
        batch_size = hf.shape[0]
        t = tf.random.uniform([batch_size], 0, T, dtype=tf.int32)
        xt, eps = add_noise(hf, t)
        t_emb = timestep_embedding(t)
        loss = model.train_on_batch([xt, lf, t_emb], eps)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")
    if (epoch+1) % 10 == 0:
        model.save_weights(os.path.join(CHECKPOINT_DIR, f"ddpm_epoch{epoch+1}.weights.h5"))

model.save_weights(os.path.join(CHECKPOINT_DIR, "ddpm_final.weights.h5"))
print("Training completed!")

# -----------------------------
# Prediction on LF volume
# -----------------------------
def predict_ddpm(model, lf_vol, n_steps=T):
    D = lf_vol.shape[2]
    hf_slices = []
    for i in range(1,D-1):
        x = slice_2p5d(lf_vol,i)
        x = tf.expand_dims(x,0)
        for t in reversed(range(n_steps)):
            t_batch = tf.constant([t], dtype=tf.int32)
            t_emb = timestep_embedding(t_batch)
            eps_pred = model([x, x, t_emb], training=False)
            a = tf.sqrt(alpha[t])
            b = tf.sqrt(1-alpha[t])
            x = (x - b * eps_pred)/a
            if t>0:
                x = x + tf.sqrt(beta[t])*tf.random.normal(tf.shape(x))
        hf_slices.append(x[0].numpy())
    return np.stack(hf_slices, axis=2)

# -----------------------------
# Sample prediction
# -----------------------------
hf_sample, _ = next(load_volumes())
lf_sample = degrade_to_lf(hf_sample)
lf_sample = crop_center(lf_sample)

hf_pred = predict_ddpm(model, lf_sample)

center = hf_pred.shape[2]//2
plt.imshow((hf_pred[:,:,center,1]+1)/2, cmap='gray')
plt.title("Predicted HF from LF")
plt.show()
