import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tqdm import tqdm
import sys
sys.path.insert(0, './') 
import os
# -------------------------
from src_simulated.config import config
data_folder = config.data_folder
subjects = config.subjects
train_day = 1
val_day = 2
test_days = [5]

# -------------------------
# Diffusion utilities
# -------------------------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)

class Diffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cum = np.cumprod(self.alphas)
        self.sqrt_alpha_cum = np.sqrt(self.alpha_cum)
        self.sqrt_one_minus_alpha_cum = np.sqrt(1.0 - self.alpha_cum)
    
    def q_sample(self, x0, t, noise):
        """
        Forward diffusion: add noise to x0 at timestep t
        """
        sqrt_alpha_cum_t = self.sqrt_alpha_cum[t]
        sqrt_one_minus_alpha_cum_t = self.sqrt_one_minus_alpha_cum[t]
        return sqrt_alpha_cum_t * x0 + sqrt_one_minus_alpha_cum_t * noise

# -------------------------
# Time embedding for UNet
# -------------------------
def get_timestep_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = np.expand_dims(timesteps, 1) * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
    return tf.convert_to_tensor(emb, dtype=tf.float32)

# -------------------------
# 3D UNet with residual blocks
# -------------------------
def conv_block(x, out_ch):
    x_res = x
    x = layers.Conv3D(out_ch, 3, padding='same')(x)
    x = layers.Activation('swish')(x)
    x = layers.Conv3D(out_ch, 3, padding='same')(x)
    # Adjust residual channels if needed
    if x_res.shape[-1] != out_ch:
        x_res = layers.Conv3D(out_ch, 1, padding='same')(x_res)
    x = layers.Add()([x, x_res])
    x = layers.Activation('swish')(x)
    return x

def downsample_block(x, out_ch):
    """
    Returns:
        x_out  = pooled + conv-block output
        x_skip = pre-pooled skip tensor
    """
    x_skip = x                       # save original (skip)
    x = layers.MaxPool3D()(x)
    x = conv_block(x, out_ch)
    return x, x_skip

def upsample_block(x, skip, out_ch):
    x = layers.UpSampling3D()(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, out_ch)
    return x

def build_unet3d(input_shape=(140,140,35,2), base_ch=32, time_emb_dim=128):

    inp = layers.Input(shape=input_shape)
    t_inp = layers.Input(shape=(1,), dtype=tf.float32)

    # ---- Time embedding ----
    t_emb = layers.Dense(time_emb_dim, activation='swish')(t_inp)
    t_emb = layers.Dense(time_emb_dim, activation='swish')(t_emb)
    t_emb = layers.Reshape((1,1,1,time_emb_dim))(t_emb)

    # -----------------------
    # Encoder (RETURN skip!)
    # -----------------------
    x1 = conv_block(inp, base_ch)           # no pooling → skip1

    x2, skip2 = downsample_block(x1, base_ch*2)

    x3, skip3 = downsample_block(x2, base_ch*4)

    # -----------------------
    # Bottleneck + time
    # -----------------------
    # Bottleneck
    b = conv_block(x3, base_ch*4)

    # Time embedding projection
    t_proj = layers.Conv3D(base_ch*4, 1, padding='same')(t_emb)
    b = layers.Add()([b, t_proj])
    # -----------------------
    # Decoder
    # -----------------------
    d3 = upsample_block(b, skip3, base_ch*2)
    d2 = upsample_block(d3, skip2, base_ch)

    out = layers.Conv3D(1, 1, padding='same')(d2)

    return models.Model([inp, t_inp], out)

# -------------------------
# Data generator for diffusion
# -------------------------
def diffusion_batch_generator(X, y, diffusion, batch_size):
    num_samples = X.shape[0]
    while True:
        idx = np.random.choice(num_samples, batch_size)
        X_batch = X[idx]
        y_batch = y[idx]
        t_batch = np.random.randint(0, diffusion.timesteps, size=(batch_size,))
        noise = np.random.randn(*y_batch.shape).astype(np.float32)
        x_t = np.zeros_like(y_batch)
        for i in range(batch_size):
            x_t[i] = diffusion.q_sample(y_batch[i], t_batch[i], noise[i])
        t_norm = t_batch / diffusion.timesteps  # normalize
        # Input channels: LF + optional noise-map
        yield [np.concatenate([X_batch, np.zeros_like(X_batch)], axis=-1), t_norm[:,None]], noise

# -------------------------
# Training function
# -------------------------

def train_model(X_train, y_train, X_val, y_val, timesteps=500, batch_size=2, epochs=10):
    diffusion = Diffusion(timesteps=timesteps)
    model = build_unet3d(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], 2))
    model.compile(optimizer=optimizers.Adam(1e-4), loss='mse')
    train_gen = diffusion_batch_generator(X_train, y_train, diffusion, batch_size)
    val_gen = diffusion_batch_generator(X_val, y_val, diffusion, batch_size)
    steps_per_epoch = max(1, X_train.shape[0] // batch_size)
    val_steps = max(1, X_val.shape[0] // batch_size)

    model.fit(train_gen,
              validation_data=val_gen,
              steps_per_epoch=steps_per_epoch,
              validation_steps=val_steps,
              epochs=epochs)
    return model, diffusion

# -------------------------
# Inference
# -------------------------
def denoise_volume(model, diffusion, y_noisy, steps=100, noise_map=None):
    x = np.random.randn(*y_noisy.shape).astype(np.float32)
    for t in reversed(range(steps)):
        t_norm = np.array([t / diffusion.timesteps], dtype=np.float32).reshape(1,1)
        x_in = x[None,...]
        cond = y_noisy[None,...]
        if noise_map is not None:
            cond = np.concatenate([cond, noise_map[None,...]], axis=-1)
        else:
            cond = np.concatenate([cond, np.zeros_like(cond)], axis=-1)
        pred_noise = model.predict([cond, t_norm], verbose=0)[0]
        alpha_cum_t = diffusion.alpha_cum[t]
        beta_t = diffusion.betas[t]
        alpha_t = diffusion.alphas[t]
        # predict x0
        x0_pred = (x - np.sqrt(1 - alpha_cum_t) * pred_noise) / np.sqrt(alpha_cum_t)
        if t > 0:
            sigma = np.sqrt(beta_t)
            x = np.sqrt(alpha_t) * x0_pred + np.sqrt(1 - alpha_t) * np.random.randn(*x0_pred.shape) * sigma
        else:
            x = x0_pred
    return x


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
# NORMALIZATION
# -----------------------------
def normalize_volume(vol, method='minmax'):
    if method=='minmax':
        vol_min, vol_max = vol.min(), vol.max()
        if vol_max - vol_min > 0:
            vol = (vol - vol_min) / (vol_max - vol_min)
        else:
            vol = np.zeros_like(vol)
    elif method=='zscore':
        mean, std = vol.mean(), vol.std()
        if std>0:
            vol = (vol - mean) / std
        else:
            vol = np.zeros_like(vol)
    return vol

def normalize_dataset(X, y, method='minmax'):
    X_norm = np.array([normalize_volume(vol, method) for vol in X])
    y_norm = np.array([normalize_volume(vol, method) for vol in y])
    return X_norm, y_norm
# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":


    # Load data
    X_train, y_train = load_data_for_days(config.subjects, [config.train_day])
    X_val, y_val     = load_data_for_days(config.subjects, [config.val_day])
    X_test, y_test   = load_data_for_days(config.subjects, config.test_days)

    # Normalize
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
    # Train
    model, diffusion = train_model(X_train, y_train, X_val, y_val, timesteps=500, batch_size=2, epochs=5)

    # Inference example
    denoised = denoise_volume(model, diffusion, X_test[0], steps=200)
    print("Denoised shape:", denoised.shape)
