import tensorflow as tf
import numpy as np

T = 1000

beta = tf.constant(
    np.linspace(1e-4, 0.02, T),
    dtype=tf.float32
)

alpha = 1.0 - beta
alpha_bar = tf.math.cumprod(alpha)

def add_noise(x0, t):
    """
    x0 : (B, H, W, C) float32
    t  : scalar int tensor
    """
    eps = tf.random.normal(tf.shape(x0), dtype=tf.float32)

    a = tf.sqrt(alpha_bar[t])
    b = tf.sqrt(1.0 - alpha_bar[t])

    # expand dims for broadcasting
    a = tf.reshape(a, [1, 1, 1, 1])
    b = tf.reshape(b, [1, 1, 1, 1])

    xt = a * x0 + b * eps
    return xt, eps

# ---------------------------
# Add these functions for prediction
# ---------------------------

def get_alpha_schedule():
    """
    Return alpha and alpha_bar for all timesteps (T)
    """
    return alpha.numpy(), alpha_bar.numpy()

def sample_step(x, eps_theta, t, alpha, alpha_bar):
    """
    One DDPM reverse step
    x          : current noisy image (B,H,W,C)
    eps_theta  : predicted noise
    t          : timestep integer
    alpha      : array of alpha values
    alpha_bar  : array of alpha_bar values
    """
    alpha_t = alpha[t]
    alpha_bar_t = alpha_bar[t]
    beta_t = 1.0 - alpha_t

    # reverse formula
    x_prev = (1.0 / np.sqrt(alpha_t)) * (x - ((1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * eps_theta)

    # add noise if t > 0
    if t > 0:
        z = tf.random.normal(tf.shape(x), dtype=x.dtype)
        sigma_t = np.sqrt(beta_t)
        x_prev += sigma_t * z

    return x_prev
