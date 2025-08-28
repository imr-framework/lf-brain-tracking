import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError

# def psnr(y_true, y_pred):
#     # reshape: (batch, depth, height, width, ch) -> (batch*depth, height, width, ch)
#     y_true_2d = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
#     y_pred_2d = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
#     return tf.reduce_mean(tf.image.psnr(y_true_2d, y_pred_2d, max_val=1.0))

# def ssim(y_true, y_pred):
#     y_true_2d = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
#     y_pred_2d = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
#     return tf.reduce_mean(tf.image.ssim(y_true_2d, y_pred_2d, max_val=1.0))

import tensorflow as tf
import tensorflow.keras.backend as K

# Your existing functions
def psnr(y_true, y_pred):
    # reshape: (batch, depth, height, width, ch) -> (batch*depth, height, width, ch)
    y_true_2d = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    y_pred_2d = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
    return tf.reduce_mean(tf.image.psnr(y_true_2d, y_pred_2d, max_val=1.0))

def ssim(y_true, y_pred):
    y_true_2d = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    y_pred_2d = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
    return tf.reduce_mean(tf.image.ssim(y_true_2d, y_pred_2d, max_val=1.0))

# Simple MSE
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Composite loss
def composite_loss(y_true, y_pred, alpha=0.8, beta=0.2, gamma=0.2):
    l_mse = mse(y_true, y_pred)
    l_ssim = 1.0 - ssim(y_true, y_pred)  # (maximize SSIM)
    # l_psnr = -psnr(y_true, y_pred)       # (maximize PSNR)
    return alpha * l_mse + beta * l_ssim
