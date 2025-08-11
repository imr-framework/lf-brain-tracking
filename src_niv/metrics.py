import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError

def psnr(y_true, y_pred):
    # reshape: (batch, depth, height, width, ch) -> (batch*depth, height, width, ch)
    y_true_2d = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    y_pred_2d = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
    return tf.reduce_mean(tf.image.psnr(y_true_2d, y_pred_2d, max_val=1.0))

def ssim(y_true, y_pred):
    y_true_2d = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    y_pred_2d = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
    return tf.reduce_mean(tf.image.ssim(y_true_2d, y_pred_2d, max_val=1.0))