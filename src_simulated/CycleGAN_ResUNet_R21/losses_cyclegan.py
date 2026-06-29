import sys
sys.path.insert(0, './')
import tensorflow as tf

def mae_ssim_id(y_true, y_pred, sample_weight=None, ssim_weight_id=0.5):
    # Convert from [-1, 1] → [0, 1]
    y_true = (y_true + 1.0) / 2.0
    y_pred = (y_pred + 1.0) / 2.0

    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = 1.0 - tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=7)
    )

    loss = mae + ssim_weight_id * ssim

    if sample_weight is not None:
        loss *= sample_weight

    return loss


def mae_ssim_cycle(y_true, y_pred, sample_weight=None, ssim_weight_cycle=0.2):
    # Convert from [-1, 1] → [0, 1]
    y_true = (y_true + 1.0) / 2.0
    y_pred = (y_pred + 1.0) / 2.0

    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = 1.0 - tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=7)
    )

    loss = mae + ssim_weight_cycle * ssim

    if sample_weight is not None:
        loss *= sample_weight

    return loss

def ssim_loss(y_true, y_pred, sample_weight=None, ssim_weight=1.0):
    # Convert from [-1, 1] → [0, 1]
    y_true = (y_true + 1.0) / 2.0
    y_pred = (y_pred + 1.0) / 2.0

    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    loss = ssim_weight * (
        1.0 - tf.reduce_mean(
            tf.image.ssim(y_true, y_pred, max_val=1.0)
        )
    )

    if sample_weight is not None:
        loss *= sample_weight

    return loss

def compute_psnr(x, y):
    return tf.image.psnr(x, y, max_val=2.0)  # if normalized to [-1,1]

def compute_ssim(x, y):
    return tf.image.ssim(x, y, max_val=2.0)