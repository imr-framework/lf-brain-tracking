import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

# --- Base Losses ---
def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def l2_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def ssim_loss(y_true, y_pred):
    # 1 - SSIM → smaller = better
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# --- Edge Loss (Gradient Difference Loss) ---
def edge_loss(y_true, y_pred):
    """Encourages sharper edges using Sobel-like gradients."""
    def gradient(img):
        dx = img[:, 1:, :, :] - img[:, :-1, :, :]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        return dx, dy
    
    dx_true, dy_true = gradient(y_true)
    dx_pred, dy_pred = gradient(y_pred)
    return tf.reduce_mean(tf.abs(dx_true - dx_pred)) + tf.reduce_mean(tf.abs(dy_true - dy_pred))


# --- Composite Losses ---

def l1_l2_ssim_loss(y_true, y_pred, w1=0.2, w2=0.5, w3=0.3):
    return w1 * l1_loss(y_true, y_pred) + w2 * l2_loss(y_true, y_pred) + w3 * ssim_loss(y_true, y_pred)

def l2_ssim_loss(y_true, y_pred, w2=0.5, w3=0.5):
    return w2 * l2_loss(y_true, y_pred) + w3 * ssim_loss(y_true, y_pred)

def l1_ssim_loss(y_true, y_pred, w1=0.3, w3=0.7):
    return w1 * l1_loss(y_true, y_pred) + w3 * ssim_loss(y_true, y_pred)

def l2_ssim_edge_loss(y_true, y_pred, w2=0.3, w3=0.5, w4=0.2):
    """MSE + SSIM + Edge (great for LF-MRI denoising)"""
    return w2 * l2_loss(y_true, y_pred) + w3 * ssim_loss(y_true, y_pred) + w4 * edge_loss(y_true, y_pred)

def l1_l2_ssim_edge_loss(y_true, y_pred, w1=0.3, w2=0.2, w3=0.3, w4=0.2):
    """MSE + SSIM + Edge (great for LF-MRI denoising)"""
    return w1 * l1_loss(y_true, y_pred) + w2 * l2_loss(y_true, y_pred) + w3 * ssim_loss(y_true, y_pred) + w4 * edge_loss(y_true, y_pred)

# incorporate GRAM matrix loss for texture preservation
def gram_matrix(y_true, y_pred):    
    """Compute Gram matrix for texture representation."""
    b, h, w, c = tf.shape(y_pred)
    features = tf.reshape(y_pred, (b, h * w, c))
    gram_y_pred = tf.matmul(features, features, transpose_a=True) / tf.cast(h * w * c, tf.float32)

    features_y_true = tf.reshape(y_true, (b, h * w, c))
    gram_y_true = tf.matmul(features_y_true, features_y_true, transpose_a=True) / tf.cast(h * w * c, tf.float32)        
    gram = tf.reduce_mean(tf.abs(gram_y_pred - gram_y_true))

    return gram