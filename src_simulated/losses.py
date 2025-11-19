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

def l1_l2_ssim_loss(y_true, y_pred, w1=0.1, w2=0.6, w3=0.3):
    return w1 * l1_loss(y_true, y_pred) + w2 * l2_loss(y_true, y_pred) + w3 * ssim_loss(y_true, y_pred)

def l2_ssim_loss(y_true, y_pred, w2=0.5, w3=0.5):
    return w2 * l2_loss(y_true, y_pred) + w3 * ssim_loss(y_true, y_pred)

def l1_ssim_loss(y_true, y_pred, w1=0.3, w3=0.7):
    return w1 * l1_loss(y_true, y_pred) + w3 * ssim_loss(y_true, y_pred)

def l2_ssim_edge_loss(y_true, y_pred, w2=0.6, w3=0.3, w4=0.1):
    """MSE + SSIM + Edge (great for LF-MRI denoising)"""
    return w2 * l2_loss(y_true, y_pred) + w3 * ssim_loss(y_true, y_pred) + w4 * edge_loss(y_true, y_pred)

def l1_l2_ssim_edge_loss(y_true, y_pred, w1=0.3, w2=0.2, w3=0.3, w4=0.2):
    """MSE + SSIM + Edge (great for LF-MRI denoising)"""
    return w1 * l1_loss(y_true, y_pred) + w2 * l2_loss(y_true, y_pred) + w3 * ssim_loss(y_true, y_pred) + w4 * edge_loss(y_true, y_pred)

def l2_edge_gram_matrix_loss(y_true, y_pred, w2=0.6, w3=0.3, w4=0.1):
    """MSE + SSIM + Edge (great for LF-MRI denoising)"""
    return w2 * l2_loss(y_true, y_pred) + w3 * gram_loss(y_true, y_pred) + w4 * edge_loss(y_true, y_pred)

# incorporate GRAM matrix loss for texture preservation

def gram_matrix_3d(x):
    """
    Computes GRAM matrix for 3D feature maps.
    Input shape: (B, H, W, D, C)
    Output: (B, C, C)
    """
    # Flatten spatial dimensions: (B, H*W*D, C)
    x = tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[4]))

    # Compute Gram: (B, C, C)
    gram = tf.matmul(x, x, transpose_a=True)
    
    # Normalize by number of elements
    num_elements = tf.cast(tf.shape(x)[1] * tf.shape(x)[2], tf.float32)
    gram = gram / num_elements

    return gram

def gram_loss(y_true, y_pred):
    """
    Computes L1 loss between Gram matrices of true & predicted 3D images.
    """
    G_true = gram_matrix_3d(y_true)
    G_pred = gram_matrix_3d(y_pred)

    return tf.reduce_mean(tf.abs(G_true - G_pred))