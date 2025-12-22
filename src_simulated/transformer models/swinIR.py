import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------
# MLP Block
# ------------------------------
class Mlp3D(layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features)
        self.act = layers.Activation('gelu')
        self.fc2 = layers.Dense(out_features)
        self.drop = layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ------------------------------
# Window Partition & Reverse (Keras-compatible)
# ------------------------------
def window_partition_3d(x, window_size):
    # x: (B, H, W, D, C)
    ws_h, ws_w, ws_d = window_size
    def partition(x):
        B, H, W, D, C = tf.unstack(tf.shape(x))
        x = tf.reshape(x, (B,
                           H // ws_h, ws_h,
                           W // ws_w, ws_w,
                           D // ws_d, ws_d,
                           C))
        x = tf.transpose(x, (0, 1, 3, 5, 2, 4, 6, 7))
        x = tf.reshape(x, (-1, ws_h, ws_w, ws_d, C))
        return x
    return layers.Lambda(partition)(x)

def window_reverse_3d(windows, window_size, H, W, D):
    ws_h, ws_w, ws_d = window_size
    def reverse(x):
        B = tf.shape(x)[0] // (H // ws_h * W // ws_w * D // ws_d)
        x = tf.reshape(x, (B,
                           H // ws_h, W // ws_w, D // ws_d,
                           ws_h, ws_w, ws_d, -1))
        x = tf.transpose(x, (0, 1, 4, 2, 5, 3, 6, 7))
        x = tf.reshape(x, (B, H, W, D, -1))
        return x
    return layers.Lambda(reverse)(windows)

# ------------------------------
# Window Attention 3D
# ------------------------------
class WindowAttention3D(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.proj = layers.Dense(dim)

    def call(self, x):
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, self.head_dim))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, v)
        out = tf.transpose(out, (0, 2, 1, 3))
        out = tf.reshape(out, (B_, N, C))
        out = self.proj(out)
        return out

# ------------------------------
# Swin Transformer Layer 3D
# ------------------------------
class SwinTransformerLayer3D(layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=4, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention3D(dim, (window_size, window_size, window_size), num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = Mlp3D(dim, int(dim * mlp_ratio))

    def call(self, x):
        H, W, D = self.input_resolution
        B = tf.shape(x)[0]
        C = x.shape[-1]
        x_short = x
        x = self.norm1(x)
        x = layers.Reshape((H, W, D, C))(x)

        # cyclic shift
        if self.shift_size > 0:
            x = tf.roll(x, shift=[-self.shift_size]*3, axis=[1,2,3])

        # partition windows
        x_windows = window_partition_3d(x, (self.window_size, self.window_size, self.window_size))
        x_windows = layers.Reshape((-1, C))(x_windows)

        # attention
        x_windows = self.attn(x_windows)

        # merge windows
        x_windows = layers.Reshape((-1, self.window_size, self.window_size, self.window_size, C))(x_windows)
        x = window_reverse_3d(x_windows, (self.window_size, self.window_size, self.window_size), H, W, D)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size]*3, axis=[1,2,3])

        x = layers.Reshape((H*W*D, C))(x)
        x = x + x_short
        x = x + self.mlp(self.norm2(x))
        return x

# ------------------------------
# Residual Swin Transformer Block 3D (RSTB)
# ------------------------------
class RSTB3D(layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=4, mlp_ratio=4.):
        super().__init__()
        self.blocks = [
            SwinTransformerLayer3D(dim, input_resolution, num_heads, window_size, 0, mlp_ratio)
            for _ in range(depth)
        ]
        self.conv = layers.Conv3D(dim, 3, padding='same')

    def call(self, x):
        x_short = x
        for blk in self.blocks:
            x = blk(x)
        # convert to 3D spatial
        N = tf.shape(x)[1]
        C = x.shape[-1]
        D = W = H = tf.cast(tf.round(tf.pow(tf.cast(N, tf.float32), 1/3)), tf.int32)
        x_spatial = layers.Reshape((H, W, D, C))(x)
        x_spatial = self.conv(x_spatial)
        x = layers.Reshape((H*W*D, C))(x_spatial)
        return x + x_short

# ------------------------------
# SwinIR3D Model
# ------------------------------
def build_swinir3d(input_shape=(64,64,32,1), upscale=(1,1,1), embed_dim=48, depths=2, num_heads=4, num_rstb=2):
    inputs = keras.Input(shape=input_shape)
    H, W, D, C = input_shape
    x = layers.Conv3D(embed_dim, 3, padding='same')(inputs)
    x = layers.Reshape((H*W*D, embed_dim))(x)

    # deep feature extractor
    for _ in range(num_rstb):
        x = RSTB3D(embed_dim, input_resolution=(H,W,D), depth=depths, num_heads=num_heads)(x)

    # high-resolution reconstruction
    H_up, W_up, D_up = H*upscale[0], W*upscale[1], D*upscale[2]
    if upscale != (1,1,1):
        x_spatial = layers.Reshape((H, W, D, embed_dim))(x)
        x_spatial = layers.UpSampling3D(size=upscale)(x_spatial)
        x = layers.Reshape((H_up*W_up*D_up, embed_dim))(x_spatial)

    x_spatial = layers.Reshape((H_up, W_up, D_up, embed_dim))(x)
    x_out = layers.Conv3D(C, 3, padding='same')(x_spatial)

    model = keras.Model(inputs, x_out)
    return model

# ------------------------------
# Test
# ------------------------------
if __name__ == "__main__":
    model3d = build_swinir3d(input_shape=(64,64,32,1), upscale=(2,2,2))
    model3d.summary()