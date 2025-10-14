import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# ---------------------
# Helpers: window ops
# ---------------------
def window_partition(x, window_size):
    """
    x: (B, D, H, W, C)
    window_size: tuple (wd, wh, ww)
    returns windows: (num_windows*B, wd*wh*ww, C)
    """
    wd, wh, ww = window_size
    B, D, H, W, C = tf.unstack(tf.shape(x))
    # pad if necessary (should be divisible)
    x_reshaped = tf.reshape(x, (B,
                                D // wd, wd,
                                H // wh, wh,
                                W // ww, ww,
                                C))
    # permute to bring windows together: (B, nd, nh, nw, wd, wh, ww, C)
    x_permuted = tf.transpose(x_reshaped, perm=[0,1,3,5,2,4,6,7])
    windows = tf.reshape(x_permuted, (-1, wd*wh*ww, C))
    return windows

def window_reverse(windows, window_size, D, H, W):
    """
    windows: (num_windows*B, wd*wh*ww, C)
    returns x: (B, D, H, W, C)
    """
    wd, wh, ww = window_size
    C = tf.shape(windows)[-1]
    B = tf.cast(tf.shape(windows)[0] // ((D//wd)*(H//wh)*(W//ww)), tf.int32)
    x = tf.reshape(windows, (B, D//wd, H//wh, W//ww, wd, wh, ww, C))
    x = tf.transpose(x, perm=[0,1,4,2,5,3,6,7])  # (B, nd, wd, nh, wh, nw, ww, C)
    x = tf.reshape(x, (B, D, H, W, C))
    return x

# ---------------------
# Attention mask for shifted windows (Swin style)
# ---------------------
def compute_attention_mask(D, H, W, window_size, shift_size):
    """
    Create attention mask for 3D shifted windows.
    D,H,W are divisible by window dims.
    window_size and shift_size are tuples.
    Returns mask of shape (num_windows, wd*wh*ww, wd*wh*ww) to add large negative to invalid positions.
    """
    wd, wh, ww = window_size
    sd, sh, sw = shift_size

    # create a 3D label map
    img_mask = np.zeros((1, D, H, W, 1), dtype=np.int32)
    cnt = 0
    d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, None)) if sd>0 else (slice(0, None),)
    # Simpler systematic labeling: split into blocks of wd/wh/ww along each axis with overlap shift
    d_splits = np.arange(0, D, wd)
    h_splits = np.arange(0, H, wh)
    w_splits = np.arange(0, W, ww)
    for i, ds in enumerate(d_splits):
        for j, hs in enumerate(h_splits):
            for k, ws in enumerate(w_splits):
                d0, d1 = ds, min(ds+wd, D)
                h0, h1 = hs, min(hs+wh, H)
                w0, w1 = ws, min(ws+ww, W)
                img_mask[0, d0:d1, h0:h1, w0:w1, 0] = cnt
                cnt += 1

    # shift mask for attention if shifted
    if any([sd, sh, sw]):
        # roll for shift and compute partition
        shifted = np.roll(img_mask, shift=(-sd, -sh, -sw), axis=(1,2,3))
    else:
        shifted = img_mask

    # partition into windows
    nd = D // wd
    nh = H // wh
    nw = W // ww
    num_windows = nd * nh * nw
    mask_windows = np.reshape(shifted, (1, nd, wd, nh, wh, nw, ww, 1))
    mask_windows = np.transpose(mask_windows, (0,1,3,5,2,4,6,7))
    mask_windows = np.reshape(mask_windows, (-1, wd*wh*ww))
    # mask windows shape: (num_windows, wd*wh*ww)
    attn_mask = np.expand_dims(mask_windows, 1) - np.expand_dims(mask_windows, 2)
    # positions with nonzero difference => set to -inf later
    attn_mask = np.where(attn_mask!=0, -100.0, 0.0).astype(np.float32)  # float mask to add to logits
    return tf.constant(attn_mask)  # (num_windows, ws*ws*ws, ws*ws*ws)

# ---------------------
# MLP block
# ---------------------
def mlp_block(x, hidden_dim, dropout_rate=0.0):
    x = layers.Dense(hidden_dim, activation='gelu')(x)
    if dropout_rate>0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(tf.shape(x)[-1])(x)
    if dropout_rate>0:
        x = layers.Dropout(dropout_rate)(x)
    return x

# ---------------------
# Swin Transformer Block 3D (windowed + shifted)
# ---------------------
class SwinTransformerBlock3D(layers.Layer):
    def __init__(self, dim, window_size=(2,8,8), num_heads=4, shift_size=(0,0,0), mlp_ratio=4.0, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        # use a Dense projection to match MultiHeadAttention expected last dim
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim//num_heads, dropout=dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp_hidden = int(dim * mlp_ratio)

    def call(self, x, attn_mask=None, training=None):
        """
        x: (B, D, H, W, C)
        attn_mask: (num_windows, ws3, ws3) or None
        """
        B = tf.shape(x)[0]
        D = tf.shape(x)[1]
        H = tf.shape(x)[2]
        W = tf.shape(x)[3]
        C = x.shape[-1]

        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size

        shortcut = x
        x = self.norm1(x)

        # cyclic shift if needed
        if any([sd, sh, sw]):
            x = tf.roll(x, shift=[-sd, -sh, -sw], axis=[1,2,3])

        # partition windows
        x_windows = window_partition(x, self.window_size)  # (num_windows*B, ws3, C)
        seq_len = tf.shape(x_windows)[1]

        # apply attention per window
        # MultiHeadAttention expects (batch, seq_len, dim)
        if attn_mask is not None:
            # attn_mask shape: (num_windows, seq_len, seq_len)
            # we need to tile it for batch: (num_windows*B, seq_len, seq_len)
            # However, num_windows here equals total windows per sample. We only have combined windows (num_windows*B).
            num_total_windows = tf.shape(x_windows)[0]
            # attn_mask_batch = tf.tile(attn_mask, [B,1,1])  # simpler but careful with shapes
            # To be safe: compute windows per sample:
            win_per_sample = (D//wd)*(H//wh)*(W//ww)
            attn_mask_tiled = tf.tile(attn_mask, [B,1,1])
            # If shapes mismatch due to dynamic dims, rely on broadcasting in attention with "attention_mask" below.
            attention_mask = attn_mask_tiled
        else:
            attention_mask = None

        # apply attention
        x_attn = self.attn(x_windows, x_windows, attention_mask=attention_mask)  # (num_total_windows, seq_len, C)
        x = x_windows + x_attn

        # merge windows back
        x = window_reverse(x, self.window_size, D, H, W)  # (B, D, H, W, C)

        # reverse cyclic shift
        if any([sd, sh, sw]):
            x = tf.roll(x, shift=[sd, sh, sw], axis=[1,2,3])

        # FFN / MLP
        x = shortcut + x
        y = self.norm2(x)
        y = mlp_block(y, self.mlp_hidden, dropout_rate=self.dropout)
        out = x + y
        return out

# ---------------------
# Stage: several Swin blocks with alternating shifts
# ---------------------
def swin_stage(x, dim, depth, window_size, num_heads, mlp_ratio=4.0, dropout=0.0):
    """
    x: (B, D, H, W, C)
    applies `depth` SwinTransformerBlock3D with alternating shift sizes
    """
    sd = window_size[0] // 2
    sh = window_size[1] // 2
    sw = window_size[2] // 2
    for i in range(depth):
        if i % 2 == 0:
            shift = (0,0,0)
        else:
            shift = (sd, sh, sw)
        x = SwinTransformerBlock3D(dim=dim, window_size=window_size, num_heads=num_heads,
                                   shift_size=shift, mlp_ratio=mlp_ratio, dropout=dropout)(x)
    return x

# ---------------------
# Full model builder
# ---------------------
def build_3d_swin_srr(input_shape=(32,128,128,1),
                      patch_size=(2,4,4),
                      embed_dim=64,
                      window_size=(2,8,8),    # in pixels AFTER patch embedding
                      depths=2,                # number of blocks in the single stage
                      num_heads=4,
                      mlp_ratio=4.0,
                      dropout=0.0,
                      learn_residual=True):
    """
    Build a 3D Swin-style SRR model.
    - input_shape: (D,H,W,C)
    - patch_size: patching via Conv3D stride (pd,ph,pw)
    - embed_dim: channels after patch embedding
    - window_size: window size in pixels *after* patch embedding factor
    """
    D, H, W, C = input_shape
    pd, ph, pw = patch_size

    assert D % pd == 0 and H % ph == 0 and W % pw == 0, "patch_size must divide input dims"

    # Patch embed
    inp = layers.Input(shape=input_shape, name='input_volume')
    x = layers.Conv3D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid', name='patch_embed')(inp)
    # x shape: (B, Dp, Hp, Wp, embed_dim)
    Dp = D // pd
    Hp = H // ph
    Wp = W // pw

    # Optional small conv to mix channels
    x = layers.Conv3D(embed_dim, 3, padding='same', activation='relu', name='conv_mix1')(x)

    # Confirm that window_size divides the post-patch dims
    ws_pd = window_size[0] // pd if window_size[0] % pd == 0 else None
    # But easier: define window_size in the embedded-grid coordinates (wd,wh,ww)
    wd = max(1, window_size[0] // pd)
    wh = max(1, window_size[1] // ph)
    ww = max(1, window_size[2] // pw)
    window_size_grid = (wd, wh, ww)

    # If embed grid dims are divisible by window grid dims
    assert Dp % wd == 0 and Hp % wh == 0 and Wp % ww == 0, f"embedded grid must be divisible by window grid. Got Dp={Dp}, window wd={wd}."

    # Apply a small conv to get to working channels
    x = layers.Conv3D(embed_dim, 1, padding='same', activation=None, name='to_work_ch')(x)

    # Swin Stage(s)
    # For simplicity we use single stage composed of `depths` Swin blocks operating on (Dp,Hp,Wp,embed_dim)
    # But our window_partition functions expect input in real-pixel coords, so we will treat window_size_grid as pixel-wise
    # To keep consistency, we will upsample grid dims to "pseudo-pixels" by factor 1 (i.e. call window ops using window_size_grid)
    # Convert to float32 to avoid dtype issues in layers
    x = tf.cast(x, tf.float32)

    # Apply swin stage
    for i in range(depths):
        # alternate shift inside SwinTransformerBlock3D is handled there
        x = SwinTransformerBlock3D(dim=embed_dim,
                                   window_size=window_size_grid,
                                   num_heads=num_heads,
                                   shift_size=(0,0,0) if i%2==0 else (window_size_grid[0]//2, window_size_grid[1]//2, window_size_grid[2]//2),
                                   mlp_ratio=mlp_ratio,
                                   dropout=dropout,
                                   name=f'swin_block_{i}')(x)

    # Final convs to reconstruct
    x = layers.Conv3D(embed_dim, 3, padding='same', activation='relu', name='refine1')(x)
    x = layers.Conv3D(embed_dim//2, 3, padding='same', activation='relu', name='refine2')(x)

    # Upsample back to original resolution by reversing patch embedding (UpSampling by patch_size)
    x = layers.UpSampling3D(size=patch_size, name='upsample')(x)
    out_res = layers.Conv3D(1, 1, padding='same', activation=None, name='reconstruct')(x)

    if learn_residual:
        inp_proj = layers.Conv3D(1, 1, padding='same', activation=None, name='input_proj')(inp)
        out = layers.Add(name='residual_add')([inp_proj, out_res])
    else:
        out = out_res

    model = Model(inputs=inp, outputs=out, name='3D_Swin_SRR')
    return model

# ---------------------
# Example usage
# ---------------------
if __name__ == "__main__":
    tf.keras.backend.clear_session()
    model = build_3d_swin_srr(input_shape=(32,128,128,1),
                              patch_size=(2,4,4),
                              embed_dim=64,
                              window_size=(4,32,32),  # window in original pixels; patch_size=(2,4,4) => window grid=(2,8,8)
                              depths=2,
                              num_heads=4,
                              mlp_ratio=4.0,
                              dropout=0.0,
                              learn_residual=True)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
    # Test forward pass
    x = np.random.randn(1,32,128,128,1).astype(np.float32)
    y = model.predict(x, batch_size=1)
    print("Output shape:", y.shape)
