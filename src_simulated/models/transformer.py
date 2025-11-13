import tensorflow as tf
from tensorflow.keras import layers, Model, Input

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.drop1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        y = self.norm1(x)
        attn = self.mha(y, y, training=training)
        attn = self.drop1(attn, training=training)
        x = x + attn
        y = self.norm2(x)
        x = x + self.mlp(y, training=training)
        return x

def build_3d_transformer_srr(
    input_shape=(64,64,16,1),
    patch_size=(4,4,2),
    embed_dim=64,
    depth=4,
    num_heads=8,
    mlp_dim=512,
    dropout=0.0,
    upsample_scale=(1,1,1),
    output_channels=1
):
    H, W, D, C = input_shape
    pH, pW, pD = patch_size

    assert H % pH == 0 and W % pW == 0 and D % pD == 0, "Input dims must be divisible by patch dims."

    n_hp = H // pH
    n_wp = W // pW
    n_dp = D // pD
    n_tokens = n_hp * n_wp * n_dp

    inp = Input(shape=input_shape)

    # Patch embedding
    x = layers.Conv3D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid", name="patch_embed")(inp)
    # x shape: (batch, n_hp, n_wp, n_dp, embed_dim)

    # flatten patches into tokens
    x = layers.Reshape((n_tokens, embed_dim))(x)  # (batch, n_tokens, embed_dim)

    # positional embeddings
    pos_emb_layer = layers.Embedding(input_dim=n_tokens, output_dim=embed_dim, name="pos_emb")
    # create positions tensor (this is a constant int sequence; ok to use here)
    positions = tf.range(start=0, limit=n_tokens, delta=1)
    p = pos_emb_layer(positions)                # (n_tokens, embed_dim)
    p = layers.Lambda(lambda z: tf.expand_dims(z, 0), name="pos_expand")(p)  # -> (1, n_tokens, embed_dim)
    x = layers.Add(name="add_pos")([x, p])      # broadcast add over batch

    # Transformer encoder stack
    for i in range(depth):
        x = TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, name=f"transformer_block_{i}")(x)

    # Reshape tokens back to patch-grid
    x = layers.Reshape((n_hp, n_wp, n_dp, embed_dim))(x)

    # Decoder: revert patch embedding
    x = layers.Conv3DTranspose(filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid", name="unpatch")(x)

    x = layers.Conv3D(filters=embed_dim//2, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv3D(filters=embed_dim//4, kernel_size=3, padding="same", activation="relu")(x)

    x = layers.Conv3D(filters=C, kernel_size=1, padding="same", name="to_input_channels")(x)
    low_res_skip = x

    suH, suW, suD = upsample_scale
    if (suH, suW, suD) != (1,1,1):
        x = layers.UpSampling3D(size=upsample_scale, name="upsample")(x)
        x = layers.Conv3D(filters=embed_dim//4, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.Conv3D(filters=embed_dim//8, kernel_size=3, padding="same", activation="relu")(x)

    out = layers.Conv3D(filters=output_channels, kernel_size=3, padding="same", name="final_conv")(x)

    if (suH, suW, suD) != (1,1,1):
        inp_upsampled = layers.UpSampling3D(size=upsample_scale, name="input_upsample")(inp)
        out = layers.Add(name="final_residual")([out, inp_upsampled])
    else:
        out = layers.Add(name="final_residual")([out, inp])

    model = Model(inp, out, name="3D_Patch_Transformer_SRR")
    return model

# Build and test
model = build_3d_transformer_srr(
    input_shape=(64,64,32,1),
    patch_size=(4,4,2),
    embed_dim=256,
    depth=6,
    num_heads=8,
    mlp_dim=512,
    dropout=0.0,
    upsample_scale=(2,2,2),
    output_channels=1
)

model.summary()