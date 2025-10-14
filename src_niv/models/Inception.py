import tensorflow as tf
from tensorflow.keras import layers, Model

# -----------------------------
# 3D Convolutions helpers
# -----------------------------
def conv3x3(filters, name=None):
    return layers.Conv3D(filters, kernel_size=3, padding="same",
                         kernel_initializer="he_normal", use_bias=False, name=name)

def conv1x1(filters, name=None):
    return layers.Conv3D(filters, kernel_size=1, padding="same",
                         kernel_initializer="he_normal", use_bias=False, name=name)

def conv5x5(filters, name=None):
    # Approximate 5x5 with two 3x3 convs
    return tf.keras.Sequential([
        layers.Conv3D(filters, 3, padding="same", kernel_initializer="he_normal", use_bias=False),
        layers.Conv3D(filters, 3, padding="same", kernel_initializer="he_normal", use_bias=False)
    ], name=name)

# -----------------------------
# Residual Inception3D block
# -----------------------------
def inception3d_residual(x, out_filters, activation="relu", name=None):
    act = layers.Activation(activation)

    # split filters among 3 branches
    f1 = out_filters // 3
    f2 = out_filters // 3
    f3 = out_filters - f1 - f2

    # Branch1: 1x1
    b1 = conv1x1(f1)(x)
    b1 = act(b1)

    # Branch2: 3x3
    b2 = conv3x3(f2)(x)
    b2 = act(b2)

    # Branch3: 5x5
    b3 = conv5x5(f3)(x)
    b3 = act(b3)

    # Concatenate
    out = layers.Concatenate(axis=-1, name=f"{name}_concat" if name else None)([b1, b2, b3])

    # Residual connection: match channels if needed
    if x.shape[-1] != out.shape[-1]:
        x_res = conv1x1(out.shape[-1])(x)
    else:
        x_res = x

    out = layers.Add()([out, x_res])
    out = act(out)
    return out

# -----------------------------
# Double Residual-Inception block
# -----------------------------
def double_inception_res_block(x, filters, activation="relu", name=None, dropout=0.0):
    y = inception3d_residual(x, filters, activation=activation, name=None if name is None else f"{name}_inc1")
    if dropout > 0.0:
        y = layers.Dropout(dropout)(y)
    y = inception3d_residual(y, filters, activation=activation, name=None if name is None else f"{name}_inc2")
    return y

# -----------------------------
# Residual Inception UNet 3D
# -----------------------------
def build_res_inception_unet_3d(
    input_shape=(64, 64, 40, 1),
    out_channels=1,
    base_filters=16,
    levels=2,
    activation="relu",
    dropout=0.0,
    final_activation="linear",
    upsample_mode="transpose"
):
    inputs = layers.Input(shape=input_shape)
    skips = []
    x = inputs

    # Encoder
    for L in range(levels):
        filters = base_filters * (2 ** L)
        x = double_inception_res_block(x, filters, activation=activation, name=f"enc_L{L}", dropout=dropout)
        skips.append(x)
        x = layers.MaxPool3D(pool_size=2, padding="same")(x)

    # Bottleneck
    filters = base_filters * (2 ** levels)
    x = double_inception_res_block(x, filters, activation=activation, name="bottleneck", dropout=dropout)

    # Decoder
    for L in reversed(range(levels)):
        filters = base_filters * (2 ** L)
        # Upsample
        if upsample_mode == "transpose":
            x = layers.Conv3DTranspose(filters, kernel_size=2, strides=2, padding="same",
                                       kernel_initializer="he_normal", use_bias=False)(x)
        else:
            x = layers.UpSampling3D(size=2)(x)
            x = conv3x3(filters)(x)
        x = layers.Activation(activation)(x)

        # Concatenate skip
        x = layers.Concatenate(axis=-1)([x, skips[L]])
        x = double_inception_res_block(x, filters, activation=activation, dropout=dropout)

    # Output
    outputs = layers.Conv3D(out_channels, kernel_size=1, padding="same", activation=final_activation)(x)
    model = Model(inputs, outputs, name="ResInceptionUNet3D")
    return model

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    model = build_res_inception_unet_3d(
        input_shape=(64, 64, 40, 1),
        out_channels=1,
        base_filters=16,
        levels=2,
        activation="relu",
        dropout=0.1,
        final_activation="sigmoid",
        upsample_mode="transpose"
    )
    
    model.summary()