import tensorflow as tf
from tensorflow.keras import layers, Model

# -----------------------------
# Helpers: Inception (3D) and Double-Inception block
# -----------------------------
def conv3x3(filters, name=None):
    return layers.Conv3D(filters, kernel_size=3, padding="same",
                         kernel_initializer="he_normal", use_bias=False, name=name)

def conv1x1(filters, name=None):
    return layers.Conv3D(filters, kernel_size=1, padding="same",
                         kernel_initializer="he_normal", use_bias=False, name=name)

def inception3d_block(x, out_filters, bottleneck_ratio=0.5, use_bn=False, activation="relu", name=None):
    """
    3D Inception-like block with 4 branches:
      - 1x1
      - 1x1 -> 3x3
      - 1x1 -> 5x5 (implemented as two 3x3)
      - MaxPool3D -> 1x1
    out_filters should be divisible by 4 for balanced split (we still allow non-divisible).
    """
    act = layers.Activation(activation)
    bn = layers.BatchNormalization if use_bn else lambda : (lambda z: z)

    # compute branch filter sizes (try to split evenly)
    f1 = max(1, out_filters // 4)
    f2 = max(1, out_filters // 4)
    f3 = max(1, out_filters // 4)
    f4 = out_filters - (f1 + f2 + f3)
    # bottleneck intermediate filters
    b = max(1, int(bottleneck_ratio * (out_filters // 2)))

    # Branch 1: 1x1
    b1 = conv1x1(f1)(x)
    if use_bn: b1 = bn()(b1)
    b1 = act(b1)

    # Branch 2: 1x1 -> 3x3
    b2 = conv1x1(b)(x)
    if use_bn: b2 = bn()(b2)
    b2 = act(b2)
    b2 = conv3x3(f2)(b2)
    if use_bn: b2 = bn()(b2)
    b2 = act(b2)

    # Branch 3: 1x1 -> 3x3 -> 3x3 (approx 5x5 receptive field)
    b3 = conv1x1(b)(x)
    if use_bn: b3 = bn()(b3)
    b3 = act(b3)
    b3 = conv3x3(b3.shape[-1] if isinstance(b3.shape[-1], int) else b)(b3)  # keep intermediate
    if use_bn: b3 = bn()(b3)
    b3 = act(b3)
    b3 = conv3x3(f3)(b3)
    if use_bn: b3 = bn()(b3)
    b3 = act(b3)

    # Branch 4: pool -> 1x1
    b4 = layers.MaxPool3D(pool_size=3, padding="same", strides=1)(x)
    b4 = conv1x1(f4)(b4)
    if use_bn: b4 = bn()(b4)
    b4 = act(b4)

    # Concatenate
    out = layers.Concatenate(axis=-1, name=None if name is None else f"{name}_concat")([b1, b2, b3, b4])
    # final 1x1 to fuse and set out_filters
    out = conv1x1(out_filters, name=None if name is None else f"{name}_proj")(out)
    if use_bn: out = bn()(out)
    out = act(out)
    return out

def double_inception_block(x, filters, bottleneck_ratio=0.5, use_bn=False, activation="relu", name=None, dropout=0.0):
    """
    Two Inception3D modules stacked (Double-Inception), optional dropout between them.
    """
    y = inception3d_block(x, filters, bottleneck_ratio=bottleneck_ratio,
                          use_bn=use_bn, activation=activation,
                          name=None if name is None else f"{name}_inc1")
    if dropout and dropout > 0.0:
        y = layers.Dropout(dropout)(y)
    y = inception3d_block(y, filters, bottleneck_ratio=bottleneck_ratio,
                          use_bn=use_bn, activation=activation,
                          name=None if name is None else f"{name}_inc2")
    return y

# -----------------------------
# Inception-UNet builder
# -----------------------------
def build_inception_unet_3d(
    input_shape,
    out_channels=1,
    base_filters=16,
    levels=2,
    bottleneck_ratio=0.5,
    use_bn=True,
    activation="relu",
    dropout=0.0,
    final_activation="sigmoid",
    upsample_mode="transpose"  # or "nearest+conv"
):
    """
    Builds a 3D Inception-UNet with Double-Inception blocks.
    - input_shape: (D, H, W, C)
    - levels: 3, 4 or 5 (encoder/decoder depth)
    - base_filters: number of filters at first level; doubles each down step
    """
    assert levels in (2, 3, 4, 5), "levels must be 3, 4 or 5"
    inputs = layers.Input(shape=input_shape)

    # Encoder path
    skips = []
    x = inputs
    for L in range(levels):
        filters = base_filters * (2 ** L)
        x = double_inception_block(x, filters, bottleneck_ratio=bottleneck_ratio,
                                   use_bn=use_bn, activation=activation,
                                   name=f"enc_L{L}", dropout=dropout)
        skips.append(x)
        # downsample (don't downsample after last encoder level; we will downsample levels-1 times)
        x = layers.MaxPool3D(pool_size=2, padding="same", name=f"enc_pool_L{L}")(x)

    # Bottleneck
    bottleneck_filters = base_filters * (2 ** levels)
    x = double_inception_block(x, bottleneck_filters, bottleneck_ratio=bottleneck_ratio,
                               use_bn=use_bn, activation=activation,
                               name="bottleneck", dropout=dropout)

    for L in reversed(range(levels)):
        filters = base_filters * (2 ** L)
        # upsample
        if upsample_mode == "transpose":
            x = layers.Conv3DTranspose(filters, kernel_size=2, strides=2, padding="same",
                                        kernel_initializer="he_normal", use_bias=False,
                                        name=f"dec_up_L{L}")(x)
            if use_bn: x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
        else:
            x = layers.UpSampling3D(size=2)(x)
            x = conv3x3(filters)(x)
            if use_bn: x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)

        # ✅ safer concat (no Lambda)
        x = layers.Concatenate(axis=-1, name=f"concat_L{L}")([x, skips[L]])

        # double inception block
        x = double_inception_block(x, filters, bottleneck_ratio=bottleneck_ratio,
                                    use_bn=use_bn, activation=activation,
                                    name=f"dec_L{L}", dropout=dropout)

    # Final conv
    outputs = layers.Conv3D(out_channels, kernel_size=1, padding="same",
                            activation=final_activation, name="final_conv")(x)

    model = Model(inputs, outputs, name=f"InceptionUNet3D_levels{levels}")
    return model

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example input shape: (depth, height, width, channels)
    in_shape = (32, 128, 128, 1)   # adjust to your volumes (D,H,W,C)
    model = build_inception_unet_3d(
        input_shape=in_shape,
        out_channels=1,
        base_filters=16,
        levels=2,               # choose 3, 4 or 5
        bottleneck_ratio=0.5,
        use_bn=True,
        activation="relu",
        dropout=0.1,
        final_activation="sigmoid",
        upsample_mode="transpose"
    )
    model.summary()
