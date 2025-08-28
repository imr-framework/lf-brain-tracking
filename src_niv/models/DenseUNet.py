import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------------
# Dense Block (no BatchNorm)
# -------------------------------
def dense_block(x, filters, activation="relu", name=None):
    """Simple dense block with two Conv3D layers, concatenating features."""
    conv1 = layers.Conv3D(filters, 3, padding="same", activation=activation, name=f"{name}_conv1")(x)
    concat1 = layers.Concatenate(name=f"{name}_concat1")([x, conv1])
    conv2 = layers.Conv3D(filters, 3, padding="same", activation=activation, name=f"{name}_conv2")(concat1)
    concat2 = layers.Concatenate(name=f"{name}_concat2")([concat1, conv2])
    return concat2

# -------------------------------
# Dense U-Net 3D
# -------------------------------
def build_dense_unet_3d(
    input_shape,
    out_channels=1,                 # number of segmentation labels
    base_filters=16,
    levels=3,
    activation="relu",
    final_activation="sigmoid",
    dense_blocks_per_level=2        # choice: 1 or 2
):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    skips = []

    # -------- Encoder --------
    for L in range(levels):
        filters = base_filters * (2 ** L)
        for b in range(dense_blocks_per_level):
            x = dense_block(x, filters, activation=activation, name=f"enc{L}_block{b+1}")
        skips.append(x)
        x = layers.MaxPool3D(pool_size=2, name=f"pool{L}")(x)

    # -------- Bottleneck --------
    filters = base_filters * (2 ** levels)
    for b in range(dense_blocks_per_level):
        x = dense_block(x, filters, activation=activation, name=f"bottleneck_block{b+1}")

    # -------- Decoder --------
    for L in reversed(range(levels)):
        filters = base_filters * (2 ** L)
        x = layers.Conv3DTranspose(filters, 2, strides=2, padding="same", name=f"up{L}")(x)
        x = layers.Concatenate(name=f"dec{L}_concat")([x, skips[L]])
        for b in range(dense_blocks_per_level):
            x = dense_block(x, filters, activation=activation, name=f"dec{L}_block{b+1}")

    # -------- Output --------
    outputs = layers.Conv3D(out_channels, 1, padding="same", activation=final_activation, name="output")(x)

    return models.Model(inputs, outputs, name="DenseUNet3D")


# -------------------------------
# Example Run
# -------------------------------
if __name__ == "__main__":
    in_shape = (32, 128, 128, 1)   # (D, H, W, C)
    model = build_dense_unet_3d(
        input_shape=in_shape,
        out_channels=1,             # 3 or 4 labels
        base_filters=16,
        levels=3,
        activation="relu",
        final_activation="sigmoid",
        dense_blocks_per_level=2    # 👈 choice: 1 or 2
    )
    model.summary()
