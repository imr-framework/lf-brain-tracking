import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib
matplotlib.use("Agg")  # for headless environments

# ---------------------
# Residual Block (BN-free)
# ---------------------
def residual_block_3d(x, filters, kernel_size=3, res_scale=0.1, name=None):
    """
    Single 3D residual block (no BatchNorm).
    Structure: Conv3D -> ReLU -> Conv3D -> scale -> add(skip) -> ReLU
    """
    shortcut = x

    # First conv + ReLU
    y = layers.Conv3D(filters, kernel_size, padding='same', use_bias=True,
                      kernel_initializer='he_normal', name=(f"{name}_conv1" if name else None))(x)
    y = layers.Activation('relu', name=(f"{name}_act1" if name else None))(y)

    # Second conv
    y = layers.Conv3D(filters, kernel_size, padding='same', use_bias=True,
                      kernel_initializer='he_normal', name=(f"{name}_conv2" if name else None))(y)

    # Project shortcut if channel mismatch
    in_ch = shortcut.shape[-1]
    if in_ch is None or in_ch != filters:
        shortcut = layers.Conv3D(filters, kernel_size=1, padding='same', use_bias=True,
                                 kernel_initializer='he_normal', name=(f"{name}_proj" if name else None))(shortcut)

    # Residual scaling
    if res_scale != 1.0:
        y = layers.Lambda(lambda z: z * res_scale,
                          name=(f"{name}_scale" if name else None))(y)

    out = layers.Add(name=(f"{name}_add" if name else None))([shortcut, y])
    out = layers.Activation('relu', name=(f"{name}_out_act" if name else None))(out)
    
    return out


# ---------------------
# Full 3D Residual SRR Model
# ---------------------
def build_3d_residual(input_shape=(32, 128, 128, 1),
                      num_filters=64,
                      num_blocks=5,
                      kernel_size=3,
                      res_scale=0.1,
                      learn_residual=True,
                      name='3D_Residual_NoBN'):
    """
    BN-free 3D residual network for SRR
    """
    inp = layers.Input(shape=input_shape, name='input_volume')

    # Stem convolution
    x = layers.Conv3D(num_filters, kernel_size=3, padding='same', activation='relu',
                      kernel_initializer='he_normal', name='stem_conv')(inp)

    # Residual blocks
    for i in range(num_blocks):
        x = residual_block_3d(x, filters=num_filters, kernel_size=kernel_size,
                              res_scale=res_scale, name=f'res_block_{i+1}')

    # Refinement convolutions
    x = layers.Conv3D(num_filters, kernel_size=3, padding='same', activation='relu',
                      kernel_initializer='he_normal', name='refine_conv1')(x)
    x = layers.Conv3D(num_filters // 2, kernel_size=3, padding='same', activation='relu',
                      kernel_initializer='he_normal', name='refine_conv2')(x)

    # Output projection (linear, no activation)
    out_res = layers.Conv3D(1, kernel_size=1, padding='same', activation=None, name='reconstruct')(x)

    # Residual learning (optional)
    if learn_residual:
        inp_proj = layers.Conv3D(1, kernel_size=1, padding='same', activation=None, name='input_proj')(inp)
        out = layers.Add(name='residual_add')([inp_proj, out_res])
    else:
        out = out_res

    model = Model(inputs=inp, outputs=out, name=name)
    return model


# ---------------------
# Example usage / test
# ---------------------
if __name__ == "__main__":
    tf.keras.backend.clear_session()

    model = build_3d_residual(
        input_shape=(32, 128, 128, 1),
        num_filters=64,
        num_blocks=5,
        kernel_size=3,
        res_scale=0.1,
        learn_residual=True
    )

    # Print model summary
    import sys
    model.summary(print_fn=lambda x: sys.stdout.write(x + "\n"))

    # Compile model (example)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='mse',
                  metrics=['mae'])

    # Quick forward pass
    import numpy as np
    x = np.random.randn(1, 32, 128, 128, 1).astype(np.float32)
    y = model.predict(x, batch_size=1)
    print("Output shape:", y.shape)
