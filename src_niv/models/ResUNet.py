import tensorflow as tf
from tensorflow.keras import layers, models

def res_conv_block(x, filters, kernel_size=3):
    """Residual convolutional block"""
    shortcut = layers.Conv3D(filters, kernel_size=1, padding="same")(x)

    x = layers.Conv3D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.Conv3D(filters, kernel_size, padding="same")(x)

    x = layers.Add()([x, shortcut])  # residual connection
    x = layers.Activation("relu")(x)
    return x


def residual_srr_unet(input_shape=(32, 128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    # ---------------- Encoder ----------------
    c1 = res_conv_block(inputs, 32)  # Level 1
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)   # -> (16, 64, 64)

    c2 = res_conv_block(p1, 64)      # Level 2
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)   # -> (8, 32, 32)

    c3 = res_conv_block(p2, 128)     # Level 3
    p3 = layers.MaxPooling3D((2, 2, 2))(c3)   # -> (4, 16, 16)

    # ---------------- Bottleneck ----------------
    bn = res_conv_block(p3, 256)

    # ---------------- Decoder ----------------
    u3 = layers.UpSampling3D((2, 2, 2))(bn)   # -> (8, 32, 32)
    u3 = layers.Concatenate()([u3, c3])
    c4 = res_conv_block(u3, 128)

    u2 = layers.UpSampling3D((2, 2, 2))(c4)   # -> (16, 64, 64)
    u2 = layers.Concatenate()([u2, c2])
    c5 = res_conv_block(u2, 64)

    u1 = layers.UpSampling3D((2, 2, 2))(c5)   # -> (32, 128, 128)
    u1 = layers.Concatenate()([u1, c1])
    c6 = res_conv_block(u1, 32)

    # ---------------- Output ----------------
    residual = layers.Conv3D(1, (1, 1, 1), activation="linear")(c6)

    # Add residual correction to input
    outputs = layers.Add()([inputs, residual])  

    model = models.Model(inputs, outputs, name="Residual_SRR_UNet3D")
    return model

# Example usage
model = residual_srr_unet(input_shape=(32, 128, 128, 1))
model.summary()

import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K

# -------------------------
# Residual conv block
# -------------------------
def res_conv_block(x, filters, kernel_size=3):
    shortcut = layers.Conv3D(filters, 1, padding="same")(x)
    x = layers.Conv3D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.Conv3D(filters, kernel_size, padding="same")(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

# -------------------------
# Attention gate (3D)
# - aligns g (gating signal) to x (skip) spatially using UpSampling3D if needed
# -------------------------
def attention_gate_3d(x, g, inter_channels):
    """
    x : skip connection tensor (encoder feature)
    g : gating tensor (decoder / deeper feature)
    inter_channels : number of filters in intermediate projections
    """
    # 1x1 projections
    theta_x = layers.Conv3D(inter_channels, 1, strides=1, padding="same")(x)  # (D_x,H_x,W_x,inter)
    phi_g = layers.Conv3D(inter_channels, 1, strides=1, padding="same")(g)     # (D_g,H_g,W_g,inter)

    # Get static shapes
    sx = K.int_shape(theta_x)  # (batch, D_x, H_x, W_x, C)
    sg = K.int_shape(phi_g)    # (batch, D_g, H_g, W_g, C)

    # If gating is smaller spatially, upsample it to match theta_x
    if None not in (sx[1], sg[1]) and (sx[1] != sg[1] or sx[2] != sg[2] or sx[3] != sg[3]):
        # compute integer upsampling factors (fallback to 2 if mismatch not divisible)
        def _factor(a, b):
            return a // b if (a is not None and b is not None and a % b == 0) else None

        dz = _factor(sx[1], sg[1])
        dy = _factor(sx[2], sg[2])
        dx = _factor(sx[3], sg[3])

        if dz and dy and dx:
            phi_g = layers.UpSampling3D(size=(dz, dy, dx))(phi_g)
        else:
            # fallback: upsample by 2 until sizes roughly match (safe default)
            phi_g = layers.UpSampling3D(size=(sx[1] // max(1, sg[1] or 1),
                                              sx[2] // max(1, sg[2] or 1),
                                              sx[3] // max(1, sg[3] or 1)))(phi_g)

    # combine
    add_xg = layers.Add()([theta_x, phi_g])
    act = layers.Activation("relu")(add_xg)
    psi = layers.Conv3D(1, 1, padding="same")(act)
    psi = layers.Activation("sigmoid")(psi)

    # scale skip connection
    out = layers.Multiply()([x, psi])
    return out

# -------------------------
# Build 3-level Residual Attention UNet (3D)
# -------------------------
def residual_srr_att_dsunet(input_shape=(32,128,128,1), base_filters=32):
    """
    input_shape: (Z, X, Y, C)
    base_filters: number of filters at first level (doubles each down)
    """
    inp = layers.Input(shape=input_shape)

    # Encoder level 1
    c1 = res_conv_block(inp, base_filters)        # (Z, X, Y, base)
    p1 = layers.MaxPool3D(pool_size=(2,2,2))(c1)  # down

    # Encoder level 2
    c2 = res_conv_block(p1, base_filters*2)
    p2 = layers.MaxPool3D(pool_size=(2,2,2))(c2)

    # Encoder level 3
    c3 = res_conv_block(p2, base_filters*4)
    p3 = layers.MaxPool3D(pool_size=(2,2,2))(c3)

    # Bottleneck
    bn = res_conv_block(p3, base_filters*8)

    # Decoder level 3 (upsample bottleneck -> match c3)
    g3 = layers.Conv3D(base_filters*4, 1, padding="same")(bn)   # gating
    att3 = attention_gate_3d(c3, g3, inter_channels=base_filters*2)
    u3 = layers.UpSampling3D(size=(2,2,2))(bn)
    u3 = layers.Concatenate()([u3, att3])
    c4 = res_conv_block(u3, base_filters*4)

    # Decoder level 2
    g2 = layers.Conv3D(base_filters*2, 1, padding="same")(c4)
    att2 = attention_gate_3d(c2, g2, inter_channels=base_filters)
    u2 = layers.UpSampling3D(size=(2,2,2))(c4)
    u2 = layers.Concatenate()([u2, att2])
    c5 = res_conv_block(u2, base_filters*2)

    # Decoder level 1
    g1 = layers.Conv3D(base_filters, 1, padding="same")(c5)
    att1 = attention_gate_3d(c1, g1, inter_channels=max(1, base_filters//2))
    u1 = layers.UpSampling3D(size=(2,2,2))(c5)
    u1 = layers.Concatenate()([u1, att1])
    c6 = res_conv_block(u1, base_filters)

    # Deep supervision optional: produce auxiliary preds at intermediate resolutions
    dsv3 = layers.Conv3D(1, 1, activation="linear", padding="same")(c4)
    dsv3_up = layers.UpSampling3D(size=(4,4,4))(dsv3)   # bring to input resolution

    dsv2 = layers.Conv3D(1, 1, activation="linear", padding="same")(c5)
    dsv2_up = layers.UpSampling3D(size=(2,2,2))(dsv2)

    dsv1 = layers.Conv3D(1, 1, activation="linear", padding="same")(c6)

    # fuse deep supervision outputs
    fuse = layers.Add()([dsv1, dsv2_up, dsv3_up])

    # residual prediction and final add
    residual = layers.Conv3D(1, (1,1,1), padding="same", activation="linear")(fuse)
    out = layers.Add()([inp, residual])

    model = Model(inputs=inp, outputs=out, name="ResAttUNet3D_3level")
    return model

import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K

# -------------------------
# Residual conv block
# -------------------------
def res_conv_block(x, filters, kernel_size=3):
    shortcut = layers.Conv3D(filters, 1, padding="same")(x)
    x = layers.Conv3D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.Conv3D(filters, kernel_size, padding="same")(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

# -------------------------
# Attention gate (3D)
# -------------------------
def attention_gate_3d(x, g, inter_channels):
    theta_x = layers.Conv3D(inter_channels, 1, strides=1, padding="same")(x)
    phi_g = layers.Conv3D(inter_channels, 1, strides=1, padding="same")(g)

    sx = K.int_shape(theta_x)
    sg = K.int_shape(phi_g)

    # Upsample gating if needed
    if None not in (sx[1], sg[1]) and (sx[1] != sg[1] or sx[2] != sg[2] or sx[3] != sg[3]):
        dz = sx[1] // max(1, sg[1] or 1)
        dy = sx[2] // max(1, sg[2] or 1)
        dx = sx[3] // max(1, sg[3] or 1)
        phi_g = layers.UpSampling3D(size=(dz, dy, dx))(phi_g)

    add_xg = layers.Add()([theta_x, phi_g])
    act = layers.Activation("relu")(add_xg)
    psi = layers.Conv3D(1, 1, padding="same")(act)
    psi = layers.Activation("sigmoid")(psi)

    out = layers.Multiply()([x, psi])
    return out

# -------------------------
# Residual Attention UNet (3D) – 3 levels, no deep supervision
# -------------------------
def residual_att_unet_3d(input_shape=(32,128,128,1), base_filters=16, out_channels=1):
    inp = layers.Input(shape=input_shape)

    # Encoder
    c1 = res_conv_block(inp, base_filters)
    p1 = layers.MaxPool3D(pool_size=(2,2,2))(c1)

    c2 = res_conv_block(p1, base_filters*2)
    p2 = layers.MaxPool3D(pool_size=(2,2,2))(c2)

    c3 = res_conv_block(p2, base_filters*4)
    p3 = layers.MaxPool3D(pool_size=(2,2,2))(c3)

    # Bottleneck
    bn = res_conv_block(p3, base_filters*8)

    # Decoder level 3
    g3 = layers.Conv3D(base_filters*4, 1, padding="same")(bn)
    att3 = attention_gate_3d(c3, g3, inter_channels=base_filters*2)
    u3 = layers.UpSampling3D(size=(2,2,2))(bn)
    u3 = layers.Concatenate()([u3, att3])
    c4 = res_conv_block(u3, base_filters*4)

    # Decoder level 2
    g2 = layers.Conv3D(base_filters*2, 1, padding="same")(c4)
    att2 = attention_gate_3d(c2, g2, inter_channels=base_filters)
    u2 = layers.UpSampling3D(size=(2,2,2))(c4)
    u2 = layers.Concatenate()([u2, att2])
    c5 = res_conv_block(u2, base_filters*2)

    # Decoder level 1
    g1 = layers.Conv3D(base_filters, 1, padding="same")(c5)
    att1 = attention_gate_3d(c1, g1, inter_channels=max(1, base_filters//2))
    u1 = layers.UpSampling3D(size=(2,2,2))(c5)
    u1 = layers.Concatenate()([u1, att1])
    c6 = res_conv_block(u1, base_filters)

    # Final output
    out = layers.Conv3D(out_channels, 1, padding="same", activation="sigmoid")(c6)

    model = Model(inputs=inp, outputs=out, name="ResAttUNet3D_3level")
    return model

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example usage
    model = residual_srr_unet(input_shape=(32, 128, 128, 1))
    model.summary()

    # model = residual_att_unet_3d(input_shape=(32,128,128,1), base_filters=32, out_channels=1)
    # model.summary()
