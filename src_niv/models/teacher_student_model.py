import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, Model
import numpy as np
import os

# -----------------------------
# Utilities: residual block + SE-like attention fusion
# -----------------------------
def res_conv_block(x, filters, kernel_size=3):
    """Residual convolutional block"""
    shortcut = layers.Conv3D(filters, 1, padding="same")(x)
    x = layers.Conv3D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.Conv3D(filters, kernel_size, padding="same")(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def se_fusion(concat_feat, reduction=8):
    """
    Simple squeeze-excite style fusion block for concatenated LF+HF features.
    concat_feat: tensor with channels = C_l + C_h
    returns: fused tensor with same shape as concat_feat
    """
    ch = concat_feat.shape[-1]
    se = layers.GlobalAveragePooling3D()(concat_feat)         # (batch, channels)
    se = layers.Dense(max(1, ch//reduction), activation='relu')(se)
    se = layers.Dense(ch, activation='sigmoid')(se)
    se = layers.Reshape([1,1,1,ch])(se)
    return layers.Multiply()([concat_feat, se])

# -----------------------------
# Encoder builder (returns skips + bottleneck)
# -----------------------------
def build_encoder(input_shape, base_filters=32, name="encoder"):
    inp = layers.Input(shape=input_shape)
    # Level 1
    c1 = res_conv_block(inp, base_filters)
    p1 = layers.MaxPool3D((2,2,2))(c1)   # down1
    # Level 2
    c2 = res_conv_block(p1, base_filters*2)
    p2 = layers.MaxPool3D((2,2,2))(c2)   # down2
    # Level 3
    c3 = res_conv_block(p2, base_filters*4)
    p3 = layers.MaxPool3D((2,2,2))(c3)   # down3
    # Bottleneck
    bn = res_conv_block(p3, base_filters*8)
    return Model(inp, [c1, c2, c3, bn], name=name)

# -----------------------------
# Decoder that consumes fused features and fused skips
# -----------------------------
def build_decoder(base_filters=32, name="decoder"):
    # Provide input shapes generically (we will call on tensors)
    bn_in = layers.Input(shape=(None, None, None, base_filters*16))  # fused channels at bottleneck
    s3 = layers.Input(shape=(None, None, None, base_filters*8))      # fused c3
    s2 = layers.Input(shape=(None, None, None, base_filters*4))      # fused c2
    s1 = layers.Input(shape=(None, None, None, base_filters*2))      # fused c1

    # Up 1
    u3 = layers.UpSampling3D((2,2,2))(bn_in)
    u3 = layers.Concatenate()([u3, s3])
    c4 = res_conv_block(u3, base_filters*8)

    # Up 2
    u2 = layers.UpSampling3D((2,2,2))(c4)
    u2 = layers.Concatenate()([u2, s2])
    c5 = res_conv_block(u2, base_filters*4)

    # Up 3
    u1 = layers.UpSampling3D((2,2,2))(c5)
    u1 = layers.Concatenate()([u1, s1])
    c6 = res_conv_block(u1, base_filters*2)

    # Final conv to reconstruct residual
    residual = layers.Conv3D(1, (1,1,1), padding="same", activation="linear")(c6)
    # Output will be residual; caller can add shortcut if needed
    return Model([bn_in, s3, s2, s1], residual, name=name)

# -----------------------------
# Build teacher model (dual encoders + fusion + decoder)
# -----------------------------
def build_teacher_model(input_shape, base_filters=32):
    # inputs
    lf_in = layers.Input(shape=input_shape, name="LF_input")
    hf_in = layers.Input(shape=input_shape, name="HF_input")

    # encoders (shared architecture but separate weights)
    enc_lf = build_encoder(input_shape, base_filters, name="LF_encoder")
    enc_hf = build_encoder(input_shape, base_filters, name="HF_encoder")

    lf_c1, lf_c2, lf_c3, lf_bn = enc_lf(lf_in)
    hf_c1, hf_c2, hf_c3, hf_bn = enc_hf(hf_in)

    # fuse each skip and bottleneck with SE fusion
    fused_c1 = se_fusion(layers.Concatenate()([lf_c1, hf_c1]))
    fused_c2 = se_fusion(layers.Concatenate()([lf_c2, hf_c2]))
    fused_c3 = se_fusion(layers.Concatenate()([lf_c3, hf_c3]))
    fused_bn = se_fusion(layers.Concatenate()([lf_bn, hf_bn]))

    # build decoder and produce residual prediction
    decoder = build_decoder(base_filters=base_filters)
    residual_pred = decoder([fused_bn, fused_c3, fused_c2, fused_c1])

    # teacher output = lf_input + residual_pred (residual learning)
    out = layers.Add(name="teacher_output")([lf_in, residual_pred])

    teacher_model = Model([lf_in, hf_in], out, name="teacher_model")
    # Also return components for distillation later
    return teacher_model, enc_lf, enc_hf, decoder

# -----------------------------
# Build student model (single LF encoder + same decoder)
# -----------------------------
def build_student_model(input_shape, enc_lf, decoder, base_filters=32):
    # enc_lf and decoder are reused
    lf_in = layers.Input(shape=input_shape, name="LF_input_student")
    # get LF features
    lf_c1, lf_c2, lf_c3, lf_bn = enc_lf(lf_in)

    # For student-only inference, we must produce fused features compatible with decoder.
    # Strategy: duplicate LF features to simulate HF features, then apply se_fusion on concat(LF,LF)
    fused_c1 = se_fusion(layers.Concatenate()([lf_c1, lf_c1]))
    fused_c2 = se_fusion(layers.Concatenate()([lf_c2, lf_c2]))
    fused_c3 = se_fusion(layers.Concatenate()([lf_c3, lf_c3]))
    fused_bn = se_fusion(layers.Concatenate()([lf_bn, lf_bn]))

    residual_pred = decoder([fused_bn, fused_c3, fused_c2, fused_c1])
    out = layers.Add(name="student_output")([lf_in, residual_pred])

    student_model = Model(lf_in, out, name="student_model")
    return student_model

# -----------------------------
# Example usage: build, compile, train teacher -> distill student
# -----------------------------
def train_teacher_and_student(
    lf_train, hf_train,
    input_shape=(32,128,128,1),
    base_filters=32,
    teacher_epochs=50,
    student_epochs=50,
    distill_alpha=1.0,
    save_dir="./models"
):
    os.makedirs(save_dir, exist_ok=True)

    # Build teacher
    teacher_model, enc_lf, enc_hf, decoder = build_teacher_model(input_shape, base_filters)
    teacher_model.compile(optimizer=optimizers.Adam(1e-4),
                          loss=losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanAbsoluteError()])
    print(teacher_model.summary())

    # Train teacher on (LF, HF) -> HF
    teacher_model.fit([lf_train, hf_train], hf_train,
                      batch_size=1, epochs=teacher_epochs)

    teacher_path = os.path.join(save_dir, "teacher_model.keras")
    teacher_model.save(teacher_path)
    print("Saved teacher model to", teacher_path)

    # Build student (re-using LF encoder weights and decoder architecture)
    # Important: use enc_lf and decoder trained with teacher; keep weights
    student_model = build_student_model(input_shape, enc_lf, decoder, base_filters)
    student_model.compile(optimizer=optimizers.Adam(1e-4), loss=losses.MeanSquaredError(), metrics=["mae"])
    print(student_model.summary())

    # Distillation stage: student tries to match teacher outputs (on LF only).
    # Prepare teacher predictions on training set (soft targets)
    teacher_preds = teacher_model.predict([lf_train, hf_train], batch_size=1)

    # Student training: minimize reconstruction to teacher_preds and optional MSE to hf_train
    # Composite loss: L = mse(student_pred, teacher_pred) + beta * mse(student_pred, hf_true)
    beta = 0.0  # set >0 if you want student to also match ground-truth directly
    def composite_loss(y_true, y_pred):
        # y_true will be tuple passed as teacher_preds if we wrap dataset; simplest: use teacher_preds array directly
        return tf.reduce_mean(tf.square(y_true - y_pred))

    # Fit student to teacher_preds
    student_model.fit(lf_train, teacher_preds, batch_size=1, epochs=student_epochs)

    student_path = os.path.join(save_dir, "student_model.keras")
    student_model.save(student_path)
    print("Saved student model to", student_path)

    return teacher_model, student_model

# -----------------------------
# Quick demo run (synthetic data) - replace with your real data
# -----------------------------
if __name__ == "__main__":
    # synthetic small example: replace with your LF/HF arrays shaped (N, Z, X, Y, 1)
    N = 2
    Z, X, Y = 32, 128, 128
    lf_train = np.random.rand(N, Z, X, Y, 1).astype("float32")
    hf_train = np.random.rand(N, Z, X, Y, 1).astype("float32")

    teacher, student = train_teacher_and_student(
        lf_train, hf_train,
        input_shape=(Z,X,Y,1),
        base_filters=16,      # reduce filters for demo to save memory
        teacher_epochs=2,
        student_epochs=2,
        save_dir="./models_demo"
    )

    # example inference with student model (single input)
    lf_test = np.random.rand(1, Z, X, Y, 1).astype("float32")
    pred = student.predict(lf_test)
    print("student prediction shape:", pred.shape)