# prompt: generate UNet encoder with 2 heads encoder 1 takesZSSR output as encoder 1 input, encoder 2 takes Hf input then merge feature maps at bottle neck and decoder to give 256,256,64 ;  with target as high field; make way to map 16 to 64 and use np.abs for conversition to float

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize  # ✅ Required for 3D resizing
# import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_dual_encoder_unet(input_shape_zssr, input_shape_hf, output_shape=(256, 256, 64, 1)):
    """
    Builds a U-Net model with two encoders and a shared decoder.
    Encoder 1 takes ZSSR output, Encoder 2 takes HF input.
    Features are merged at the bottleneck.

    Args:
        input_shape_zssr (tuple): Shape of the ZSSR input (height, width, depth, channels).
        input_shape_hf (tuple): Shape of the HF input (height, width, depth, channels).
        output_shape (tuple): Desired output shape (height, width, depth, channels).

    Returns:
        tf.keras.Model: The dual-encoder U-Net model.
    """

    # --- Encoder 1 (ZSSR Input) ---
    inputs_zssr = Input(shape=input_shape_zssr, name='zssr_input')

    conv1_zssr = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs_zssr)
    conv1_zssr = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv1_zssr)
    pool1_zssr = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1_zssr)

    conv2_zssr = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(pool1_zssr)
    conv2_zssr = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv2_zssr)
    pool2_zssr = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2_zssr)

    conv3_zssr = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(pool2_zssr)
    conv3_zssr = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv3_zssr)
    pool3_zssr = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3_zssr)

    # --- Encoder 2 (HF Input) ---
    inputs_hf = Input(shape=input_shape_hf, name='hf_input')

    conv1_hf = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs_hf)
    conv1_hf = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv1_hf)
    pool1_hf = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1_hf)

    conv2_hf = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(pool1_hf)
    conv2_hf = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv2_hf)
    pool2_hf = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2_hf)

    conv3_hf = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(pool2_hf)
    conv3_hf = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv3_hf)
    pool3_hf = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3_hf)

    # --- Bottleneck (Merge Features) ---
    # Concatenate features from both encoders at the deepest level
    merge_bottleneck = layers.concatenate([pool3_zssr, pool3_hf], axis=-1)

    conv4 = layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same')(merge_bottleneck)
    conv4 = layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv4)

    # --- Decoder (Upsampling Path) ---
    # Skip connections come from the corresponding levels of the HF encoder (or can be designed differently)
    # Here, we'll use skip connections from the HF encoder for simplicity, mapping to the target HF.
    # Alternatively, you could add skip connections from both encoders and concatenate before the upsampling.
    # Let's use HF skip connections and the merged bottleneck.

    up5 = layers.Conv3DTranspose(128, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv4)
    # Concatenate with skip connection from HF encoder
    merge5 = layers.concatenate([conv3_hf, up5], axis=-1)
    conv5 = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(merge5)
    conv5 = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = layers.Conv3DTranspose(64, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
    # Concatenate with skip connection from HF encoder
    merge6 = layers.concatenate([conv2_hf, up6], axis=-1)
    conv6 = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(merge6)
    conv6 = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = layers.Conv3DTranspose(32, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
    # Concatenate with skip connection from HF encoder
    merge7 = layers.concatenate([conv1_hf, up7], axis=-1)
    conv7 = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(merge7)
    conv7 = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv7)

    # Output layer
    # Output shape should match the desired target HF shape (256, 256, 64, 1)
    outputs = layers.Conv3D(output_shape[-1], kernel_size=(1, 1, 1), activation='linear', padding='same')(conv7)

    model = models.Model(inputs=[inputs_zssr, inputs_hf], outputs=outputs)

    return model

# Example usage
zssr_input_shape = (32, 128, 128, 1)  # Example ZSSR output shape
hf_input_shape = (32, 128, 128, 1)     # Example HF input shape
model = build_dual_encoder_unet(zssr_input_shape, hf_input_shape)
model.summary()