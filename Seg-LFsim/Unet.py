import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np

def unet_3d(input_shape, num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)

    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)

    c3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling3D((2, 2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = layers.UpSampling3D((2, 2, 2))(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c5)

    u6 = layers.UpSampling3D((2, 2, 2))(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling3D((2, 2, 2))(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c7)

    outputs = layers.Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(c7)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def iou_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Example usage:
if __name__ == "__main__":
    # Example input shape: (depth, height, width, channels)
    input_shape = (64, 64, 64, 1)
    model = unet_3d(input_shape)
    model.compile(optimizer='adam',
                  loss=dice_loss,
                  metrics=[dice_coefficient, iou_coefficient])
    model.summary()

    # Dummy data for demonstration
    X = np.random.rand(2, 64, 64, 64, 1).astype(np.float32)
    y = np.random.randint(0, 2, (2, 64, 64, 64, 1)).astype(np.float32)

    # Training
    model.fit(X, y, epochs=2, batch_size=1)