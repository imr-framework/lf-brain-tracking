#implement diffusion model for denoising images in keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DiffusionModel(keras.Model):
    def __init__(self, image_size, channels, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        
        # Create beta schedule
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        
        # Define a simple UNet-like architecture for denoising
        self.encoder = keras.Sequential([
            layers.InputLayer(input_shape=(image_size, image_size, channels + 1)),  # +1 for timestep embedding
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
        ])
        
        self.decoder = keras.Sequential([
            layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2D(channels, 3, padding='same', activation='sigmoid'),
        ])
    
    def call(self, x, t):
        # Normalize timestep to [0, 1]
        t_normalized = t / self.timesteps
        t_embedding = tf.expand_dims(tf.expand_dims(t_normalized, axis=1), axis=1)
        t_embedding = tf.tile(t_embedding, [1, self.image_size, self.image_size])
        
        # Concatenate timestep embedding to input
        x_input = tf.concat([x, t_embedding], axis=-1)
        
        # Pass through encoder and decoder
        encoded = self.encoder(x_input)
        decoded = self.decoder(encoded)
        
        return decoded
    
    def add_noise(self, x0, t):
        
        # x0: (B, H, W, C) or (B, D, H, W, C)
        # t: (B,) - timesteps
        alpha_bar_t = tf.gather(self.alpha_bar, t)  # shape: (B,)

        # reshape for broadcasting
        # for 2D images:
        alpha_bar_t = tf.reshape(alpha_bar_t, (-1, 1, 1, 1))
        # for 3D volumes:
        # alpha_bar_t = tf.reshape(alpha_bar_t, (-1, 1, 1, 1, 1))

        noise = tf.random.normal(shape=tf.shape(x0))
        noisy_image = tf.sqrt(alpha_bar_t) * x0 + tf.sqrt(1 - alpha_bar_t) * noise

        return noisy_image, noise

    def sample(self, batch_size):
        x_t = tf.random.normal(shape=(batch_size, self.image_size, self.image_size, self.channels))
        for t in reversed(range(self.timesteps)):
            t_batch = tf.fill([batch_size], t)
            predicted_noise = self.call(x_t, t_batch)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            if t > 0:
                noise = tf.random.normal(shape=tf.shape(x_t))
            else:
                noise = 0
            x_t = (1 / tf.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / tf.sqrt(1 - alpha_bar_t) * predicted_noise) + tf.sqrt(self.betas[t]) * noise
        return x_t
# Example usage:
if __name__ == "__main__":
    image_size = 32
    channels = 3
    model = DiffusionModel(image_size, channels)
    
    # Dummy data
    x0 = tf.random.normal(shape=(4, image_size, image_size, channels))
    t = tf.constant([10, 20, 30, 40], dtype=tf.int32)
    
    # Add noise
    noisy_images, noise = model.add_noise(x0, t)
    
    # Denoise
    denoised_images = model.call(noisy_images, t)
    
    # Sample new images
    sampled_images = model.sample(batch_size=4)
    print("Sampled images shape:", sampled_images.shape)
