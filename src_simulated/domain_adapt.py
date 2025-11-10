# domain adaptation model fastCUT implementation
import tensorflow as tf
from tensorflow.keras import layers, Model, Input   
class FastCUTDomainAdaptation(Model):
    def __init__(self, input_shape=(256, 256, 3), ngf=64):
        super(FastCUTDomainAdaptation, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(ngf, kernel_size=7, strides=1, padding='same', activation='relu'),
            layers.Conv2D(ngf * 2, kernel_size=4, strides=2, padding='same', activation='relu'),
            layers.Conv2D(ngf * 4, kernel_size=4, strides=2, padding='same', activation='relu'),
        ])
        
        self.residual_blocks = tf.keras.Sequential([
            *[self._residual_block(ngf * 4) for _ in range(6)]
        ])
        
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(ngf * 2, kernel_size=4, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(ngf, kernel_size=4, strides=2, padding='same', activation='relu'),
            layers.Conv2D(3, kernel_size=7, strides=1, padding='same', activation='tanh'),
        ])
    
    def _residual_block(self, filters):
        block = tf.keras.Sequential([
            layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters, kernel_size=3, strides=1, padding='same'),
        ])
        return block
    
    def call(self, x):
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        return x
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
    
        