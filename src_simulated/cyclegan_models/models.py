import sys
sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, LeakyReLU,
    Activation, Concatenate, Add, UpSampling2D, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.utils import register_keras_serializable

# Load config = CycleGANConfig() class from config.py
from src_simulated.cyclegan_models.config import config
# import losses from losses.py
from src_simulated.cyclegan_models.losses_cyclegan import *

EPOCHS = config.EPOCHS
TEST = config.TEST

#parameters for descriminator
DISC_LOSS = config.DISC_LOSS
DISC_LEARNING_RATE = config.DISC_LEARNING_RATE
DISC_BETA_1 = config.DISC_BETA_1
DISC_LOSS_WEIGHTS = config.DISC_LOSS_WEIGHTS

#parameters for generator
GEN_LOSS_1 = config.GEN_LOSS_1
GEN_LOSS_2 = config.GEN_LOSS_2
GEN_LOSS_3 = config.GEN_LOSS_3
GEN_LOSS_4 = config.GEN_LOSS_4
GEN_LEARNING_RATE = config.GEN_LEARNING_RATE
GEN_BETA_1 = config.GEN_BETA_1
GEN_LOSS_WEIGHTS = config.GEN_LOSS_WEIGHTS

@register_keras_serializable(package='custom_layers')
class InstanceNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.gamma = self.add_weight(
            shape=(dim,),
            initializer="ones",
            trainable=True,
            name="gamma"
        )
        self.beta = self.add_weight(
            shape=(dim,),
            initializer="zeros",
            trainable=True,
            name="beta"
        )
        super().build(input_shape)

    def call(self, inputs):
        # Compute mean/variance over spatial dimensions except channel
        axes = [i for i in range(1, len(inputs.shape)) if i != self.axis]
        mean, var = tf.nn.moments(inputs, axes=axes, keepdims=True)
        normalized = (inputs - mean) / tf.math.sqrt(var + self.epsilon)
        return self.gamma * normalized + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config

def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    # source image input
    in_image = Input(shape=image_shape)

    # C64: 4x4 kernel, stride 2x2
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    # C128: 4x4 kernel, stride 2x2
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C256: 4x4 kernel, stride 2x2
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    # Optional C512 block (not in original paper)
    # d = Conv2D(512, (4, 4), strides=(2, 2), padding='same',
    #            kernel_initializer=init)(d)
    # d = InstanceNormalization(axis=-1)(d)
    # d = LeakyReLU(alpha=0.2)(d)

    # Second last layer: 4x4 kernel, stride 1x1
    d = Conv2D(512, (4, 4), padding='same',
               kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    # PatchGAN output
    patch_out = Conv2D(1, (4, 4), padding='same',
                       kernel_initializer=init)(d)

    # Define model
    model = Model(in_image, patch_out)

    # Compile model
    # This slows down changes to the discriminator relative to the generator
    model.compile(
        loss=DISC_LOSS,
        optimizer=Adam(
            learning_rate=DISC_LEARNING_RATE,
            beta_1=DISC_BETA_1
        ),
        loss_weights=DISC_LOSS_WEIGHTS
    )

    return model


# generator a resnet block to be used in the generator
# residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layers.
def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Add()([g, input_layer])
	return g

# define the  generator model - encoder-decoder type architecture

#c7s1-k denote a 7×7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1.
#dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2.
# Rk denotes a residual block that contains two 3 × 3 convolutional layers
# uk denotes a 3 × 3 fractional-strided-Convolution InstanceNorm-ReLU layer with k filters and stride 1/2

#The network with 6 residual blocks consists of:
#c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3

#The network with 9 residual blocks consists of:
#c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128, u64,c7s1-3

# def define_generator(image_shape, n_resnet=6):
#     # weight initialization
#     init = RandomNormal(stddev=0.02)
    
#     # image input
#     in_image = Input(shape=image_shape)
    
#     # c7s1-64
#     g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
#     g = InstanceNormalization(axis=-1)(g)
#     g = Activation('relu')(g)
    
#     # d128
#     g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
#     g = InstanceNormalization(axis=-1)(g)
#     g = Activation('relu')(g)
    
#     # d256
#     g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
#     g = InstanceNormalization(axis=-1)(g)
#     g = Activation('relu')(g)
    
#     # R256
#     for _ in range(n_resnet):
#         g = resnet_block(256, g)
    
#     # u128 (upsample + conv)

#     g = UpSampling2D(size=(2,2), interpolation='nearest')(g)
#     g = Conv2D(128, (3,3), padding='same', kernel_initializer=init)(g)
#     g = InstanceNormalization(axis=-1)(g)
#     g = Activation('relu')(g)
    
#     # u64
#     g = UpSampling2D(size=(2,2), interpolation='nearest')(g)
#     g = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(g)
#     g = InstanceNormalization(axis=-1)(g)
#     g = Activation('relu')(g)
    
#     # c7s1-3 (output)
#     g = Conv2D(1, (7,7), padding='same', kernel_initializer=init)(g)
#     out_image = Activation('tanh')(g)  # no instance norm here
    
#     # define model
#     model = Model(in_image, out_image)
#     # model.summary()
#     return model

# # with Conv2dtranspose

def define_generator(image_shape, n_resnet=9):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(1, (7,7), padding='same', kernel_initializer=init)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	model.summary()
	return model

#Generator with skip connections (U-Net like)

# from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Activation, Concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.initializers import RandomNormal
# # from tensorflow_addons.layers import InstanceNormalization

# def define_generator(image_shape, n_resnet=9):
#     init = RandomNormal(stddev=0.02)

#     # Input
#     in_image = Input(shape=image_shape)

#     # -------- Encoder --------
#     # c7s1-64
#     g1 = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
#     g1 = InstanceNormalization(axis=-1)(g1)
#     g1 = Activation('relu')(g1)

#     # d128
#     g2 = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g1)
#     g2 = InstanceNormalization(axis=-1)(g2)
#     g2 = Activation('relu')(g2)

#     # d256
#     g3 = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g2)
#     g3 = InstanceNormalization(axis=-1)(g3)
#     g3 = Activation('relu')(g3)

#     # -------- Residual Blocks --------
#     r = g3
#     for _ in range(n_resnet):
#         r = resnet_block(256, r)

#     # -------- Decoder --------
#     # u128 + skip from g2
#     u1 = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same',
#                           kernel_initializer=init)(r)
#     u1 = InstanceNormalization(axis=-1)(u1)
#     u1 = Activation('relu')(u1)
#     u1 = Concatenate()([u1, g2])

#     # u64 + skip from g1
#     u2 = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same',
#                           kernel_initializer=init)(u1)
#     u2 = InstanceNormalization(axis=-1)(u2)
#     u2 = Activation('relu')(u2)
#     u2 = Concatenate()([u2, g1])

#     # -------- Output --------
#     out = Conv2D(1, (7,7), padding='same', kernel_initializer=init)(u2)
#     out_image = Activation('tanh')(out)

#     model = Model(in_image, out_image)
#     model.summary()
#     return model


# We define a composite model that will be used to train each generator separately.
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    # Make the generator of interest trainable
    # (other models remain frozen)
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False

    # Adversarial loss
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)

    # Identity loss
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)

    # Cycle loss (forward)
    output_f = g_model_2(gen1_out)

    # Cycle loss (backward)
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    # Define composite model
    model = Model(
        inputs=[input_gen, input_id],
        outputs=[output_d, output_id, output_f, output_b]
    )

    # Optimizer
    opt = Adam(
        learning_rate=GEN_LEARNING_RATE,
        beta_1=GEN_BETA_1
    )

    # Compile model
    model.compile(
        optimizer=opt,
        loss=[
            GEN_LOSS_1,  # adversarial
            GEN_LOSS_2,  # identity
            GEN_LOSS_3,  # cycle forward
            GEN_LOSS_4   # cycle backward
        ],
        loss_weights=GEN_LOSS_WEIGHTS
    )

    return model

# call main to test
if __name__ == '__main__':
    image_shape = (128, 128, 1)
    d_model = define_discriminator(image_shape)
    g_model_1 = define_generator(image_shape)
    g_model_2 = define_generator(image_shape)
    model = define_composite_model(g_model_1, d_model, g_model_2, image_shape)