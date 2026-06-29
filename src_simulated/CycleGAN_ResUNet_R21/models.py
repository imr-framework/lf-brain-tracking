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

# With Conv2dTranspose
def define_generator(image_shape, context_dim=None, n_resnet=9):
    """
    2D ResNet-based generator with optional context input.
    
    Parameters
    ----------
    image_shape : tuple
        Shape of input image, e.g., (H, W, C)
    context_dim : int or None
        Dimension of additional context features
    n_resnet : int
        Number of resnet blocks
    """

    # weight initialization
    init = RandomNormal(stddev=0.02)
    
    # image input
    in_image = Input(shape=image_shape)
    
    if context_dim is not None:
        # context input
        in_context = Input(shape=(context_dim,))
        # expand context to spatial map and concatenate
        c = tf.keras.layers.Dense(image_shape[0]*image_shape[1])(in_context)
        c = tf.keras.layers.Reshape((image_shape[0], image_shape[1], 1))(c)
        x = Concatenate()([in_image, c])
    else:
        x = in_image

    # c7s1-64
    g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(x)
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
    
    # c7s1-1
    g = Conv2D(1, (7,7), padding='same', kernel_initializer=init)(g)
    out_image = Activation('tanh')(g)

    if context_dim is not None:
        model = Model([in_image, in_context], out_image)
    else:
        model = Model(in_image, out_image)

    # model.summary()
    return model

# With Conv2dTranspose
def define_unet_generator(image_shape, context_dim=None, n_resnet=6):
    """
    2D ResNet-based generator with optional context input.
    
    Parameters
    ----------
    image_shape : tuple
        Shape of input image, e.g., (H, W, C)
    context_dim : int or None
        Dimension of additional context features
    n_resnet : int
        Number of resnet blocks
    """

    # weight initialization
    init = RandomNormal(stddev=0.02)
    
    # image input
    in_image = Input(shape=image_shape)
    
    if context_dim is not None:
        # context input
        in_context = Input(shape=(context_dim,))
        # expand context to spatial map and concatenate
        c = tf.keras.layers.Dense(image_shape[0]*image_shape[1])(in_context)
        c = tf.keras.layers.Reshape((image_shape[0], image_shape[1], 1))(c)
        x = Concatenate()([in_image, c])
    else:
        x = in_image

    # c3s1-32 - 128x128x32
    g = Conv2D(32, (3,3), padding='same', kernel_initializer=init)(x)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # c3s1-64 - 64x64x64
    g = Conv2D(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    
    # d128 - 32x32x128
    g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    
    # d256 - 16x16x256
    g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    
    # R256 - 16x16x256
    for _ in range(n_resnet):
        g = resnet_block(256, g)
    
    # u128 - 32x32x128
    g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    
    # u64 - 64x64x64
    g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # u32 - 128x128x32
    g = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # c3s1-1 - 128x128x1
    g = Conv2D(1, (3,3), padding='same', kernel_initializer=init)(g)
    out_image = Activation('tanh')(g)

    if context_dim is not None:
        model = Model([in_image, in_context], out_image)
    else:
        model = Model(in_image, out_image)

    model.summary()
    return model

# Generator with skip connections (U-Net like)
def define_unet_skip_generator(image_shape, context_dim=None, n_resnet=6):

    init = RandomNormal(stddev=0.02)
    
    in_image = Input(shape=image_shape)
    
    if context_dim is not None:
        in_context = Input(shape=(context_dim,))
        c = tf.keras.layers.Dense(image_shape[0]*image_shape[1])(in_context)
        c = tf.keras.layers.Reshape((image_shape[0], image_shape[1], 1))(c)
        x = Concatenate()([in_image, c])
    else:
        x = in_image

    # -----------------
    # ENCODER
    # -----------------

    # e1: 128x128x32
    e1 = Conv2D(32, (3,3), padding='same', kernel_initializer=init)(x)
    e1 = InstanceNormalization(axis=-1)(e1)
    e1 = Activation('relu')(e1)

    # e2: 64x64x64
    e2 = Conv2D(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(e1)
    e2 = InstanceNormalization(axis=-1)(e2)
    e2 = Activation('relu')(e2)

    # e3: 32x32x128
    e3 = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(e2)
    e3 = InstanceNormalization(axis=-1)(e3)
    e3 = Activation('relu')(e3)

    # bottleneck: 16x16x256
    g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(e3)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # ResNet blocks
    for _ in range(n_resnet):
        g = resnet_block(256, g)

    # -----------------
    # DECODER + SKIPS
    # -----------------

    # u128: 16→32
    g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Concatenate()([g, e3])   # ⭐ skip connection

    # u64: 32→64
    g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Concatenate()([g, e2])   # ⭐ skip connection

    # u32: 64→128
    g = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Concatenate()([g, e1])   # ⭐ skip connection

    # output
    g = Conv2D(1, (3,3), padding='same', kernel_initializer=init)(g)
    out_image = Activation('tanh')(g)

    if context_dim is not None:
        model = Model([in_image, in_context], out_image)
    else:
        model = Model(in_image, out_image)

    model.summary()
    return model

def define_composite_model(g_model_1, d_model, g_model_2, image_shape, context_dim=None):
    """
    Create a composite CycleGAN model for training a generator with:
    - adversarial loss
    - identity loss
    - cycle-consistency loss

    Supports optional context input.

    Parameters
    ----------
    g_model_1 : keras.Model
        Generator being trained
    d_model : keras.Model
        Target discriminator
    g_model_2 : keras.Model
        Other generator (for cycle loss)
    image_shape : tuple
        Shape of input image
    context_dim : int or None
        Dimension of context vector
    """

    # Make only g_model_1 trainable
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False

    # Inputs
    in_image = Input(shape=image_shape)
    if context_dim is not None:
        in_context = Input(shape=(context_dim,))
        gen1_out = g_model_1([in_image, in_context])
        # Identity input also needs context
        input_id = Input(shape=image_shape)
        output_id = g_model_1([input_id, in_context])
        # Cycle outputs
        output_f = g_model_2([gen1_out, in_context])
        gen2_out = g_model_2([input_id, in_context])
        output_b = g_model_1([gen2_out, in_context])
        model_inputs = [in_image, in_context, input_id]
    else:
        gen1_out = g_model_1(in_image)
        # Discriminator output for adversarial loss
        output_d = d_model(gen1_out)
        # Identity loss
        input_id = Input(shape=image_shape)
        output_id = g_model_1(input_id)
        # Cycle outputs
        output_f = g_model_2(gen1_out)
        gen2_out = g_model_2(input_id)
        output_b = g_model_1(gen2_out)
        model_inputs = [in_image, input_id]
    
    # Discriminator output
    if context_dim is None:
        output_d = d_model(gen1_out)
        outputs = [output_d, output_id, output_f, output_b]
    else:
        # when context is used, d_model usually only takes the generated image
        output_d = d_model(gen1_out)
        outputs = [output_d, output_id, output_f, output_b]

    # Define composite model
    model = Model(inputs=model_inputs, outputs=outputs)

    # Optimizer
    opt = Adam(
        learning_rate=GEN_LEARNING_RATE,
        beta_1=GEN_BETA_1
    )

    # Compile
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
    # d_model = define_discriminator(image_shape)
    g_model_1 = define_unet_generator(image_shape, context_dim=5, n_resnet=6)
    g_model_2 = define_unet_skip_generator(image_shape, context_dim=5, n_resnet=6)
    # model = define_composite_model(g_model_1, d_model, g_model_2, image_shape)