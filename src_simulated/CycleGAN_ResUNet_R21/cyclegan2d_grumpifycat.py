# https://youtube/VzIO5_R9XEM
# https://youtube/2MSGnkir9ew

"""
cycleGAN model

Based on the code by Jason Brownlee from his blogs on https://machinelearningmastery.com/
I am adapting his code to various applications but original credit goes to Jason.

The model uses instance normalization layer:
Normalize the activations of the previous layer at each step,
i.e. applies a transformation that maintains the mean activation
close to 0 and the activation standard deviation close to 1.
Standardizes values on each output feature map rather than across features in a batch.

Download instance normalization code from here: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py
Or install keras_contrib using guidelines here: https://github.com/keras-team/keras-contrib
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import cv2

from random import random
from numpy import load, zeros, ones, asarray
from numpy.random import randint

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, LeakyReLU,
    Activation, Concatenate, Add
)

from matplotlib import pyplot

# discriminator model (70x70 patchGAN)
# C64-C128-C256-C512
#After the last layer, conv to 1-dimensional output, followed by a Sigmoid function.
# The “axis” argument is set to -1 for instance norm. to ensure that features are normalized per feature map.

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
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.math.sqrt(var + self.epsilon)
        return self.gamma * normalized + self.beta

def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)
	# C64: 4x4 kernel Stride 2x2
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128: 4x4 kernel Stride 2x2
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256: 4x4 kernel Stride 2x2
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512: 4x4 kernel Stride 2x2
    # Not in the original paper. Comment this block if you want.
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer : 4x4 kernel but Stride 1x1
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# define model
	model = Model(in_image, patch_out)
	# compile model
    #The model is trained with a batch size of one image and Adam opt.
    #with a small learning rate and 0.5 beta.
    #The loss for the discriminator is weighted by 50% for each model update.
    #This slows down changes to the discriminator relative to the generator model during training.
	model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
	model.summary()
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
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	model.summary()
	return model

#We define a composite model that will be used to train each generator separately.
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	# Make the generator of interest trainable as we will be updating these weights.
    #by keeping other models constant.
    #Remember that we use this same function to train both generators,
    #one generator at a time.
	g_model_1.trainable = True
	# mark discriminator and second generator as non-trainable
	d_model.trainable = False
	g_model_2.trainable = False

	# adversarial loss
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# identity loss
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# cycle loss - forward
	output_f = g_model_2(gen1_out)
	# cycle loss - backward
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)

	# define model graph
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])

    # define the optimizer
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
	model.compile(loss=['mse', 'mae', 'mae', 'mae'],
               loss_weights=[1, 5, 10, 10], optimizer=opt)
	model.summary()
	return model

# load and prepare training images
def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
#Remember that for real images the label (y) is 1.
def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y

# generate a batch of images, returns images and targets
#Remember that for fake images the label (y) is 0.
def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake images
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y


def save_models(step, g_model_AtoB, g_model_BtoA, output_path='src_simulated/outputs/grumpifycat'):
    # create directory if not exists
    os.makedirs(output_path, exist_ok=True)

    # -----------------------------
    # 1. Save checkpoint models
    # -----------------------------
    
    ckpt_AtoB = os.path.join(output_path, f"g_model_AtoB_{step+1:09d}.keras")
    ckpt_BtoA = os.path.join(output_path, f"g_model_BtoA_{step+1:09d}.keras")

    g_model_AtoB.save(ckpt_AtoB)
    g_model_BtoA.save(ckpt_BtoA)

    # -----------------------------
    # 2. Save "latest" models (overwrite)
    # -----------------------------
    # latest_AtoB = os.path.join(output_path, "g_model_AtoB_latest.keras")
    # latest_BtoA = os.path.join(output_path, "g_model_BtoA_latest.keras")

    # g_model_AtoB.save(latest_AtoB)
    # g_model_BtoA.save(latest_BtoA)

    # print(f"> Saved checkpoint: {ckpt_AtoB}  AND  {ckpt_BtoA}")
    print(f"> Updated latest models: {ckpt_AtoB}  AND  {ckpt_BtoA}")

# periodically generate images using the save model and plot input and output images
def summarize_performance(step, g_model, trainX, name, n_samples=5, output_path='src_simulated/outputs/grumpifycat'):
    # select a sample of input images
    X_in, _ = generate_real_samples(trainX, n_samples, 0)
    # generate translated images
    X_out, _ = generate_fake_samples(g_model, X_in, 0)
    # scale all pixels from [-1,1] to [0,1]
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    # plot real images
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_in[i].squeeze(), cmap='gray')
    # plot translated image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_out[i].squeeze(), cmap='gray')
    # save plot to file
    filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
    filename1 = os.path.join(output_path, filename1)
    pyplot.savefig(filename1)
    pyplot.close()

# update image pool for fake images to reduce model oscillation
# update discriminators using a history of generated images
#rather than the ones produced by the latest generators.
#Original paper recommended keeping an image buffer that stores
#the 50 previously created images.

def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)

# # train cyclegan models
# def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA,
#           c_model_AtoB, c_model_BtoA, dataset, epochs=1):

#     # define properties of the training run
#     n_epochs, n_batch = epochs, 1  # batch size fixed to 1
#     n_patch = d_model_A.output_shape[1]

#     # unpack dataset
#     trainA, trainB = dataset

#     # prepare image pool for fake images
#     poolA, poolB = list(), list()

#     # calculate iterations
#     bat_per_epo = int(len(trainA) / n_batch)
#     n_steps = bat_per_epo * n_epochs

#     # enumerate epochs
#     for i in range(n_steps):

#         # -----------------------------
#         # 1. REAL SAMPLES
#         # -----------------------------
#         X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
#         X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)

#         # -----------------------------
#         # 2. FAKE SAMPLES
#         # -----------------------------
#         X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
#         X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)

#         # update images in pool (buffer of 50 images)
#         X_fakeA = update_image_pool(poolA, X_fakeA)
#         X_fakeB = update_image_pool(poolB, X_fakeB)

#         # -----------------------------
#         # 3. TRAIN GENERATORS
#         # -----------------------------
#         g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch(
#             [X_realB, X_realA],
#             [y_realA, X_realA, X_realB, X_realA]
#         )

#         g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch(
#             [X_realA, X_realB],
#             [y_realB, X_realB, X_realA, X_realB]
#         )

#         # -----------------------------
#         # 4. TRAIN DISCRIMINATORS
#         # -----------------------------
#         dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
#         dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

#         dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
#         dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

#         # -----------------------------
#         # 5. LOG PROGRESS
#         # -----------------------------
#         if (i + 1) % 10 == 0:
#             print(
#                 'Iteration>%d, dA[%.3f, %.3f] dB[%.3f, %.3f] g[%.3f, %.3f]' %
#                 (i + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2)
#             )

#         # -----------------------------
#         # 6. PERIODIC PERFORMANCE CHECK
#         # -----------------------------
#         if (i + 1) % (bat_per_epo * 10) == 0:
#             summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
#             summarize_performance(i, g_model_BtoA, trainB, 'BtoA')

#         # -----------------------------
#         # 7. PERIODIC MODEL SAVE
#         # -----------------------------
#         if (i + 1) % (bat_per_epo * 5) == 0:
#             save_models(i, g_model_AtoB, g_model_BtoA,
#                         output_path ='src_simulated/outputs/cyclegan1')

# Train CycleGan model
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA,
          c_model_AtoB, c_model_BtoA, dataset, epochs=500):

    # define properties of the training run
    n_epochs, n_batch = epochs, 1  # batch size fixed to 1
    n_patch = d_model_A.output_shape[1]

    # unpack dataset
    trainA, trainB = dataset

    # prepare image pool for fake images
    poolA, poolB = list(), list()

    # calculate iterations
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs

    # enumerate epochs
    for i in range(n_steps):

        # -----------------------------
        # 1. REAL SAMPLES
        # -----------------------------
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)

        # -----------------------------
        # 2. FAKE SAMPLES
        # -----------------------------
        # Freeze generators, train discriminators
        g_model_AtoB.trainable = False
        g_model_BtoA.trainable = False
        d_model_A.trainable = True
        d_model_B.trainable = True

        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)

        # update images in pool (buffer of 50 images)
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)

        # -----------------------------
        # 3. TRAIN GENERATORS
        # -----------------------------
        # Freeze discriminators, train generators via composite model
        g_model_AtoB.trainable = True
        g_model_BtoA.trainable = True
        d_model_A.trainable = False
        d_model_B.trainable = False

        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch(
            [X_realB, X_realA],
            [y_realA, X_realA, X_realB, X_realA]
        )

        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch(
            [X_realA, X_realB],
            [y_realB, X_realB, X_realA, X_realB]
        )

        # -----------------------------
        # 4. TRAIN DISCRIMINATORS
        # -----------------------------
        # Freeze generators again
        g_model_AtoB.trainable = False
        g_model_BtoA.trainable = False
        d_model_A.trainable = True
        d_model_B.trainable = True

        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        # -----------------------------
        # 5. LOG PROGRESS
        # -----------------------------
        if (i + 1) % 100 == 0:
            print(
                'Iteration>%d, dA[%.3f, %.3f] dB[%.3f, %.3f] g[%.3f, %.3f]' %
                (i + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2)
            )

        # -----------------------------
        # 6. PERIODIC PERFORMANCE CHECK
        # -----------------------------
        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            summarize_performance(i, g_model_BtoA, trainB, 'BtoA')

        # -----------------------------
        # 7. PERIODIC MODEL SAVE
        # -----------------------------
        if (i + 1) % (bat_per_epo * 50) == 0:
            save_models(i, g_model_AtoB, g_model_BtoA,
                        output_path='src_simulated/outputs/grumpifycat')
            


import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

def load_images_from_folder(folder, image_size=(128, 128)):
    """
    Load all images from a folder, resize to `image_size`, and normalize to [-1, 1].
    Args:
        folder (str): Path to folder containing images.
        image_size (tuple): Desired image size (height, width).
    Returns:
        np.ndarray: Array of images of shape (num_images, height, width, 3)
    """
    images = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, fname)
            img = load_img(path, target_size=image_size, color_mode='rgb')  # force RGB
            img = img_to_array(img).astype(np.float32)

            # Normalize to [-1, 1]
            img = (img / 127.5) - 1.0

            images.append(img)

    images = np.array(images)
    print(f"[INFO] Loaded {images.shape[0]} images from {folder}")
    return images

# Paths
path = "src_simulated/grumpifycat/"
trainA_path = os.path.join(path, "trainA")
trainB_path = os.path.join(path, "trainB")

# Load data
dataA = load_images_from_folder(trainA_path, image_size=(128, 128))
dataB = load_images_from_folder(trainB_path, image_size=(128, 128))

print("Domain A:", dataA.shape)
print("Domain B:", dataB.shape)

# Combine into dataset list
dataset = [dataA, dataB]

# Function to quickly inspect images
def quick_inspect(A, B, n=5):
    """
    Display `n` images from domain A and B side by side.
    """
    plt.figure(figsize=(12, 5))
    for i in range(n):
        # Domain A
        plt.subplot(2, n, i+1)
        imgA = (A[i] + 1) / 2  # rescale to [0,1]
        plt.imshow(imgA.astype(np.float32))
        plt.axis('off')

        # Domain B
        plt.subplot(2, n, i+n+1)
        imgB = (B[i] + 1) / 2  # rescale to [0,1]
        plt.imshow(imgB.astype(np.float32))
        plt.axis('off')
    plt.show()

# Inspect first few images
quick_inspect(dataA, dataB)

# Set image shape for model input
image_shape = dataA.shape[1:]  # e.g., (128, 128, 3)
print("Image shape for model:", image_shape)


g_model_AtoB = define_generator(image_shape)
g_model_BtoA = define_generator(image_shape)

d_model_A = define_discriminator(image_shape)
d_model_B = define_discriminator(image_shape)

c_model_AtoB = define_composite_model(
    g_model_AtoB, d_model_B, g_model_BtoA, image_shape
)

c_model_BtoA = define_composite_model(
    g_model_BtoA, d_model_A, g_model_AtoB, image_shape
)


from datetime import datetime

start = datetime.now()

train(
    d_model_A,
    d_model_B,
    g_model_AtoB,
    g_model_BtoA,
    c_model_AtoB,
    c_model_BtoA,
    dataset,
    epochs=1000
)

end = datetime.now()
print("Training time:", end - start)
