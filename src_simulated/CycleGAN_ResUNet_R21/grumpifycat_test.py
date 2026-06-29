import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
# # Use the saved cyclegan models for image translation
from keras.models import load_model
from matplotlib import pyplot
from numpy.random import randint
import nibabel as nib

# inference test
import tensorflow as tf
from keras.layers import Layer
from keras.saving import register_keras_serializable

from sklearn.utils import resample

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


def inspect_domains(A, B, n_samples=20):
	"""
	Prints stats and displays 20 samples (top=A, bottom=B).
	"""

	# ------------------------- Helper: describe -------------------------
	def describe(name, X):
		print(f"\n{name} Dataset:")
		print(f"  Shape         : {X.shape}")
		print(f"  Min pixel     : {X.min():.4f}")
		print(f"  Max pixel     : {X.max():.4f}")
		print(f"  Mean pixel    : {X.mean():.4f}")

	# ------------------------- Print stats -------------------------
	describe("A", A)
	describe("B", B)

	# ------------------------- Visualization -------------------------
	print("\nDisplaying sample images...")

	plt.figure(figsize=(20, 4))
	n = min(n_samples, len(A), len(B))

	for i in range(n):

		# ---------- TOP ROW (A) ----------
		plt.subplot(2, n, 1 + i)
		plt.axis("off")
		imgA = (A[i] + 1) / 2
		if imgA.shape[-1] == 1:
			imgA = imgA[:, :, 0]
			plt.imshow(imgA, cmap="gray")
		else:
			plt.imshow(imgA)

		# ---------- BOTTOM ROW (B) ----------
		plt.subplot(2, n, 1 + n + i)
		plt.axis("off")
		imgB = (B[i] + 1) / 2
		if imgB.shape[-1] == 1:
			imgB = imgB[:, :, 0]
			plt.imshow(imgB, cmap="gray")
		else:
			plt.imshow(imgB)

	plt.tight_layout()
	plt.show()

# Inspect first few images
quick_inspect(dataA, dataB)

# Set image shape for model input
image_shape = dataA.shape[1:]  # e.g., (128, 128, 3)
print("Image shape for model:", image_shape)

# from cycleGAN_model import define_generator, define_discriminator, define_composite_model, train
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
print(image_shape)

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X

# plot the image, its translation, and the reconstruction
from numpy import vstack
from matplotlib import pyplot

def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']

    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0

    for i in range(len(images)):
        pyplot.subplot(1, len(images), 1 + i)
        pyplot.axis('off')

        img = images[i]

        # ✅ handle (H, W, 1) or (1, H, W)
        if img.ndim == 3:
            img = img.squeeze()

        # ✅ force grayscale
        pyplot.imshow(img, cmap='gray', vmin=0, vmax=1)
        pyplot.title(titles[i])

    pyplot.show()

def show_plot_domains(A_real, A_gen, A_rec, B_real, B_gen, B_rec, save_path=None):
    def get_middle_slice_rgb(img):
        # img: (H, W, C) or (N, H, W, C)
        if img.ndim == 4:
            img = img[0]
        # scale from [-1,1] to [0,1]
        img = (img + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        return img

    A_real_img = get_middle_slice_rgb(A_real)
    A_gen_img  = get_middle_slice_rgb(A_gen)
    A_rec_img  = get_middle_slice_rgb(A_rec)

    B_real_img = get_middle_slice_rgb(B_real)
    B_gen_img  = get_middle_slice_rgb(B_gen)
    B_rec_img  = get_middle_slice_rgb(B_rec)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    col_titles = ['Input x', 'Output G(x)', 'Reconstruction F(G(x))']
    row_labels = ['A Domain', 'B Domain']

    for row_idx, row_imgs in enumerate([
        [A_real_img, A_gen_img, A_rec_img],
        [B_real_img, B_gen_img, B_rec_img]
    ]):
        for col_idx, img in enumerate(row_imgs):
            ax = axes[row_idx, col_idx]
            if img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(img)
            ax.axis('off')

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=12, color='black')

            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx],
                              rotation=90,
                              fontsize=12,
                              labelpad=10,
                              color='black')

    plt.subplots_adjust(wspace=0, hspace=0)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)

    plt.show()

# load dataset
A_data = resample(dataset[0],
                 replace=False,
                 n_samples=50,
                 random_state=42) # reproducible results

B_data = resample(dataset[1],
                 replace=False,
                 n_samples=50,
                 random_state=42) # reproducible results

# select a sample of images
A_real = select_sample(A_data, 1)
B_real = select_sample(B_data, 1)

#print shapes
print("A_real shape:", A_real.shape)
print("B_real shape:", B_real.shape)

inspect_domains(A_real, B_real, n_samples=1)

# load the models
g_model_AtoB = load_model('src_simulated/outputs/grumpifycat/g_model_AtoB_000088000.keras', custom_objects={'InstanceNormalization': InstanceNormalization})
g_model_BtoA = load_model('src_simulated/outputs/grumpifycat/g_model_BtoA_000088000.keras', custom_objects={'InstanceNormalization': InstanceNormalization})

# generate images
A_generated = g_model_BtoA.predict(B_real)
B_generated = g_model_AtoB.predict(A_real)

# print shapes
print("A_generated shape:", A_generated.shape)
print("B_generated shape:", B_generated.shape)

# check min and max of generated images
print("A_generated range:", np.min(A_generated), np.max(A_generated))
print("B_generated range:", np.min(B_generated), np.max(B_generated))

# reconstruct images
A_reconstructed = g_model_AtoB.predict(A_generated)
B_reconstructed = g_model_BtoA.predict(B_generated)

# print shapes
print("A_reconstructed shape:", A_reconstructed.shape)
print("B_reconstructed shape:", B_reconstructed.shape)

show_plot_domains(B_real, A_generated, A_reconstructed,
                  A_real, B_generated, B_reconstructed,
                  save_path=None)