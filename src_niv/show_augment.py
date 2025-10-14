import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./data_read_code')
from src_niv.prep_data import data_ops
from src_niv.read_lf5_data import process_subject
from src_niv.utils import display_pred, load_and_preprocess_hf, load_and_preprocess_lf, visualize_hf_slices,padding_LF, visualize_lf_slices,rotate_slices, visualize_resampled, resample_volume, visualize_planes,visualize_pair, normalize_volume
from src_niv.prep_lf import normalize, resize_mri_volume
from src_niv.augment import srr_generator
from src_niv.zssr import  zero_shot_super_resolution, extract_brain, extract_lf_volumes
from src_niv.niv_srr_main import train
from src_niv.models.ResUNet import residual_srr_unet, residual_att_unet_3d
from src_niv.models.DenseUNet import build_dense_unet_3d
from src_niv.models.Inception import build_inception_unet_3d
from demo_read_data import read_lf_data
from src_niv.prep_lf import register_to_hf

import os
import cv2
import nibabel as nib
from nilearn import plotting
from nibabel.viewers import OrthoSlicer3D
import tensorflow as tf
import pydicom
import numpy as np
import matplotlib
matplotlib.use('tkagg')  # or 'Qt5Agg' depending on what's installed
import matplotlib.pyplot as plt
import ants
import nibabel as nib
from sklearn.feature_extraction import image
import os
from skimage.transform import resize  # Required for 3D resizing

# 59228
subjects1 = ['26184'] # '59877', '59175','59233', '59877', '35547'

# Define the path to the IRF_3T folder (High Field Data)
nhp_base_path = './Data/IRF_3T'
day_idx = 2
visualize = False
visualize_pairs = False
padding = False
register2_hf = True
augmentation = True

# Training parameters
steps_per_epoch = 60
epochs = 1500
batch_size = 2

# Training data

lf_input_volume_combined = []
hf_input_volume_combined = []
hf_target_volume_combined = []

for subject in subjects1:
    for day_idx in [1]:  # Assuming 0 = Day 1, 1 = Day 2
        print(f"\n=============================== Processing subject: {subject}, Day: {day_idx + 1} ===============================")
        # ----- Load HF data -----
        print(f"\n=============================== HF_MRI data processing started .............")
        resampled_volume_hf_norm = load_and_preprocess_hf(subject, day_idx, visualize)
        # ----- Load LF data -----
        print(f"\n=============================== LF_MRI data processing started .............")
        resampled_volume_lf_be_norm = load_and_preprocess_lf(subject, day_idx, visualize)
        print("Resampled LF volume shape:", resampled_volume_lf_be_norm.shape)
        print("Resampled HF volume shape:", resampled_volume_hf_norm.shape)
        # Register LF to HF
        if register2_hf:
            resampled_volume_lf_be_norm = register_to_hf(resampled_volume_lf_be_norm, resampled_volume_hf_norm)
        # Padding
        if padding:
            resampled_volume_lf_be_norm = padding_LF(
                resampled_volume_lf_be_norm,
                resampled_volume_hf_norm,
                target_slices=64
            )
            print("After padding LF volume shape:", resampled_volume_lf_be_norm.shape)
        # Visualization (optional)
        if visualize_pairs:
            visualize_pair(
                resampled_volume_lf_be_norm,
                resampled_volume_hf_norm,
                slice_indices=list(range(31))
            )
        # ----- Final Volume Preparation -----
        lf_input_volume = resampled_volume_lf_be_norm.astype(np.float32)
        hf_input_volume = resampled_volume_hf_norm.astype(np.float32)
        hf_target_volume = resampled_volume_hf_norm.astype(np.float32)
        # Slice selection
        lf_input_volume = lf_input_volume[0:32, :, :]
        hf_input_volume = hf_input_volume[0:32, :, :]
        hf_target_volume = hf_target_volume[0:32, :, :]

        print("LF Input shape:", lf_input_volume.shape)
        print("HF Input shape:", hf_input_volume.shape)
        print("HF volume shape:", hf_target_volume.shape)
        # ----- Append to Combined Lists -----
        lf_input_volume_combined.append(lf_input_volume)
        hf_input_volume_combined.append(hf_input_volume)
        hf_target_volume_combined.append(hf_target_volume)

lf_input_volume_combined = np.stack(lf_input_volume_combined)
hf_input_volume_combined = np.stack(hf_input_volume_combined)
hf_target_volume_combined = np.stack(hf_target_volume_combined)

print("LF Input shape:", lf_input_volume_combined.shape)
print("HF Input shape:", hf_input_volume_combined.shape)
print("HF volume shape:", hf_target_volume_combined.shape)

train_gen = srr_generator(lf_input_volume, hf_target_volume, batch_size=1, patch_z=32, patch_xy=128, augment=True, extra_slices=50, noise_sigma=0.03)
lf_input, hf_target = next(train_gen)
print(lf_input.shape)  # (2, 32, 128, 128, 1)
print(hf_target.shape)  # (2, 32, 128, 128, 1)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import rotate, affine_transform
import random
import os

# ---------------------------------------------
# Create output directory
# ---------------------------------------------
output_dir = "LF_Augmentations"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------
# Augmentation utility functions
# ---------------------------------------------
def rotate2d(image, angle):
    """Rotate a 2D image by a given angle."""
    return rotate(image, angle, reshape=False, mode='reflect')

def shear2d(image, shear_x=0.1, shear_y=0.0):
    """Apply simple affine shear transform."""
    transform_matrix = np.array([[1, shear_x, 0],
                                 [shear_y, 1, 0],
                                 [0, 0, 1]])
    return affine_transform(image, transform_matrix, mode='reflect')

def add_gaussian_noise(image, sigma=0.03):
    """Add Gaussian noise."""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 1)

def adjust_intensity(image, factor_range=(0.85, 1.15)):
    """Randomly adjust intensity."""
    factor = random.uniform(*factor_range)
    adjusted = image * factor
    return np.clip(adjusted, 0, 1)

# ---------------------------------------------
# Example LF slice
# ---------------------------------------------
# Assuming you already have `lf_input_volume_combined`
lf_slice = lf_input_volume_combined[0, 4, :, :]
# lf_slice = (lf_slice - lf_slice.min()) / (lf_slice.max() - lf_slice.min())  # Normalize to [0,1]

# ---------------------------------------------
# Apply augmentations
# ---------------------------------------------
augmentations = {
    "Original Slice": lf_slice,
    "Rotation +10°": rotate2d(lf_slice, 10),
    "Rotation +15°": rotate2d(lf_slice, 15),
    "Rotation −20°": rotate2d(lf_slice, -20),
    "Gaussian Noise (σ=0.05)": add_gaussian_noise(lf_slice, sigma=0.05),
    "Intensity Adjustment": adjust_intensity(lf_slice, factor_range=(0.9, 1.1))
}

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------
# Save individual figures with title written on image
# ---------------------------------------------
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ---------------------------------------------
# Save individual figures with small title at top
# ---------------------------------------------
for name, img in augmentations.items():
    # Normalize and convert image to uint8 grayscale
    img_uint8 = (255 * (img - np.min(img)) / (np.ptp(img) + 1e-8)).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    # Convert to PIL for text overlay
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Font setup (smaller size)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 8)
    except:
        font = ImageFont.load_default()

    text = name

    # Measure text size (Pillow 10+ compatible)
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        text_width, text_height = draw.textsize(text, font=font)

    # Position text at top center with a small margin
    text_x = (img_rgb.shape[1] - text_width) // 2
    text_y = 3

    # Optional: draw subtle dark background for visibility
    draw.rectangle([(0, 0), (img_rgb.shape[1], text_height + 6)], fill=(0, 0, 0))

    # Draw title text in white
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

    # Convert back to NumPy and save
    img_with_title = np.array(pil_img)
    save_name = f"{name.replace('°', 'deg').replace(' ', '_').replace('(', '').replace(')', '').replace('−', '-')}.png"
    save_path = os.path.join(output_dir, save_name)
    cv2.imwrite(save_path, cv2.cvtColor(img_with_title, cv2.COLOR_RGB2BGR))

print("✅ All augmented images saved with small titles on top.")

# ---------------------------------------------
# Combined visualization for publication
# ---------------------------------------------
titles = list(augmentations.keys())
images = list(augmentations.values())

plt.figure(figsize=(22, 5))
for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, len(images), i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title, fontsize=12)
    plt.axis('off')

plt.suptitle("Augmentation Visualization for LF-MRI Slice", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "LF_Augmentation_Panel.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ All individual augmented images and combined panel saved in: {output_dir}")