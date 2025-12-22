import os
import nibabel as nib
import numpy as np
from degrade import degrade_to_lf
# define zoom
from scipy.ndimage import zoom

HF_PATH = "Data/HF-T2/T2/"
LF_PATH = "Data/LF-T2/"

# def normalize(vol):
#     p1, p99 = np.percentile(vol, (1, 99))
#     vol = np.clip(vol, p1, p99)
#     vol = (vol - p1) / (p99 - p1)
#     return vol * 2 - 1

def normalize(vol):
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
    return vol * 2 - 1

# FUNCTION TO TAKE HF_VOLUME  AND RESAMPLE TO 1,1,2 MM AND SHOW SHAPE AND IF H, W NOT 128, 128, CROP CENTER 128, 128

def resample_to_1mm(vol, original_voxel_size, target_voxel_size=(1, 1, 2)):
    # Resample the volume to 1x1x2 mm
    # print("Original shape:", vol.shape)
    # print("Original voxel size:", original_voxel_size)
    zoom_factors = tuple(o / t for o, t in zip(original_voxel_size, target_voxel_size))
    vol = zoom(vol, zoom_factors, order=1)
    # print("Resampled shape:", vol.shape)
    return vol

def crop_center(vol, target_shape=(128, 128, None)):
    h, w, d = vol.shape
    th, tw, td = target_shape
    if td is None:
        td = d
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    start_d = (d - td) // 2
    return vol[start_h:start_h+th, start_w:start_w+tw, start_d:start_d+td]

# Visualization funstion to check slices
def visualize_slices(vol, title="Volume Slices"):
    import matplotlib.pyplot as plt
    num_slices = vol.shape[2]
    fig, axes = plt.subplots(1, min(num_slices, 5), figsize=(15, 5))
    for i in range(min(num_slices, 5)):
        axes[i].imshow(vol[:, :, i * (num_slices // 5)], cmap='gray')
        axes[i].axis('off')
    plt.suptitle(title)
    plt.show()

# Visualize all slices in a volume
def visualize_all_slices(vol, title="All Volume Slices", overlap=0.001):
    import matplotlib.pyplot as plt
    num_slices = vol.shape[2]
    cols = 5
    rows = (num_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    for i in range(num_slices):
        ax = axes[i // cols, i % cols]
        # Add overlap by blending with adjacent slices if possible
        img = vol[:, :, i].copy()
        if i > 0:
            img = (1 - overlap) * img + overlap * vol[:, :, i - 1]
        if i < num_slices - 1:
            img = (1 - overlap) * img + overlap * vol[:, :, i + 1]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    for j in range(i + 1, rows * cols):
        axes[j // cols, j % cols].axis('off')
    plt.suptitle(title)
    plt.show()

def load_volumes(visualize=False):
    hf_files = sorted(os.listdir(HF_PATH))
    for f in hf_files:
        hf = nib.load(HF_PATH + f).get_fdata()
        # read header to get original voxel size
        hdr = nib.load(HF_PATH + f).header
        original_voxel_size = hdr.get_zooms()   
        hf = normalize(hf)
        # rotate by 90 degrees counter-clockwise in-plane Different direction than previously
        hf = np.rot90(hf, k=3, axes=(0, 1))
        # Visualize original hf slices
        if visualize:
            # print("Original shape:", hf.shape)
            visualize_slices(hf, title="Original HF Slices")
        hf = resample_to_1mm(hf, original_voxel_size, target_voxel_size=(1,1,2))
        # Visualize resampled hf slices
        if visualize:
            # print("Resampled shape:", hf.shape)
            visualize_slices(hf, title="Resampled HF Slices")

        lf = degrade_to_lf(hf)
        # Visualize lf slices
        if visualize:
            visualize_slices(lf, title="Degraded LF Slices")

        hf = crop_center(hf, target_shape=(128, 128, 35))
        lf = crop_center(lf, target_shape=(128, 128, 35))

        # print("Cropped HF shape:", hf.shape)
        # print("Cropped LF shape:", lf.shape)

        if visualize:
            visualize_all_slices(hf, title="Cropped HF Slices")
            visualize_all_slices(lf, title="Cropped LF Slices")
        # take first 30 slices only
        hf = hf[:, :, :30]
        lf = lf[:, :, :30]
        # print("Final HF shape:", hf.shape)
        # print("Final LF shape:", lf.shape)
        if visualize:
            visualize_all_slices(hf, title="Final HF Slices")
            visualize_all_slices(lf, title="Final LF Slices")

        yield hf, lf

def slice_2p5d(vol, idx):
    return np.stack([
        vol[:, :, idx-1],
        vol[:, :, idx],
        vol[:, :, idx+1]
    ], axis=-1)

def dataset_generator():
    for hf, lf in load_volumes():
        for i in range(1, hf.shape[2]-1):
            yield slice_2p5d(hf, i), slice_2p5d(lf, i)

# Test the dataset generator
if __name__ == "__main__":
    # for hf, lf in dataset_generator():
    #     print("HF slice shape:", hf.shape)
    #     print("LF slice shape:", lf.shape)
    #     break

    # To visualize the processing pipeline, call load_volumes with visualize=True
    print("Visualizing volume processing steps...")
    for _ in load_volumes(visualize=True):
        break