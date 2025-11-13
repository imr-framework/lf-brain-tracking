# This file reproduces the figure in the R21 application

# Import necessary modules
import os.path
import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./src')
sys.path.append('./Niftymic_related_r21')
# sys.path.append('/Users/sairamgeethanath/Documents/Projects/Tools/Low_field/Propsa/Recon/Code2PP/')
sys.path.insert(0, '/Users/sairamgeethanath/Documents/Projects/Tools/Low_field/Propsa/Recon/Code2PP/')
# import cProcessPipeline as cPP
# from sim_input_SR import create_object, down_res
from display_vlf_ni_data import plot_anatomy_raw, plot_anatomy_nifti
# import niftyreg as nreg
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from preprocess4srr import non_local_means_denoising
from prep4srr_2step_v2 import do_resize, make_nifti, create_nifti_header, norm_data, pad_zeros
from src.ZSSR_master import configs, configs_2, ZSSR
from src.ZSSR_master.ZSSR import *

# Read the data from the scanner and convert to .npy files
dataFolder = r'./Niftymic_related_r21/Data/In_vivo/2_avg'
im_yz_folder = 'axial_circshift_1yz.npy'  # axial
im_yx_folder = 'sagittal_circshift_1yx.npy'  # sagittal
im_zx_folder = 'coronal_circshift_1zx.npy'  # coronal

im_axial = np.abs(np.load(os.path.join(dataFolder, im_yz_folder)))
im_sag = np.abs(np.load(os.path.join(dataFolder, im_yx_folder)))
im_cor = np.abs(np.load(os.path.join(dataFolder, im_zx_folder)))

im_axial = np.moveaxis(im_axial, [0, 1, 2], [1, 2, 0])  # yzx --> xyz
im_sag = np.moveaxis(im_sag, [0, 1, 2], [1, 0, 2])  # yxz --> xyz
im_cor = np.moveaxis(im_cor, [0, 1, 2], [2, 0, 1])  # zxy --> xyz

# Preprocess the data (resizing and zero-padding)
# denoise these images first
im_axial = non_local_means_denoising(im_axial)
im_sag = non_local_means_denoising(im_sag)
im_cor = non_local_means_denoising(im_cor)

im_axial = do_resize(im_data=im_axial, dim=[80, 110, 110])
im_axial = pad_zeros(im_axial)
make_nifti(data=im_axial, fname='axial_redo.nii.gz', mask=False,
               res=[2, 2, 2], dim_info=[2, 1, 0]) # phase, freq, slice

im_sag = do_resize(im_data=im_sag, dim=[110, 110, 80])
im_sag = pad_zeros(im_sag)
make_nifti(data=im_sag, fname='sag_redo.nii.gz', mask=False,
               res=[2, 2, 2], dim_info=[0, 1, 2])  # phase, freq, slice

im_cor = do_resize(im_data=im_cor, dim=[110, 80, 110])
im_cor = pad_zeros(im_cor)
make_nifti(data=im_cor, fname='cor_redo.nii.gz', mask=False,
               res=[2, 2, 2], dim_info=[0, 2, 1])  # phase, freq, slice

# Save the nifti files for NiftyMIC processing via docker
thresh = 0
# ----- Axial -----
im_axial_mask = im_axial > thresh
make_nifti(data=im_axial_mask, fname='axial_mask_redo.nii.gz', mask=True,
               res=[2, 2, 2], dim_info=[2, 1, 0])  # phase, freq, slice

# Visualize the image and mask to confirm correctness
# plot_anatomy_raw(im_axial, clim=[0, 2048])

# ----- Sagittal -----
# im_sag = nib.load('/Users/sairamgeethanath/Documents/Projects/Tools/Low_field/OSI/Superresolution/super_resolution/Data/In_vivo/2_avg/circ-shifted/sag.nii.gz').get_fdata()
im_sag_mask = im_sag > thresh
make_nifti(data=im_sag_mask, fname='sag_mask_redo.nii.gz', mask=True,
               res=[2, 2, 2], dim_info=[2, 1, 0])  # phase, freq, slice

# Visualize the image and mask to confirm correctness
# plot_anatomy_raw(im_sag, clim=[0, 2048])

# ----- Coronal -----
# im_cor = nib.load('/Users/sairamgeethanath/Documents/Projects/Tools/Low_field/OSI/Superresolution/super_resolution/Data/In_vivo/2_avg/circ-shifted/cor.nii.gz').get_fdata()
im_cor_mask = im_cor > thresh
make_nifti(data=im_cor_mask, fname='cor_mask_redo.nii.gz', mask=True,
               res=[2, 2, 2], dim_info=[2, 1, 0])  # phase, freq, slice

# Visualize the image and mask to confirm correctness
plot_anatomy_raw(im_cor, clim=[0, 2048])

# # Run the docker NiftyMIC SRR command to generate SRR output
# print("Running NiftyMIC SRR via Docker...")
# docker_command = [
#     "docker", "run", "--rm",
#     "-v", f"{os.getcwd()}:{os.getcwd()}",
#     "-w", os.getcwd(),
#     "renbem/niftymic",
#     "niftymic_reconstruct_volume",
#     "--filenames", "axial_redo.nii.gz", "cor_redo.nii.gz", "sag_redo.nii.gz",
#     "--filenames-masks", "axial_mask_redo.nii.gz", "cor_mask_redo.nii.gz", "sag_mask_redo.nii.gz",
#     "--alpha", "0.02",
#     "--outlier-rejection", "0",
#     "--threshold-first", "0.5",
#     "--threshold", "0.7",
#     "--intensity-correction", "1",
#     "--two-step-cycles", "1",
#     "--isotropic-resolution", "2",
#     "--output", "srr_1cycle_2mm_Huber_test2.nii.gz",
#     "--verbose", "1",
#     "--reconstruction-type", "HuberL2"
# ]

# result = subprocess.run(docker_command, capture_output=True, text=True)
# print(result.stdout)
# print('Finished running')
# if result.returncode != 0:
#     print("Error:", result.stderr)

# Output: SRR
im_srr = nib.load('srr_1cycle_2mm_Huber_test2.nii.gz').get_fdata()
plot_anatomy_raw(im_srr, clim=[0, 2048])

# ZSSR implementation in all directions by factor of 2

# -------------------------------------------------------------
# Load input LF/HF-SRR volume
# -------------------------------------------------------------
print('Passing through ZSSR ..........')

img = nib.load('srr_1cycle_2mm_Huber_test2.nii.gz')
im_srr = img.get_fdata()
affine = img.affine

X, Y, Z = im_srr.shape
print("Input shape:", im_srr.shape)


# ============================================================
# 0. Slice-based ZSSR Runner
# ============================================================
def run_zssr_slice(slice2d, recon_config):
    """Run ZSSR on a single 2D slice with normalization."""
    minv, maxv = slice2d.min(), slice2d.max()
    s = (slice2d - minv) / (maxv - minv + 1e-8)
    s3 = np.stack([s, s, s], axis=-1)

    out = ZSSR(
        input_img=s3,
        conf=recon_config,
        ground_truth=None,
        kernels=None
    ).run()

    return out


# ============================================================
# 1. PASS x — Upscale z only
# ============================================================
def upscale_x(volume, recon_config):
    X, Y, Z = volume.shape
    recon_config.scale_factors = [[1, 2]] 

    out = np.zeros((X, Y, Z*2), dtype=float)

    for x in range(X):
        print(f"[Upscale X] Slice {x+1}/{X}")
        slice_yz = volume[x, :, :]        # shape (Y, Z)
        sr_slice = run_zssr_slice(slice_yz, recon_config)

        # replicate along the X dimension
        out[x, :, :] = sr_slice[:, :, 0]

    return out


# ============================================================
# 2. PASS y — Upscale x only
# ============================================================
def upscale_y(volume, recon_config):
    X, Y, Z = volume.shape
    recon_config.scale_factors = [[2, 1]]

    out = np.zeros((X*2, Y, Z), dtype=float)

    for y in range(Y):
        print(f"[Upscale Y] Slice {y+1}/{Y}")
        slice_xz = volume[:, y, :]      # shape (X, Z)
        sr_slice = run_zssr_slice(slice_xz, recon_config)

        out[:, y, :] = sr_slice[:, :, 0]

    return out

# ============================================================
# 3. PASS z — Upscale y only
# ============================================================
def upscale_z(volume, recon_config):
    X, Y, Z = volume.shape
    recon_config.scale_factors = [[1, 2]]  

    out = np.zeros((X, Y*2, Z), dtype=float)

    for z in range(Z):
        print(f"[Upscale Z] Slice {z+1}/{Z}")
        slice_xy = volume[:, :, z]      # shape (X, Y)
        sr_slice = run_zssr_slice(slice_xy, recon_config)

        out[:, :, z] = sr_slice[:, :, 0]

    return out

# ============================================================
# 4. FINAL PIPELINE  — X → Y → Z
# ============================================================
def run_xyz_progressive_zssr(im_srr, recon_config):
    print("\n==============================")
    print("Step 1: Upscaling X (×2)")
    print("==============================")
    vol_x2 = upscale_x(im_srr, recon_config)

    print("\n==============================")
    print("Step 2: Upscaling Y (×2)")
    print("==============================")
    vol_xy2 = upscale_y(vol_x2, recon_config)

    print("\n==============================")
    print("Step 3: Upscaling Z (×2)")
    print("==============================")
    vol_xyz2 = upscale_z(vol_xy2, recon_config)

    print("\nFinal SR Shape:", vol_xyz2.shape)
    return vol_xyz2

# ============================================================
# 5. RUN FULL PIPELINE
# ============================================================

# change of recon.config
recon_config = configs.Config()
# recon_config.scale_factors = [[np.sqrt(target_resolution_fact[0]), 1]]
recon_config.scale_factors = [[1, 2]]
recon_config.max_iters = 50
recon_config.min_iters = 20
recon_config.width = 32
recon_config.depth = 4
recon_config.noise_std = 0.0
recon_config.crop_size = 32
num_rows = 16
num_cols = 14

vol_xz = run_xyz_progressive_zssr(im_srr, recon_config)

# -------------------------------------------------------------
# Final 3D SR output
# -------------------------------------------------------------
im_srr_3d_zssr = vol_xz
print("\nFinal SR shape:", im_srr_3d_zssr.shape)

# -------------------------------------------------------------
# Save NIfTI with corrected voxel spacing
# (halving voxel spacing because resolution doubled)
# -------------------------------------------------------------
new_affine = affine.copy()
new_affine[:3, :3] /= 2   # voxel spacing becomes half = ×2 SR

out_img = nib.Nifti1Image(im_srr_3d_zssr.astype(np.float32), new_affine)
nib.save(out_img, "srr_zssr_3D_full.nii.gz")

print("\nSaved: srr_zssr_3D_full.nii.gz")
  
viewing = True

if viewing:
    # Choose middle slice index
    mid_x = im_srr.shape[0] // 2
    mid_y = im_srr.shape[1] // 2
    mid_z = im_srr.shape[2] // 2

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # 1. Original SRR middle axial slice
    axes[0].imshow(im_srr[:, :, mid_z], cmap='gray')
    axes[0].set_title('Original SRR (input)')
    axes[0].axis('off')

    # 2. Final 3D ZSSR axial slice
    axes[1].imshow(im_srr_3d_zssr[:, :, mid_z*2], cmap='gray')
    axes[1].set_title('3D ZSSR Output (final)')
    axes[1].axis('off')

    # 3. Sagittal comparison
    axes[2].imshow(im_srr[mid_x, :, :].T, cmap='gray')
    axes[2].set_title('Original Sagittal')
    axes[2].axis('off')

    axes[3].imshow(im_srr_3d_zssr[mid_x*2, :, :].T, cmap='gray')
    axes[3].set_title('ZSSR Sagittal')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()



