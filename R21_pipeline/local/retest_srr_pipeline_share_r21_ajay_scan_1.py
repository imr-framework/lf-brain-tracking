import os
import sys
import subprocess
import numpy as np
import nibabel as nib
import ants

sys.path.insert(0, './')
sys.path.append('./src')

from display_vlf_ni_data import plot_anatomy_raw 
from preprocess4srr import non_local_means_denoising
from prep4srr_2step_v2 import make_nifti, pad_zeros, do_resize

# ========
# SETTINGS
# ========
niftymic = False
visualize = True

dataFolder = r'R21_pipeline/DataSRR_AJ/ajay_scan_1/npy1'
subjectID = 'sub_0011'

outputfolder = os.path.join(
    r'R21_pipeline/DataSRR_AJ/ajay_scan_1/Outputs',
    subjectID
)

os.makedirs(outputfolder, exist_ok=True)

# =========================
# LOAD DATA
# =========================
im_axial = np.abs(np.load(os.path.join(dataFolder, 'axial_circshift_1zy.npy')))
im_sag   = np.abs(np.load(os.path.join(dataFolder, 'sagittal_circshift_1yx.npy')))
im_cor   = np.abs(np.load(os.path.join(dataFolder, 'coronal_circshift_1zx.npy')))

# =========================
# REORIENT TO XYZ
# =========================
im_axial = np.moveaxis(im_axial, [0,1,2], [1,0,2])  # yzx → xyz

im_sag   = np.moveaxis(im_sag,   [0,1,2], [0,2,1])  # yxz → xyz
im_sag   = np.moveaxis(im_sag,   [0,1,2], [1,0,2])  # yxz → xyz

im_cor   = np.moveaxis(im_cor,   [0,1,2], [0,2,1])  # zxy → xyz

# =========================
# DENOISING
# =========================
im_axial = non_local_means_denoising(im_axial)
im_sag   = non_local_means_denoising(im_sag)
im_cor   = non_local_means_denoising(im_cor)

# =========================
# PAD ONLY (NO RESIZE)
# =========================
im_axial = do_resize(im_data=im_axial, dim=[110, 110, 80])
im_axial = pad_zeros(im_axial)


im_sag = do_resize(im_data=im_sag, dim=[80, 110, 110])
im_sag = pad_zeros(im_sag)

im_cor = do_resize(im_data=im_cor, dim=[110, 80, 110])
im_cor = pad_zeros(im_cor)

# =========================
# SAVE NIFTI (CRITICAL FIX)
# =========================

axial_path = os.path.join(outputfolder, 'axial_redo.nii.gz')
sag_path   = os.path.join(outputfolder, 'sag_redo.nii.gz')
cor_path   = os.path.join(outputfolder, 'cor_redo.nii.gz')

# TRUE ACQUISITION SPACING (VERY IMPORTANT)
make_nifti(im_axial, axial_path, mask=False, res=[2,2,2], dim_info=[0,1,2])
make_nifti(im_sag,   sag_path,   mask=False, res=[2,2,2], dim_info=[2,0,1])
make_nifti(im_cor,   cor_path,   mask=False, res=[2,2,2], dim_info=[0,2,1])

# =========================
# SKULL STRIPPING / MASKS
# =========================

th_ax  = np.percentile(im_axial, 0)
th_sag = np.percentile(im_sag, 0)
th_cor = np.percentile(im_cor, 0)

ax_mask = (im_axial > th_ax).astype(np.uint8)
sg_mask = (im_sag > th_sag).astype(np.uint8)
cr_mask = (im_cor > th_cor).astype(np.uint8)

ax_mask_path = os.path.join(outputfolder, 'axial_mask_redo.nii.gz')
sg_mask_path = os.path.join(outputfolder, 'sag_mask_redo.nii.gz')
cr_mask_path = os.path.join(outputfolder, 'cor_mask_redo.nii.gz')

make_nifti(ax_mask, ax_mask_path, mask=True, res=[2,2,2], dim_info=[0,1,2])
make_nifti(sg_mask, sg_mask_path, mask=True, res=[2,2,2], dim_info=[2,0,1])
make_nifti(cr_mask, cr_mask_path, mask=True, res=[2,2,2], dim_info=[0,2,1])

print("Shapes after registration:")
print("Axial:", im_axial.shape)
print("Sag :", im_sag.shape)
print("Cor :", im_cor.shape)


# =========================
# VISUALIZATION
# =========================
if visualize:
    # plot_anatomy_raw(im_axial, clim=[0,2048])
    plot_anatomy_raw(im_axial, clim=[0,2048])
    # plot_anatomy_raw(ax_mask, clim=[0,2])
    # plot_anatomy_raw(im_sag,   clim=[0,2048])
    plot_anatomy_raw(im_sag,   clim=[0,2048])
    # plot_anatomy_raw(sg_mask,   clim=[0,2])
    # plot_anatomy_raw(im_cor,   clim=[0,2048])
    plot_anatomy_raw(im_cor,   clim=[0,2048])
    # plot_anatomy_raw(cr_mask,   clim=[0,2])

# axial_img = ants.image_read(axial_brain_path)
# sag_img   = ants.image_read(sag_brain_path)
# cor_img   = ants.image_read(cor_brain_path)

# axial_msk = ants.image_read(ax_mask_path)
# sag_msk   = ants.image_read(sg_mask_path)
# cor_msk   = ants.image_read(cr_mask_path)

# # =========================
# # REFERENCE = AXIAL
# # =========================

# fixed = axial_img

# # =========================
# # FUNCTION: RIGID REGISTRATION
# # =========================

# def rigid_register(fixed, moving):

#     # Rigid transform (good for orthogonal stacks)
#     reg = ants.registration(
#         fixed=fixed,
#         moving=moving,
#         type_of_transform='Rigid',   # IMPORTANT
#         verbose=True
#     )

#     warped = reg['warpedmovout']
#     transform = reg['fwdtransforms']

#     return warped, transform

# # =========================
# # REGISTER SAGITTAL → AXIAL
# # =========================

# sag_reg, sag_tf = rigid_register(fixed, sag_img)
# cor_reg, cor_tf = rigid_register(fixed, cor_img)

# # =========================
# # APPLY SAME TRANSFORMS TO MASKS
# # =========================

# sag_mask_reg = ants.apply_transforms(
#     fixed=fixed,
#     moving=sag_msk,
#     transformlist=sag_tf,
#     interpolator='nearestNeighbor'
# )

# cor_mask_reg = ants.apply_transforms(
#     fixed=fixed,
#     moving=cor_msk,
#     transformlist=cor_tf,
#     interpolator='nearestNeighbor'
# )

# # =========================
# # SAVE REGISTERED VOLUMES
# # =========================

# axial_reg_path = os.path.join(outputfolder, 'axial_reg.nii.gz')
# sag_reg_path   = os.path.join(outputfolder, 'sag_reg.nii.gz')
# cor_reg_path   = os.path.join(outputfolder, 'cor_reg.nii.gz')

# axial_mask_reg_path = os.path.join(outputfolder, 'axial_mask_reg.nii.gz')
# sag_mask_reg_path   = os.path.join(outputfolder, 'sag_mask_reg.nii.gz')
# cor_mask_reg_path   = os.path.join(outputfolder, 'cor_mask_reg.nii.gz')

# ants.image_write(axial_img, axial_reg_path)
# ants.image_write(sag_reg,   sag_reg_path)
# ants.image_write(cor_reg,   cor_reg_path)

# ants.image_write(axial_msk, axial_mask_reg_path)
# ants.image_write(sag_mask_reg, sag_mask_reg_path)
# ants.image_write(cor_mask_reg, cor_mask_reg_path)

# print("Registration completed and saved.")

# print("Shapes after registration:")
# print("Axial:", axial_img.shape)
# print("Sag :", sag_reg.shape)
# print("Cor :", cor_reg.shape)

# # =========================
# # NIFTYMIC OUTPUT
# # =========================

niftymic_output = os.path.join(
    outputfolder,
    'srr_output_2mm.nii.gz'
)

# =========================
# RUN NIFTYMIC
# =========================
if niftymic:

    print("Running NiftyMIC...")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.getcwd()}:{os.getcwd()}",
        "-w", os.getcwd(),
        "renbem/niftymic",
        "niftymic_reconstruct_volume",

        "--filenames",
        axial_path, cor_path, sag_path,

        "--filenames-masks",
        ax_mask_path, cr_mask_path, sg_mask_path,

        # "--filenames", 
        # axial_reg_path, cor_reg_path, sag_reg_path,
        # "--filenames-masks", 
        # axial_mask_reg_path, cor_mask_reg_path, sag_mask_reg_path,

        "--alpha", "0.02",
        "--outlier-rejection", "0",
        "--threshold-first", "0.5",
        "--threshold", "0.7",
        "--intensity-correction", "1",
        "--two-step-cycles", "1",
        "--isotropic-resolution", "2",
        "--reconstruction-type", "HuberL2",
        "--output", niftymic_output,
        "--verbose", "1"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)

    if result.returncode != 0:
        print("ERROR:")
        print(result.stderr)

# ===================
# LOAD RESULT
# ===================
if visualize:

    print("Loading SRR output...")

    im_srr = nib.load(niftymic_output).get_fdata()

    print("SRR shape:", im_srr.shape)

    plot_anatomy_raw(im_srr, clim=[0,2048])
