import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
from nifti_write import make_nifti
# Load input and first stage recon
write_nifti = False
# nii_name = './Data/MW/VLF_invivo/reoriented_skull_stripped_raw_srr7_reoriented.nii.gz'
nii_name = './Data/MW/IRF_NHP/Selected_to_process/Processed/35528_D56_nce_post/nhp_35528_D_56.nii.gz'
low_res_im = nib.load(nii_name).get_fdata()
print(low_res_im.shape)
nii_srr_name = './Data/MW/IRF_NHP/Selected_to_process/Processed/35528_D56_nce_post/nhp_35528_D_56ZSSR_2D_ms_srr4.nii.gz'
# srr_1 = np.load('./output_volume_VLF_invivo.npy')
srr_1 = nib.load(nii_srr_name).get_fdata()

# Visualize input and SRR - 1st stage
s0 = OrthoSlicer3D(low_res_im)
s0.clim = [0, 2048]
s0.show()

s1 = OrthoSlicer3D(srr_1)
s1.clim = [0, 2048]
s1.show()

if write_nifti is True:
    make_nifti(data=srr_1, fname='zssr_invivo_112mm.nii.gz', mask=False,
               res=[1, 1, 2], dim_info=[2, 1, 0]) # phase, freq, slice

