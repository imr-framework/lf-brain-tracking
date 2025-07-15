import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
from nifti_write import make_nifti

# Load input and first stage recon
write_nifti = True
fname_lowres = '/home/mri4all/Documents/Tools/SRR/Data/MW/IRF_NHP/59877/NHP_59877.npy'
fname_srr = './srr_1cycle_2mm_Huber.nii.gz'


srr_1 = nib.load(fname_srr).get_fdata()

# low_res_im = np.load(fname_lowres)
# print(low_res_im.shape)
# srr_1 = np.load(fname_srr)
# print(srr_1.shape)

# Visualize input and SRR - 1st stage
# s0 = OrthoSlicer3D(low_res_im)
# s0.clim = [0, 128]
# s0.show()

s1 = OrthoSlicer3D(srr_1)
# s1.clim = [0, 1.25]
s1.show()

