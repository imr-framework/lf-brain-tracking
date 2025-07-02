import numpy as np
import nibabel as nib
from read_kea3d import kea3d
from kea2nifti import make_nifti
from nibabel.viewers import OrthoSlicer3D

# List paths
data_folder = './demo_data/'
sub_folder = '3DTSE/1'
fname_nii = 'demo.nii'

sample_data = kea3d(data_folder=data_folder, sub_folder=sub_folder)
kspace = sample_data.kspace_gauss_filter
im = np.fft.fftshift(np.fft.fftn((np.fft.fftshift(kspace))))

s = OrthoSlicer3D(np.abs(im))
s.clim =[0, 2048]
s.cmap = 'gray'
s.show()

# Make nifti in case of need for further inputs to other software 
# Determine resolution from acq_par
make_nifti(im, fname = fname_nii, mask=False, res=[sample_data.res_dim1, sample_data.res_dim2, sample_data.res_dim3], dim_info=[0, 1, 2])

