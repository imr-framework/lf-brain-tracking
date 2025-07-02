import sys
import numpy as np
import nibabel.nifti1 as nib
import matplotlib.pyplot as plt
import cv2


# TODO:
# 1. Improve masking
# 2. Better visualization
# 3. Fine tune recon input

def make_nifti(data: np.ndarray = 0, fname: str = 'test.nii.gz', mask: bool = False,
               res: np.ndarray = [6, 2, 2], dim_info: np.ndarray = [1, 2, 0]):
  if mask is False:
    fact = 2047
    data_norm = norm_data(data, fact)
    ni_img = nib.Nifti1Image(data_norm, affine=np.eye(4))
    header = ni_img.header
    header_new = create_nifti_header(header, dtype='raw', res= res, dim_info=dim_info)
    ni_img = nib.Nifti1Image(data_norm, affine=np.eye(4), header=header_new)
  else:
      ni_img = nib.Nifti1Image(data.astype(float), affine=np.eye(4), dtype=np.int16)
      header = ni_img.header
      header_new = create_nifti_header(header, dtype='mask', res= res, dim_info=dim_info)
      ni_img = nib.Nifti1Image(data, affine=np.eye(4), header=header_new)
  nib.save(ni_img, fname)
  return ni_img


def create_nifti_header(header: dict = 0, dtype: str='raw',
                        res: np.ndarray = [6, 2, 2], dim_info: np.ndarray = [1, 2, 0]):

    header_new = header
    header_new["xyzt_units"] = 2
    header_new.set_dim_info(phase=dim_info[0], freq=dim_info[1], slice=dim_info[2])  # Phase, Frequency, Slice
    header_new["pixdim"] = [1, res[0], res[1], res[2], 1, 1, 1, 1]

    if dtype == 'raw':
      header_new["data_type"] = 'compat'  # risky?
      header_new["cal_max"] = 2048.0
      header_new["cal_min"] = 0  # header_info['pixdim'][1:4]  = [2,2,2]
    elif dtype == 'mask':
      header_new["data_type"] = 'mask'  # risky?
      header_new["cal_max"] = 1
      header_new["cal_min"] = 0  # header_info['pixdim'][1:4]  = [2,2,2]
    return header_new

def norm_data(data, fact):
    data_new = (data - np.min(data))/(np.max(data) - np.min(data))
    data_new = data_new * fact
    return data_new


def get_mask_thresh(data: np.ndarray = 0, patch_sz: np.ndarray=None):
    if patch_sz is None:
      patch_sz = [8, 8]

    dim1 = patch_sz[0] - 1
    dim2 = patch_sz[1] - 1
    nx, ny, nz = data.shape
    corner1 = np.squeeze(data[:, 0:dim1, 0:dim2])
    corner2 = np.squeeze(data[:, ny-dim1:ny, 0:dim2])
    corner3 = np.squeeze(data[:, 0:dim1, nz-dim1:nz])
    corner4 = np.squeeze(data[:, ny-dim1:ny, nz - dim1:nz])

    thresh = np.mean(corner1 + corner2 + corner3 + corner4)
    return thresh

def do_resize(im_data: np.ndarray = 0, dim:int =1):
    im_data_new = np.zeros([dim, dim, dim], dtype=float)
    nx, ny, nz = im_data.shape
    n_idx = np.argmin([nx, ny, nz])

    if n_idx == 0:    # axial
      for z in range(nz):
        im_data_new[:, :, z] = cv2.resize(np.squeeze(im_data[:, :, z]), [dim, dim])

    if n_idx == 1:    # sag
      for z in range(nz):
        im_data_new[:, :, z] = cv2.resize(np.squeeze(im_data[:, :, z]), [dim, dim])

    if n_idx == 2:    # cor
      for z in range(nx):
        im_data_new[z, :, :] = cv2.resize(np.squeeze(im_data[z, :, :]), [dim, dim])

    return im_data_new



