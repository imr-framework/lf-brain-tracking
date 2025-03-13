import sys
import numpy as np
import nibabel.nifti1 as nib
import matplotlib.pyplot as plt



# TODO:
# 1. Improve masking
# 2. Better visualization
# 3. Fine tune recon input

def make_nifti(data: np.ndarray = 0, fname: str = 'test.nii.gz', mask: bool = False,
               res: np.ndarray = [6, 2, 2], dim_info: np.ndarray = [1, 2, 0], norm:bool=False):
  if mask is False:
    if norm is True:
      fact = 2047
      data = norm_data(data, fact)
    ni_img = nib.Nifti1Image(data, affine=np.eye(4))
    header = ni_img.header
    header_new = create_nifti_header(header, dtype='raw', res= res, dim_info=dim_info)
    ni_img = nib.Nifti1Image(data, affine=np.eye(4), header=header_new)
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
    return header

def norm_data(data, fact):
    data_new = (data - np.min(data))/(np.max(data) - np.min(data))
    data_new = data_new*fact
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



if __name__ == '__main__':
    mode = 'scanner'

    if mode == 'scanner':
      dataFolder = r'./Data/Phantom'
      im_zy_folder = 'baseline_zy'  # axial
      im_yx_folder = 'baseline_xy'  # coronal
      im_xz_folder = 'baseline_zx'  # sagittal

      ''' Acq params '''
      noiseScan = 0
      freqDrift = -300

      im_axial, _, _, _ = cPP.prepData(dataFolder, im_zy_folder, noiseScan, freqDrift)
      im_cor, _, _, _ = cPP.prepData(dataFolder, im_yx_folder, noiseScan, freqDrift)
      im_sag, _, _, _ = cPP.prepData(dataFolder, im_xz_folder, noiseScan, freqDrift)

      # align all to x, y and z
      im_axial = np.moveaxis(im_axial, [0, 2], [2, 0])  # yzx --> xyz
      im_cor = np.moveaxis(im_cor, [0, 1], [1, 0])  # yxz --> xyz
      im_sag = np.moveaxis(im_sag, [0, 1,  2], [2, 0, 1])  # zxy --> xyz

      im_axial = np.abs(im_axial)
      im_sag = np.abs(im_sag)
      im_cor = np.abs(im_cor)

      # Save original files as niftis for visualizaing later
      make_nifti(data=im_axial, fname='axial_org.nii.gz', mask=False,
                 res=[6, 2, 2], dim_info=[2, 1, 0])  # phase, freq, slice
      make_nifti(data=im_sag, fname='sag_org.nii.gz', mask=False,
                 res=[2, 6, 2], dim_info=[1, 2, 0])  # phase, freq, slice
      make_nifti(data=im_cor, fname='cor_org.nii.gz', mask=False,
                 res=[2, 2, 6], dim_info=[0, 1, 2])  # phase, freq, slice


      # do pre-processing on data - resize to same sizes
      im_axial = do_resize(im_data=im_axial, dim=110)
      plot_anatomy_raw(im_axial)

      im_sag = do_resize(im_data=im_sag, dim=110)
      plot_anatomy_raw(im_sag)

      im_cor = do_resize(im_data=im_cor, dim=110)
      plot_anatomy_raw(im_cor)

    elif mode == 'sim':
      FOVx, FOVy, FOVz = 160.0, 220.0, 220.0
      object_im = create_object(res=2, FOV=[FOVx, FOVy, FOVz], phantom='SL3D')
      im_axial = down_res(object_im, res=2, dres=6, axis_dres=0)  # case axial = X
      im_sag = down_res(object_im, res=2, dres=6, axis_dres=1)  # case sag = Y
      im_cor = down_res(object_im, res=2, dres=6, axis_dres=2)  # case cor = Z

    # Get thresh for mask
    thresh = get_mask_thresh(data=im_axial, patch_sz=[8, 8])
    # thresh = 0
    # Write axial data to nifti file
    make_nifti(data=im_axial, fname='axial.nii.gz', mask=False,
               res=[2, 2, 2], dim_info=[2, 1, 0]) # phase, freq, slice
    axial_mask = im_axial > thresh
    make_nifti(data=axial_mask, fname='axial_mask.nii.gz', mask=True,
               res=[2, 2, 2], dim_info=[2, 1, 0])  # phase, freq, slice

    # Write sagittal data to nifti file
    make_nifti(data=im_sag, fname='sag.nii.gz', mask=False,
               res=[2, 2, 2], dim_info=[0, 2, 1])  # phase, freq, slice
    sag_mask = im_sag > thresh
    make_nifti(data=sag_mask, fname='sag_mask.nii.gz', mask=True,
               res=[2, 2, 2], dim_info=[0, 2, 1])  # phase, freq, slice

    # Write coronal data to nifti file
    make_nifti(data=im_cor, fname='cor.nii.gz', mask=False,
               res=[2, 2, 2], dim_info=[0, 1, 2])  # phase, freq, slice
    cor_mask = im_cor > thresh
    make_nifti(data=cor_mask, fname='cor_mask.nii.gz', mask=True,
               res=[2, 2, 2], dim_info=[0, 1, 2])  # phase, freq, slice

    debug = False
    if debug is True:
      
      im_sr = nib.load('srr2.nii.gz')
      data = im_sr.get_fdata()
      plot_anatomy_raw(data)

      header = im_sr.header
      print(header)


