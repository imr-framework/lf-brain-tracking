import os.path
import sys
# sys.path.append('/Users/sairamgeethanath/Documents/Projects/Tools/Low_field/Propsa/Recon/Code2PP/')
# sys.pathpip n.insert(0, '/Users/sairamgeethanath/Documents/Projects/Tools/Low_field/Propsa/Recon/Code2PP/')
# import cProcessPipeline as cPP
# from sim_input_SR import create_object, down_res
from display_vlf_ni_data import plot_anatomy_raw, plot_anatomy_nifti
# import niftyreg as nreg
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from preprocess4srr import non_local_means_denoising


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
                        res: np.ndarray = None, dim_info: np.ndarray = None):

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
    im_data_new = np.zeros([dim[0], dim[1], dim[2]], dtype=float)
    nx, ny, nz = im_data.shape
    n_idx = np.argmin([nx, ny, nz])

    if n_idx == 0:    # axial
      for z in range(nz):
        im_data_new[:, :, z] = cv2.resize(np.squeeze(im_data[:, :, z]), [dim[1], dim[0]])   #  width, height

    if n_idx == 1:    # cor
      for x in range(nx):
        im_data_new[x, :, :] = cv2.resize(np.squeeze(im_data[x, :, :]), [dim[0], dim[1]])

    if n_idx == 2:    # sag
      for z in range(nx):
        im_data_new[z, :, :] = cv2.resize(np.squeeze(im_data[z, :, :]), [dim[2], dim[1]])

    return im_data_new

def pad_zeros(im_data):
  nx, ny, nz = im_data.shape
  k = np.argmax([nx, ny, nz])
  if k == 0:
    nx_diff = 0
    ny_diff = int(0.5*(nx - ny))
    nz_diff = int(0.5*(nx - nz))
  elif k == 1:
    nx_diff = int(0.5*(ny - nx))
    ny_diff = 0
    nz_diff = int(0.5*(ny - nz))
  elif k == 2:
    nx_diff = int(0.5*(nz - nx))
    ny_diff = int(0.5*(nz - ny))
    nz_diff = 0

  im_data_new = np.pad(im_data, ((nx_diff, nx_diff), (ny_diff, ny_diff), (nz_diff, nz_diff)),
                       constant_values=(0, 0))
  return im_data_new




if __name__ == '__main__':
    mode = 'scanner'

    if mode == 'scanner':
      dataFolder = r'./Niftymic_related/Data/In_vivo/2_avg'
      im_yz_folder = 'axial_circshift_1yz.npy'  # axial
      im_yx_folder = 'sagittal_circshift_1yx.npy'  # sagittal
      im_zx_folder = 'coronal_circshift_1zx.npy'  # coronal

      im_axial = np.abs(np.load(os.path.join(dataFolder, im_yz_folder)))
      im_sag = np.abs(np.load(os.path.join(dataFolder, im_yx_folder)))
      im_cor = np.abs(np.load(os.path.join(dataFolder, im_zx_folder)))

      im_axial = np.moveaxis(im_axial, [0, 1, 2], [1, 2, 0])  # yzx --> xyz
      im_sag = np.moveaxis(im_sag, [0, 1, 2], [1, 0, 2])  # yxz --> xyz
      im_cor = np.moveaxis(im_cor, [0, 1, 2], [2, 0, 1])  # zxy --> xyz

      # denoise these images first
      im_axial = non_local_means_denoising(im_axial)
      im_sag = non_local_means_denoising(im_sag)
      im_cor = non_local_means_denoising(im_cor)

      # do pre-processing on data - resize to same sizes
      im_axial = do_resize(im_data=im_axial, dim=[80, 110, 110])
      im_axial = pad_zeros(im_axial)
      plot_anatomy_raw(im_axial,clim=[0, 1500])

      im_sag = do_resize(im_data=im_sag, dim=[110, 110, 80])
      im_sag = pad_zeros(im_sag)
      plot_anatomy_raw(im_sag,clim=[0, 1500])

      im_cor = do_resize(im_data=im_cor, dim=[110, 80, 110])
      im_cor = pad_zeros(im_cor)
      plot_anatomy_raw(im_cor, clim=[0, 1500])

    # Get thresh for mask
    thresh = get_mask_thresh(data=im_axial, patch_sz=[8, 8])
    # thresh = 0
    # Write axial data to nifti file
    make_nifti(data=im_axial, fname='Niftymic_related/axial_redo.nii.gz', mask=False,
               res=[2, 2, 2], dim_info=[0, 1, 2]) # phase, freq, slice - [2, 1, 0]


    axial_mask = im_axial > thresh
    plot_anatomy_raw(axial_mask, clim=[0, 2])
    make_nifti(data=axial_mask, fname='Niftymic_related/axial_mask_redo.nii.gz', mask=True,
               res=[2, 2, 2], dim_info=[0, 1, 2])  # phase, freq, slice - 2, 1, 0
    



    # Write sagittal data to nifti file
    make_nifti(data=im_sag, fname='Niftymic_related/sag_redo.nii.gz', mask=False,
               res=[2, 2, 2], dim_info=[0, 1, 2])  # phase, freq, slice
    plot_anatomy_nifti('Niftymic_related/sag_redo.nii.gz')

    sag_mask = im_sag > thresh
    plot_anatomy_raw(sag_mask, clim=[0, 2])
    make_nifti(data=sag_mask, fname='Niftymic_related/sag_mask_redo.nii.gz', mask=True,
               res=[2, 2, 2], dim_info=[0, 1, 2])  # phase, freq, slice

    # Write coronal data to nifti file
    make_nifti(data=im_cor, fname='Niftymic_related/cor_redo.nii.gz', mask=False,
               res=[2, 2, 2], dim_info=[0, 1, 2])  # phase, freq, slice - [0, 2, 1]
    plot_anatomy_nifti('Niftymic_related/cor_redo.nii.gz')

    cor_mask = im_cor > thresh
    plot_anatomy_raw(cor_mask, clim=[0, 2])
    make_nifti(data=cor_mask, fname='Niftymic_related/cor_mask_redo.nii.gz', mask=True,
               res=[2, 2, 2], dim_info=[0, 1, 2])  # phase, freq, slice


    
    debug = True
    if debug is True:
      


      im_sr = nib.load('Data/MW/VLF_invivo/niftymic_outputs_ajay/srr_1cycle_2mm_Huber_retest_v3.nii.gz')
      data = im_sr.get_fdata()
      plot_anatomy_raw(data, clim=[0, 1500])

      header = im_sr.header
      print(header)

      # # im_sr_data = im_sr.dataobj / np.max(im_sr.dataobj)
      # im4 = np.squeeze(axial_mask[:, :, 55])
      # # fig = plt.figure()
      # plt.subplot(131)
      # plt.imshow(np.abs(im4), cmap='gray')
      # # plt.show()
      # # ax.set_aspect('equal')
      #
      # im4 = np.squeeze(sag_mask[:, :, 55])
      # plt.subplot(132)
      # plt.imshow(np.abs(im4), cmap='gray')
      # # plt.show()
      # # ax.set_aspect('equal')
      #
      #
      # im4 = np.squeeze(cor_mask[:, :, 10])
      # plt.subplot(133)
      # plt.imshow(np.abs(im4), cmap='gray')
      # plt.show()
      # # ax.set_aspect('equal')
