# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16
@author: Sairam Geethanath
Input: 3 3D files with different resolutions
Purpose: convert to one high 3D res
Output: SR volume
TODO: display, PEP8 header and formatting
"""
# import cProcessPipeline as cPP
import numpy as np
# from roipoly import RoiPoly
from display_vlf_ni_data import plot_anatomy_raw
import matplotlib.pyplot as plt
import time
from sim_input_SR import create_object, down_res
from scipy.interpolate import RegularGridInterpolator
from scipy.misc import imresize


def get_SR_vol(im_axial, im_cor, im_sag, res, target_res=2, radius=4):
    # assumes in plane resolution is fine and only through plane needs interpolation  - im_yz
    res_in_plane = res[0]
    res_through_plane = res[1]

    im1 = np.squeeze(im_axial[12, :, :])
    im2 = np.squeeze(im_cor[55, :, :])
    im3 = np.squeeze(im_sag[40, :, :])

    plt.figure()
    plt.imshow(np.abs(im1), cmap='gray')
    plt.show()

    plt.imshow(np.abs(im2), cmap='gray')
    plt.show()

    plt.imshow(np.abs(im3), cmap='gray')
    plt.show()

    # Calculate 3D-FOV based on inputs for axial acquisition
    FOV_x = im_axial.shape[0] * res_through_plane
    FOV_y = im_axial.shape[1] * res_in_plane
    FOV_z = im_axial.shape[2] * res_in_plane

    # Get the data for each
    # axial

    coords_axial, intensities_axial = get_ortho_kernel(im_axial, xmin_ind=0, ymin_ind=0, zmin_ind=0,
                                                       xmax_ind=im_axial.shape[0], ymax_ind=im_axial.shape[1],
                                                       zmax_ind=im_axial.shape[2],
                                                       x_res=res_through_plane, y_res=res_in_plane, z_res=res_in_plane,
                                                       target_res=2)

    coords_cor, intensities_cor = get_ortho_kernel(im_cor, xmin_ind=0, ymin_ind=0, zmin_ind=0,
                                                   xmax_ind=im_cor.shape[0], ymax_ind=im_cor.shape[1],
                                                   zmax_ind=im_cor.shape[2],
                                                   x_res=res_in_plane, y_res=res_in_plane, z_res=res_through_plane,
                                                   target_res=2)

    coords_sag, intensities_sag = get_ortho_kernel(im_sag, xmin_ind=0, ymin_ind=0, zmin_ind=0,
                                                   xmax_ind=im_sag.shape[0], ymax_ind=im_sag.shape[1],
                                                   zmax_ind=im_sag.shape[2],
                                                   x_res=res_in_plane, y_res=res_through_plane, z_res=res_in_plane,
                                                   target_res=2)

    # coords = np.array(coords_axial + coords_cor + coords_sag)
    coords = np.array(coords_axial)

    # intensities = np.array(intensities_axial + intensities_cor + intensities_sag)
    intensities = np.array(intensities_axial)

    # Build the interpolator
    f = RegularGridInterpolator(coords, intensities)  # interpolator

    # Create a SR_vol with target resolution

    # Identify co-ords to get interpolated values for
    # x = np.arange(0, FOV_x, target_res)
    # y = np.arange(0, FOV_y, target_res)
    # z = np.arange(0, FOV_z, target_res)
    # [X, Y, Z] = np.meshgrid(x, y, z, indexing='ij')
    dx, dy, dz = target_res, target_res, target_res
    new_grid = np.mgrid[0:FOVx:dx, 0:FOVy:dy, 0:FOVz:dz]  # new grid
    new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))  # reorder axes for evaluation

    print('Starting interpolation')
    t = time.time()
    # interp = NearestNDInterpolator(coords, intensities)
    # SR_vol = interp(X, Y, Z)
    SR_vol = f(new_grid)

    im4 = np.squeeze(SR_vol[37, :, :])
    plt.imshow(np.abs(im4), cmap='gray')
    plt.show()

    t2 = time.time() - t
    print('Time elapsed:' + str(t2) + 'seconds')
    return SR_vol, im_axial, im_cor, im_sag


def get_ortho_kernel(im, xmin_ind, ymin_ind, zmin_ind,
                     xmax_ind, ymax_ind, zmax_ind,
                     x_res, y_res, z_res, target_res=2):
    coords = []
    intensities = []

    for x_ind in range(xmin_ind, xmax_ind):
      for y_ind in range(ymin_ind, ymax_ind):
        for z_ind in range(zmin_ind, zmax_ind):
          if im[x_ind, y_ind, z_ind] != 0:
            coords.append((x_ind * x_res, y_ind * y_res, z_ind * z_res))
            intensities.append((np.squeeze(im[x_ind, y_ind, z_ind])))

    return coords, intensities


if __name__ == '__main__':
    mode = 'scanner_data'
    if mode == 'sim':
        FOVx, FOVy, FOVz = 160.0, 220.0, 220.0
        object_im = create_object(res=2, FOV=[FOVx, FOVy, FOVz], phantom='SL3D')
        # plot_anatomy_raw(object_im)
        im_axial = down_res(object_im, res=2, dres=6, axis_dres=0)  # case axial = X
        im_sag = down_res(object_im, res=2, dres=6, axis_dres=1)  # case sagittal = Y
        im_cor = down_res(object_im, res=2, dres=6, axis_dres=2)  # case coronal = Z

        SR_vol, im_ax, im_cor, im_sag = get_SR_vol(np.abs(im_axial), np.abs(im_cor), np.abs(im_sag), res=[2, 6],
                                                 target_res=2)

    elif mode == 'scanner_data':
      # import distortionCorrection2 as gradUnwarp
      ''' Read three data files in one folder - 0us, 50us, raw'''
      dataFolder = r'./Data'
      im_zy_folder = '1zy'  # axial
      im_yx_folder = '1yx'  # coronal
      im_xz_folder = '1xz'  # sagittal

      ''' Acq params '''
      noiseScan = 0
      freqDrift = -300

      # im_axial, _, _, _ = cPP.prepData(dataFolder, im_zy_folder, noiseScan, freqDrift)
      # im_cor, _, _, _ = cPP.prepData(dataFolder, im_yx_folder, noiseScan, freqDrift)
      # im_sag, _, _, _ = cPP.prepData(dataFolder, im_xz_folder, noiseScan, freqDrift)

      # np.save('axial.npy', im_axial, allow_pickle=True, fix_imports=True)
      # np.save('cor.npy', im_cor, allow_pickle=True, fix_imports=True)
      # np.save('sag.npy', im_sag, allow_pickle=True, fix_imports=True)

      im_axial = np.load('axial.npy')
      im_cor = np.load('cor.npy')
      im_sag = np.load('sag.npy')

      # align all to x, y and z
      im_axial = np.moveaxis(im_axial, [0, 2], [2, 0])  # yzx --> xyz
      im_cor = np.moveaxis(im_cor, [0, 1], [1, 0])  # yxz --> xyz
      im_sag = np.moveaxis(im_sag, [1, 2], [2, 1])  # xzy --> xyz



      SR_vol, im_ax, im_cor, im_sag = get_SR_vol(np.abs(im_axial), np.abs(im_cor), np.abs(im_sag), res=[2, 6],
                                                 target_res=2)

      plot_anatomy_raw(im_ax)
      plot_anatomy_raw(im_cor)
      plot_anatomy_raw(im_sag)
      plot_anatomy_raw(SR_vol)
