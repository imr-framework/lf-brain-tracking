# This file performs ZSSR 2D SRR slice by slice for low field MRI

from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt
from colorama import Fore, Style
import sys
import numpy as np
from nifti_write import make_nifti
from ZSSR_master import configs, configs_2, ZSSR
from roipoly import RoiPoly
import matplotlib.image as img
import nibabel as nib
import cv2

def do_contrast_enhance(im_slice:np.ndarray=None, method:str='custom', fact:float=1.10):
    im_ce = np.zeros_like(im_slice)
    if method is 'nc':
        im_slice = 255 * do_norm_im(im_slice)
        im_slice = im_slice.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
        im_cl = clahe.apply(im_slice)
            
        # morphological
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        # Top Hat Transform
        topHat = cv2.morphologyEx(im_slice, cv2.MORPH_TOPHAT, kernel)
        # Black Hat Transform
        blackHat = cv2.morphologyEx(im_slice, cv2.MORPH_BLACKHAT, kernel)
        im_cl = im_slice + topHat - blackHat
    else:
        # im_slice[np.where(im_slice <1350)] = im_slice[np.where(im_slice <1350)] / fact
        # im_slice[np.where(im_slice >1550)]  *= fact
        im_slice[im_slice < 900] = im_slice[im_slice < 900] / fact
        im_cl = im_slice
        
    im_ce = im_cl
    return im_ce

def do_norm_im(im_slice:np.ndarray=0):
    max_slice = np.max(im_slice)
    min_slice = np.min(im_slice)
    im_slice_norm = (im_slice - min_slice) / (max_slice - min_slice)
    return im_slice_norm

def do_collage_ZSSR_nhp(low_res_im:np.ndarray=None, recon_conf:configs_2.Config=None, 
                        debug_mode:bool=False, fname_file_save:str=None, 
                        contrast_enhance:bool=False, write_nifti:bool=True,
                        num_iters_srr:int=1):  
    ''' Input: low-resolution collage
        Output: high resolution collage in direction and factor specified by config
    '''
    # debug_mode = False
    # recon_conf = configs.Config()
    # low_res_im = nib.load(nii_name).get_fdata()
    # fname_file_save = nii_name[:-7]
    # contrast_enhance = False
    # write_nifti = True
    # num_iters_srr = 2

    # ----------------------------------------------
    new_dimx = int(np.ceil(low_res_im.shape[0] * recon_conf.scale_factors[0][0]))
    new_dimy = int(np.ceil(low_res_im.shape[1] * recon_conf.scale_factors[0][1]))
    high_res_im = np.zeros((new_dimx, new_dimy), dtype=float)
    
    if debug_mode is True:
            plt.imshow(low_res_im)
            plt.show()
    
    if contrast_enhance is True:
        low_res_im_ce = do_contrast_enhance(low_res_im)
        low_res_im = low_res_im_ce  

    
    for iter_srr in range(num_iters_srr):
        input_img = np.zeros((low_res_im.shape[0], low_res_im.shape[1], 3))
        seg = do_norm_im(low_res_im)

        input_img[:, :, 0] = seg
        input_img[:, :, 1] = seg
        input_img[:, :, 2] = seg

        np.nan_to_num(input_img, copy=False,nan=0.0)
        
        if np.sum(input_img) > 0:
            # do first pass - x, y all slices
            net = ZSSR.ZSSR(input_img = input_img, conf=recon_conf, ground_truth=None, kernels=None)
            output_img = net.run()
            high_res_im = np.squeeze(output_img[:, :, 0])
            print(Fore.YELLOW + str(slice))
        else:
            print(Fore.RED + 'No data found!')
            output_img = input_img
            print(Style.RESET_ALL)

        if debug_mode is True:
            fig = plt.figure(1)
            print(Fore.GREEN + "Debug mode: display now")
            plt.imshow(seg, cmap='gray')
    
            print(Fore.GREEN + "Debug mode: display now")
            plt.imshow(np.squeeze(input_img), cmap='gray')
            # plt.clim(0.2, 0.8)
            # plt.show()
            if iter_srr == 0:
                plt.savefig('low_res_slice_125'+str(slice) + '.png')

            fig = plt.figure(2)
            print(Fore.GREEN + "Debug mode: display now")
            plt.imshow(seg, cmap='gray')
    
            print(Fore.GREEN + "Debug mode: display now")
            plt.imshow(np.squeeze(output_img[:, :, 0]), cmap='gray')
            # plt.clim(0.2, 0.8)
            # plt.show()
            plt.savefig('high_res_slice_'+str(slice) + '.png')
        low_res_im = high_res_im
        new_dimx = int(np.ceil(low_res_im.shape[0] * recon_conf.scale_factors[0][0]))
        new_dimy = int(np.ceil(low_res_im.shape[1] * recon_conf.scale_factors[0][1]))
        high_res_im = np.zeros((new_dimx, new_dimy))
        
        if contrast_enhance is True:
            high_res_im_ce = do_contrast_enhance(high_res_im)
            high_res_im = high_res_im_ce  
        if write_nifti is True:
            make_nifti(data=low_res_im * 2048, fname= fname_file_save + 'ZSSR_2D_ms_srr_1dot2x_iter'+ str(iter_srr) +'.nii.gz', mask=False,
                res=[1, 1, 5], dim_info=[2, 1, 0]) # phase, freq, slice    

        if debug_mode is True:
            img = nib.load(fname_file_save + 'ZSSR_2D_ms_srr_1dot2x_iter'+ str(iter_srr) +'.nii.gz')
            s = OrthoSlicer3D(img.get_fdata())
            s.clim = (0, 2048)
            s.show()
    print(Style.RESET_ALL)
    high_res_im = low_res_im
    return high_res_im
 