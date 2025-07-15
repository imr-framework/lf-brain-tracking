# This file performs ZSSR 2D SRR slice by slice for low field MRI

from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt
from colorama import Fore, Style
import sys
import numpy as np
# sys.path.append('./Code/ZSSR_master')
# import Code.ZSSR_master.configs
from ZSSR_master import configs, configs_2, ZSSR
from roipoly import RoiPoly
import matplotlib.image as img
import nibabel as nib
import cv2

def do_contrast_enhance(im:np.ndarray=None, method:str='custom', fact:float=1.10):
    im_ce = np.zeros_like(im)
    for slice in range(im.shape[2]):
        im_slice = np.squeeze(im[:, :, slice])
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
        im_ce[:, :, slice] = im_cl


    return im_ce

def do_norm_im(im_slice:np.ndarray=0):
    max_slice = np.max(im_slice)
    min_slice = np.min(im_slice)
    im_slice_norm = (im_slice - min_slice) / (max_slice - min_slice)
    return im_slice_norm

if __name__=='__main__':
    debug_mode = True
    recon_conf = configs.Config()
    nii_name = './Data/MW/IRF_NHP/35528/nhp_35528_D_56.nii.gz'
    nhp_num = '35528_D_56'
    low_res_im = nib.load(nii_name).get_fdata()
    recon_conf2 = configs_2.Config()
    contrast_enhance = False

    slices = low_res_im.shape[2]
    new_dimx = int(low_res_im.shape[0] * recon_conf.scale_factors[0][0])
    new_dimy = int(low_res_im.shape[1] * recon_conf.scale_factors[0][1])
    new_dimz = int(low_res_im.shape[2] * recon_conf2.scale_factors[0][1])
    high_res_im = np.zeros((new_dimx, new_dimy, slices), dtype=float)
    high_res_volume = np.zeros((new_dimx, new_dimy, new_dimz), dtype=float)
    
    if debug_mode is True:
            s = OrthoSlicer3D(low_res_im)
            s.clim = (0, 2048)
            s.show()
    
    # if contrast_enhance is True:
    low_res_im_ce = do_contrast_enhance(low_res_im)
    low_res_im = low_res_im_ce


    for slice in range(0, slices):
        input_img = np.zeros((low_res_im.shape[0], low_res_im.shape[1], 3))
        
        im_slice = np.squeeze(low_res_im[:, :, slice])
        im_slice_norm = do_norm_im(im_slice)
        seg = im_slice_norm 

        input_img[:, :, 0] = seg
        input_img[:, :, 1] = seg
        input_img[:, :, 2] = seg

        np.nan_to_num(input_img, copy=False,nan=0.0)
        
        if np.sum(input_img) > 0:
            # do first pass - x, y all slices
            net = ZSSR.ZSSR(input_img = input_img, conf=recon_conf, ground_truth=None, kernels=None)
            output_img = net.run()
            high_res_im[ :, :, slice] = np.squeeze(output_img[:, :, 0])
            print(Fore.YELLOW + str(slice))
        else:
            print(Fore.RED + 'No data found!')
            output_img = input_img
            print(Style.RESET_ALL)

        if debug_mode is True:
            fig = plt.figure(1)
            print(Fore.GREEN + "Debug mode: display now")
            plt.imshow(im_slice_norm, cmap='gray')
     
            print(Fore.GREEN + "Debug mode: display now")
            plt.imshow(np.squeeze(input_img), cmap='gray')
            # plt.clim(0.2, 0.8)
            # plt.show()
            plt.savefig('low_res_slice_'+str(slice) + '.png')



            fig = plt.figure(2)
            print(Fore.GREEN + "Debug mode: display now")
            plt.imshow(im_slice_norm, cmap='gray')
     
            print(Fore.GREEN + "Debug mode: display now")
            plt.imshow(np.squeeze(output_img[:, :, 0]), cmap='gray')
            # plt.clim(0.2, 0.8)
            # plt.show()
            plt.savefig('high_res_slice_'+str(slice) + '.png')

    np.save('output_im' + nhp_num, high_res_im)

    for x in range(0, high_res_im.shape[0]):
        input_img = np.zeros((high_res_im.shape[1], low_res_im.shape[2], 3))
        
        im_slice = np.squeeze(high_res_im[x, :, :])

        im_slice_norm = do_norm_im(im_slice)
        seg = im_slice_norm 

        input_img[:, :, 0] = seg
        input_img[:, :, 1] = seg
        input_img[:, :, 2] = seg

        np.nan_to_num(input_img, copy=False,nan=0.0)
        
        if np.sum(input_img) > 0:
            # do first pass - x, y all slices
            net = ZSSR.ZSSR(input_img = input_img, conf=recon_conf2, ground_truth=None, kernels=None)
            output_img = net.run()
            high_res_volume[ x, :, :] = np.squeeze(output_img[:, :, 0])
            print(Fore.YELLOW + str(slice))
            print(Style.RESET_ALL)
        else:
            print(Fore.RED + 'No data found!')
            output_img = input_img
            print(Style.RESET_ALL)
         
        print(Fore.GREEN + str(x))

        
    np.save('output_volume' + nhp_num, high_res_volume)
    if debug_mode is True:
        s = OrthoSlicer3D(high_res_volume)
        s.clim = (0, 2048)
        s.show()
    print(Style.RESET_ALL)
 