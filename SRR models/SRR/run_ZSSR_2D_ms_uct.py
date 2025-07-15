# This file performs ZSSR 2D SRR slice by slice for low field MRI
import pydicom as dicom
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


def run_zssr_workflow(im_low_res_data:np.ndarray=None, recon_config=None):
    input_img = np.zeros((im_low_res_data.shape[0], im_low_res_data.shape[2], 3))
    srr_img = np.zeros((int(im_low_res_data.shape[0] * recon_config.scale_factors[0][0] * 2) + 1,
                         im_low_res_data.shape[1], 
                         int(im_low_res_data.shape[2] * recon_config.scale_factors[0][1] + 1)))            
    
    recon_config.scale_factors_first = recon_config.scale_factors
    for slice in range(im_low_res_data.shape[1]):
        im = np.squeeze(im_low_res_data[:, slice, :])
        input_img[:, :, 0] = im
        input_img[:, :, 1] = im
        input_img[:, :, 2] = im
        recon_config.scale_factors = recon_config.scale_factors_first
        net = ZSSR.ZSSR(input_img = input_img, conf=recon_config, ground_truth=None, kernels=None)
        stage1_img = net.run()
        # Second pass
        recon_config.scale_factors = [[2, 1]] # second step is always by 2X
        net = ZSSR.ZSSR(input_img = stage1_img, conf=recon_config, ground_truth=None, kernels=None)
        output_img = net.run()

        srr_img[:, slice, :] = output_img[:, :, 0]
    return srr_img



def do_norm_im(im_slice:np.ndarray=0):
    max_slice = np.max(im_slice)
    min_slice = np.min(im_slice)
    im_slice_norm = (im_slice - min_slice) / (max_slice - min_slice)
    return im_slice_norm

if __name__=='__main__':
    debug_mode = True
    recon_conf = configs.Config()
    dcm_name = './Data/UCT_Hyperfine/T5/T5/191_11561329_24M/2.25.325273824029857559961672379501533755282.dcm'
    subject_id = 's191_11173897_3M'
    ds = dicom.dcmread(dcm_name)  #Check for pixel data - "PixelData" in ds
    print(ds.PulseSequenceName)
    low_res_im = ds.pixel_array

    # Based on data axis - restructuring for uniformity
    # low_res_im = np.moveaxis(low_res_im, [0, 1, 2], [2, 0, 1])
    # print(low_res_im.shape)

    recon_conf2 = configs_2.Config()
    contrast_enhance = True
    slices = low_res_im.shape[2]
    new_dimx = int(low_res_im.shape[0] * recon_conf2.scale_factors[0][0])
    new_dimy = int(low_res_im.shape[1] * recon_conf2.scale_factors[0][1])
    new_dimz = int(low_res_im.shape[2] * recon_conf2.scale_factors[0][1])
    high_res_im = np.zeros((new_dimx, new_dimy, slices), dtype=float)
    high_res_volume = np.zeros((new_dimx, new_dimy, new_dimz), dtype=float)
    
    if debug_mode is True:
            s = OrthoSlicer3D(low_res_im)
            # s.clim = (1000, 1800)
            s.show()


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
            net = ZSSR.ZSSR(input_img = input_img, conf=recon_conf2, ground_truth=None, kernels=None)
            output_img = net.run()
            high_res_im[ :, :, slice] = np.squeeze(output_img[:, :, 0])
            print(Fore.YELLOW + str(slice))
        else:
            print(Fore.RED + 'No data found!')
            output_img = input_img
            print(Style.RESET_ALL)

        if debug_mode is True:
            fig = plt.figure(2)
            # print(Fore.GREEN + "Debug mode: display now")
            plt.imshow(np.squeeze(input_img), cmap='gray')
            plt.savefig('low_res_slice_'+str(slice) + '.png')

            fig = plt.figure(3)
            # print(Fore.GREEN + "Debug mode: display now")
            plt.imshow(np.squeeze(output_img[:, :, 0]), cmap='gray')
            plt.savefig('high_res_slice_'+str(slice) + '.png')

    np.save('output_im' + subject_id, high_res_im)

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
 