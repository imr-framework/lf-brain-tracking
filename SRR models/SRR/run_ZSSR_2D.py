# This file performs ZSSR 2D SRR slice by slice for low field MRI

from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt
from colorama import Fore, Style
import sys
import numpy as np
# sys.path.append('./Code/ZSSR_master')
# import Code.ZSSR_master.configs
from ZSSR_master import configs, ZSSR
from roipoly import RoiPoly
import matplotlib.image as img
import nibabel as nib

if __name__=='__main__':
    debug_mode = True
    recon_conf = configs.Config()
    low_res_im = np.abs(np.load('./Data/MW/axial_circshift_1yz.npy'))
    # low_res_im = np.abs(np.load('./Data/MW/coronal_circshift_1zx.npy'))
    # low_res_im = np.abs(np.load('./Data/MW/sagittal_circshift_1yx.npy'))
    
    nii_name = './Data/MW/reoriented_skull_stripped_raw_srr7_reoriented.nii.gz'
    low_res_im = nib.load(nii_name).get_fdata()

    input_img_fname = './Code/ZSSR_master/test_data/brain_T2w.png'
    slices = low_res_im.shape[2]
    new_dimx = int(low_res_im.shape[0] * recon_conf.scale_factors[0][0])
    new_dimy = int(low_res_im.shape[1] * recon_conf.scale_factors[0][1])
    high_res_im = np.zeros((new_dimx, new_dimy, slices), dtype=float)
    
    if debug_mode is True:
            s = OrthoSlicer3D(low_res_im)
            s.show()

    
    for slice in range(0, slices):
        input_img = np.zeros((low_res_im.shape[0], low_res_im.shape[1], 3))
        
        im_slice = np.squeeze(low_res_im[:, :, slice])
        max_slice = np.max(im_slice)
        min_slice = np.min(im_slice)

        im_slice_norm = (im_slice - min_slice) / (max_slice - min_slice)
        


        seg = im_slice_norm 

        input_img[:, :, 0] = seg
        input_img[:, :, 1] = seg
        input_img[:, :, 2] = seg

        np.nan_to_num(input_img, copy=False,nan=0.0)
        
           
        if np.sum(input_img) > 0:
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


            # r = RoiPoly(color='r', fig = fig)
            # mask = r.get_mask(im_slice_norm)
            plt.imshow(seg,interpolation='nearest', cmap='gray')
            plt.imsave('%s/input_img1.png' %
                           (recon_conf.result_path),
                           input_img, vmin=0, vmax=1)
            plt.title('Segmented input - please save this file')
            plt.show()
    
            print(Fore.GREEN + "Debug mode: display now")
            plt.imshow(np.squeeze(output_img[:, :, 0]), cmap='gray')
            plt.show()

    print(Style.RESET_ALL)
    np.save('output.npy', high_res_im)