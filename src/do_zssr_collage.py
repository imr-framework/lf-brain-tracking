import nibabel as nib
import numpy as np
from utils import preprocess_img_nhp, mosaic_all_slices, mosaic_to_3D, get_num_cols_rows
from nifti_write import make_nifti
from ZSSR_master import configs, configs_2, ZSSR
from ZSSR_2D_ms_nhp_im_process_collage import do_collage_ZSSR_nhp
from colorama import Fore, Style
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D
from roipoly import RoiPoly
import cv2

def do_mask_image(img:np.ndarray=None):
    masked_image = np.copy(img)
    mask_store = np.zeros_like(img, dtype=bool)
    for slice in range(masked_image.shape[2]):
        im_slice = np.squeeze(masked_image[:, :, slice])
        plt.imshow(im_slice, cmap='gray')
        r = RoiPoly(color='r')
        mask = r.get_mask(im_slice)
        mask_store[:, :, slice] = mask
        masked_image[:, :, slice] = np.multiply(im_slice, mask)
        
    return masked_image, mask_store

def do_ZSSR_steps(img:np.ndarray=None, recon_conf:configs.Config=None,
                    num_cols:int=5, num_rows:int=5,
                    fname_zssr:str=None, fspec:str='', scale_fact:int=2):
    fext='.nii.gz'
    # Convert to mosaic
    input2zssr = mosaic_all_slices(img, debug=False,
                                    filename=fname_zssr +'_input.png', savefig=True,
                                    num_cols=num_cols, num_rows=num_rows) # adds a 90 degree rotation

    print(input2zssr.shape)

    # Perform collage ZSSR
    im_collage = do_collage_ZSSR_nhp(low_res_im=input2zssr, recon_conf=recon_conf, 
                            debug_mode=False, fname_file_save=fname_zssr + fspec + fext, 
                            contrast_enhance=False, write_nifti=False,
                            num_iters_srr=1)
    # plt.imshow(im_collage,cmap='gray')
    # plt.show()
    print(im_collage.shape)
    new_dims = [img.shape[0], int(img.shape[1] * scale_fact),
                img.shape[2]]
    srr_volume = mosaic_to_3D(im_collage, orig_dim1=new_dims[0], 
                                        orig_dim2=new_dims[1], orig_dim3=new_dims[2])
    return srr_volume

# ----------------------------------------------
# target res = 1, 1, 2 - 2x, 2x, 2.665x

def do_zssr_recon(img:np.ndarray = 0, fname:str='', do_preprocess:bool= False,
                   do_postprocess:bool=False, viewing = False,
              padding:bool = False, mask_image:bool = False, reuse_mask:bool = False):

    # Read input nifti file in slices; initialize paths to save files
    fname_processed = fname[:-7] + '_preprocessed.nii.gz'
    fname_zssr = fname[:-7] + '_zssr'
    fext = '.nii.gz'
    
    recon_conf = configs.Config()
    recon_conf.scale_factors = [[2, 1]]

    # Step 0 - Threshold 
    T = 0.01 *  (np.max(img))
    img[img < T] = 0
    img = img / np.max(img) # Normalize
   

    # ----------------------------------------------
    # offset the slice wrap
    img = np.roll(img, -3, axis=2)
    if mask_image is True:
        if reuse_mask is True:
            mask_store = np.load('mask_nhp_invivo.npy')
            img = np.multiply(img, mask_store)
        else:
            img, mask_store = do_mask_image(img)
            np.save('mask_nhp_invivo.npy', mask_store)


    # check for circshift in image
    # mosaic_all_slices(np.rot90(img, k=0), debug=True,
    #                                     filename=fname_zssr +'_input.png', savefig=True,
    #                                     num_cols=3, num_rows=5)

    # ----------------------------------------------
    # Step 1 -  Choose slice axis (coronal), preprocess and perform ZSSR (2x) 
    # in unrolled mode and return the processed 3D volume
    #-----------------------------------------------
    # Get coronal view
    
    coronal_img = np.swapaxes(img, 1, 2)      # 77 X 128 X 120
    coronal_img_shape = coronal_img.shape
    print(Fore.YELLOW + 'Input image dims after axis swap: ' +str(coronal_img_shape))
    # Step 1A - Preprocess
    if padding is True:
        # Pad the image
        n = 2
        coronal_img = np.pad(coronal_img, ((0, 0), (n, n-1), (0, 0)), mode='constant', constant_values=0.01)
        print(Fore.YELLOW + 'Input image dims after padding: ' + str(coronal_img.shape))

    if do_preprocess:
        processed_coronal_img = preprocess_img_nhp(coronal_img, debug=False)
        make_nifti(processed_coronal_img, fname = fname_processed, mask=False, 
            res=[2, 5, 2], dim_info=[0, 1, 2])
    else:
        processed_coronal_img = coronal_img

    # Perform ZSSR
    recon_conf.scale_factors = [[2, 1]]
    num_cols, num_rows = get_num_cols_rows(processed_coronal_img)
    coronal_2x_volume = do_ZSSR_steps(img=processed_coronal_img, recon_conf=recon_conf,
                                        num_cols=num_cols, num_rows=num_rows,
                                        fname_zssr=fname_zssr, fspec='_output_coronal', scale_fact=2)
    print(Fore.GREEN + 'New shape of volume is: '+ str(coronal_2x_volume.shape))
    if viewing == True:
        OrthoSlicer3D(coronal_2x_volume).show()

    # ----------------------------------------------
    # Step 2 - Axial ZSSR (2x) in unrolled mode
    #-----------------------------------------------
    print(Fore.RED + 'Input to axial SRR dim: '+ str(coronal_2x_volume.shape))
    recon_conf.scale_factors = [[2, 1]]
    num_cols, num_rows = get_num_cols_rows(coronal_2x_volume)
    axial_2x_volume = do_ZSSR_steps(img=coronal_2x_volume, recon_conf=recon_conf,
                                        num_cols=num_cols, num_rows=num_rows,
                                        fname_zssr=fname_zssr, fspec='_output_coronal', scale_fact=2)
    axial_2x_volume = np.swapaxes(axial_2x_volume, 0, 2)   # 128 X 64 X 32
    print(Fore.GREEN + 'New shape of volume is: '+ str(axial_2x_volume.shape)) 
    if viewing == True:
        OrthoSlicer3D(axial_2x_volume).show()
    
    # ----------------------------------------------
    # Step 3 - Coronal ZSSR (2x) in unrolled mode
    #-----------------------------------------------
    # coronal_img = np.swapaxes(axial_2x_volume, 1, 2)      # 128 X 32 X 64
    # coronal_img = np.swapaxes(coronal_img, 0, 2)          # 64 x 32 x 128
    # print(Fore.BLUE + 'Input to 2nd coronal SRR dim: '+ str(coronal_img.shape))
    # recon_conf.scale_factors = [[2, 1]]
    # num_cols, num_rows = get_num_cols_rows(coronal_img)
    # coronal_2p5x_volume = do_ZSSR_steps(img=coronal_img, recon_conf=recon_conf,
    #                                     num_cols=num_cols, num_rows=num_rows,
    #                                     fname_zssr=fname_zssr, fspec='_output_coronal', scale_fact=2)

    # print(Fore.GREEN + 'New shape of volume is: '+ str(coronal_2p5x_volume.shape))
   
    # ----------------------------------------------
    # Step 3 - Sagittal ZSSR (2x) in unrolled mode
    #-----------------------------------------------
    recon_conf.scale_factors = [[2, 1]]
    sagittal_2x_volume = np.zeros((axial_2x_volume.shape[0], axial_2x_volume.shape[1] * 2, axial_2x_volume.shape[2]))
    for slice in range(axial_2x_volume.shape[2]):
        input2zssr = np.rot90(axial_2x_volume[:, :, slice], k=1)

        im_collage = do_collage_ZSSR_nhp(low_res_im=input2zssr, recon_conf=recon_conf, 
                        debug_mode=False, fname_file_save=fname_zssr + 'final' + fext, 
                        contrast_enhance=False, write_nifti=False,
                        num_iters_srr=1)

        sagittal_2x_volume[:, :, slice] = im_collage.T
        if viewing == True:
            OrthoSlicer3D(sagittal_2x_volume).show()
    print(Fore.GREEN + 'New shape of volume is: '+ str(sagittal_2x_volume.shape))
    

    if do_postprocess is True:
        processed_volume = preprocess_img_nhp(sagittal_2x_volume, debug=False)
    else:
        processed_volume = sagittal_2x_volume

    # ----------------------------------------------   
    return processed_volume
    
def do_zssr_recon_slices(img:np.ndarray = 0, fname:str='', do_preprocess:bool= False,
                   do_postprocess:bool=False, viewing = False,
              padding:bool = False, mask_image:bool = False, reuse_mask:bool = False,
              target_resolution_fact=[1, 1, 2]):
    
    T = 0.01 *  (np.max(img))
    img[img < T] = 0
    img = img / np.max(img) # Normalize

    img_zssr = np.zeros((img.shape[0] *target_resolution_fact[0], 
                        img.shape[1] *target_resolution_fact[1], 
                        img.shape[2]*target_resolution_fact[2]))

    for slice in range(img.shape[0]):
        img_slice = np.squeeze(img[slice, :, :])
        input_img = np.zeros((img_slice.shape[0], img_slice.shape[1], 3))
        
        input_img[:, :, 0] = img_slice
        input_img[:, :, 1] = img_slice
        input_img[:, :, 2] = img_slice
        recon_conf = configs.Config()
        recon_conf.scale_factors = [[1, target_resolution_fact[2]]]

        if np.sum(input_img) > 0:
            # do first pass - x, y all slices
            net = ZSSR.ZSSR(input_img = input_img, conf=recon_conf, ground_truth=None, kernels=None)
            output_img = net.run()
            high_res_im = np.squeeze(output_img[:, :, 0])
            img_zssr[slice, :, :] = high_res_im
            print(Fore.YELLOW + str(slice))
        else:
            print(Fore.RED + 'No data found!')
            output_img = input_img
            print(Style.RESET_ALL)

    return img_zssr
