import nibabel as nib
import numpy as np
from src.utils import preprocess_img_nhp, mosaic_all_slices, mosaic_to_3D, get_num_cols_rows
from nifti_write import make_nifti
from ZSSR_master import configs, configs_2, ZSSR
from ZSSR_2D_ms_nhp_im_process_collage import do_collage_ZSSR_nhp
from colorama import Fore, Style
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D
from roipoly import RoiPoly
import cv2
from skimage.transform import resize
import pywt

visible = False

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
                    fname_zssr:str=None, fspec:str='', scale_fact:int=2, dims:int=1, ground_truth:np.ndarray=None, kernel=None):
    
    fext='.nii.gz'
    # Convert to mosaic
    input2zssr = mosaic_all_slices(img, debug=False,
                                    filename=fname_zssr +'_input.png', savefig=True,
                                    num_cols=num_cols, num_rows=num_rows) # adds a 90 degree rotation

    if ground_truth is not None:
        print('Ground truth provided ...')
        input2zssr_gt = mosaic_all_slices(ground_truth, debug=False,
                                    filename=fname_zssr +'_gt.png', savefig=True,
                                    num_cols=num_cols, num_rows=num_rows) # adds a 90 degree rotation

    print(input2zssr.shape)

    # check getting the volume back  - debug code
    # vol_check = mosaic_to_3D(input2zssr, orig_dim1=img.shape[0], 
    #                                     orig_dim2=img.shape[1], orig_dim3=img.shape[2], num_cols=num_cols, num_rows = num_rows)

    # print('Checking volume ...')
    # OrthoSlicer3D(vol_check).show()
    # plt.show()
    # # Perform collage ZSSR - in 1 dimension, say x direction
    
    recon_conf.scale_factors = [[1, np.sqrt(scale_fact)]]
    im_collage_x = do_collage_ZSSR_nhp(low_res_im=input2zssr, recon_conf=recon_conf, 
                            debug_mode=False, fname_file_save=fname_zssr + fspec + fext, 
                            contrast_enhance=False, write_nifti=False, kernel=kernel,
                            num_iters_srr=1, ground_truth=input2zssr_gt if ground_truth is not None else None)
    
    im_collage_x = do_collage_ZSSR_nhp(low_res_im=im_collage_x, recon_conf=recon_conf, 
                            debug_mode=False, fname_file_save=fname_zssr + fspec + fext, 
                            contrast_enhance=False, write_nifti=False, kernel=kernel,
                            num_iters_srr=1, ground_truth=input2zssr_gt if ground_truth is not None else None)
    # plt.imshow(im_collage,cmap='gray')
    # plt.show()
    print(im_collage_x.shape)

    
    if dims == 2:
        # Perform collage ZSSR - in the other dimension, say y direction
        recon_conf.scale_factors = [[scale_fact, 1]]
        im_collage = do_collage_ZSSR_nhp(low_res_im=im_collage_x, recon_conf=recon_conf, 
                                debug_mode=False, fname_file_save=fname_zssr + fspec + fext, 
                                contrast_enhance=False, write_nifti=False,
                                num_iters_srr=1)
        
        new_dims = [int(img.shape[0]* scale_fact), int(img.shape[1]* scale_fact),
                    img.shape[2]]
        srr_volume = mosaic_to_3D(im_collage, orig_dim1=new_dims[0], 
                                            orig_dim2=new_dims[1], orig_dim3=new_dims[2], num_cols=num_cols, num_rows = num_rows)
        
        # dislpay the srr_volume
        print(Fore.GREEN + 'New shape of volume after collage to 3D is: '+ str(srr_volume.shape))
        # OrthoSlicer3D(srr_volume).show()
        # plt.show()

        new_dims = [int(img.shape[0]), int(img.shape[1]* scale_fact),
                img.shape[2]* scale_fact]

    else:
        im_collage = im_collage_x
        new_dims = [int(img.shape[0]), int(img.shape[1]),
                img.shape[2]* scale_fact]
    
    srr_volume = mosaic_to_3D(im_collage, orig_dim1=new_dims[0], 
                                        orig_dim2=new_dims[1], orig_dim3=new_dims[2], num_cols=num_cols, num_rows = num_rows)
    
    # dislpay the srr_volume
    print(Fore.GREEN + 'New shape of volume after collage to 3D is: '+ str(srr_volume.shape))
    # OrthoSlicer3D(srr_volume).show()
    # plt.show()

    # # We will make this conditional on debug but using it for now
    # filename = fname_zssr + fspec + 'op_collage.png'
    # plt.imshow(im_collage, cmap='gray')
    # plt.axis('off')
    # plt.savefig(filename, bbox_inches="tight")
    # plt.show()
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

# Need to work on this function and ZSSR model new

def do_zssr_recon_slices(img:np.ndarray = 0, fname:str='', do_preprocess:bool= False,
                   do_postprocess:bool=False, viewing = False,
              padding:bool = False, mask_image:bool = False, reuse_mask:bool = False,
              target_resolution_fact=[1, 1, 2], recon_conf:configs.Config=None):
    
    # print all config values
    print(vars(recon_conf))
    
    print(Fore.GREEN + "..................Running ZSSR on slices................................." + Style.RESET_ALL)
    
    print(Fore.YELLOW + f"Input image shape: {img.shape}" + Style.RESET_ALL)
    T = 0.01 *  (np.max(img))
    img[img < T] = 0
    img = img / np.max(img) # Normalize

    img_zssr = np.zeros([int(img.shape[0] *target_resolution_fact[0]), 
                        int(img.shape[1] *target_resolution_fact[1]), 
                        int(img.shape[2]*target_resolution_fact[2])])

    # OrthoSlicer3D(img_zssr).show()

    # Perform upscaling along x axis - assuming that the data has dimensions [x, y, z]
    img_zssr_x = np.zeros([int(img.shape[0] *target_resolution_fact[0]), 
                        int(img.shape[1]), 
                        int(img.shape[2])])
    
    for slice in range(img.shape[1]): # Loop over the slice dimension which is the 3rd dimension
        
        print(Fore.GREEN + f"Running with x axis, processing slice {slice}" + Style.RESET_ALL)
        img_slice = np.squeeze(img[:, slice, :])  # Get the slice along x axis
        print(Fore.YELLOW + f"img_slice shape: {img_slice.shape}" + Style.RESET_ALL)
        input_img = np.zeros((img_slice.shape[0], img_slice.shape[1], 3))
        
        input_img[:, :, 0] = img_slice
        input_img[:, :, 1] = img_slice
        input_img[:, :, 2] = img_slice
        # recon_conf = configs.Config()
        # recon_conf.scale_factors = [[np.sqrt(target_resolution_fact[0]), 1]]

        if np.sum(input_img) > 0:
            # do first pass - x, y all slices
            print(Fore.YELLOW + f"input_img shape: {input_img.shape}" + Style.RESET_ALL)
            
            net = ZSSR.ZSSR(input_img = input_img, conf=recon_conf, ground_truth=None, kernels=None)
            int_img = net.run()

            recon_conf.scale_factors[0][0] = img_zssr_x.shape[0] / int_img.shape[0]
            net = ZSSR.ZSSR(input_img = int_img, conf=recon_conf, ground_truth=None, kernels=None)
            output_img = net.run()

            high_res_im = np.squeeze(output_img[:, :, 0])
            
            # Repeat the zssr again so it is gradual upscaling
            target_shape = img_zssr_x[:, slice, :].shape
            high_res_im = resize(high_res_im, target_shape, order=3, mode="reflect", anti_aliasing=True)

            img_zssr_x[:, slice, :] = high_res_im
            
            print(Fore.YELLOW + "High_res_im.shape:", high_res_im.shape)
            print(Fore.YELLOW + "Image after x axis ZSSR shape:", img_zssr_x.shape)

            if visible == True:
                # Display input slice, intermediate image, and output high_res_im with shape and axis info, referencing x slice
                plt.figure(figsize=(15, 4))
                plt.suptitle(f"Processing x-slice {slice}", fontsize=14)
                plt.subplot(1, 3, 1)
                plt.imshow(img_slice, cmap='gray')
                plt.title(f"Input (x-slice {slice})\nshape: {img_slice.shape}\n(axis: x-z)")
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.imshow(np.squeeze(int_img[:, :, 0]), cmap='gray')
                plt.title(f"Intermediate (x-slice {slice})\nshape: {np.squeeze(int_img[:, :, 0]).shape}\n(axis: x-z)")
                plt.axis('off')
                plt.subplot(1, 3, 3)
                plt.imshow(high_res_im, cmap='gray')
                plt.title(f"Output (x-slice {slice})\nshape: {high_res_im.shape}\n(axis: x-z)")
                plt.axis('off')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            print(Fore.YELLOW + str(slice))
        else:
            print(Fore.RED + 'No data found!')
            output_img = input_img
            print(Style.RESET_ALL)

    print("img_zssr_x[:, slice, :].shape:", img_zssr_x.shape)
    
    # Repeat for y axis
    img_zssr_y = np.zeros([int(img.shape[0] *target_resolution_fact[0]), 
                        int(img.shape[1] *target_resolution_fact[1]), 
                        int(img.shape[2])])
    
    for slice in range(img_zssr_x.shape[0]): # Loop over the slice dimension which is the  
        print(Fore.GREEN + f"Running with y axis, processing slice {slice}" + Style.RESET_ALL)
        img_slice = np.squeeze(img_zssr_x[slice, :, :])  # Get the slice along y axis
        print(Fore.YELLOW + f"img_slice shape: {img_slice.shape}" + Style.RESET_ALL)
        input_img = np.zeros((img_slice.shape[0], img_slice.shape[1], 3))
        
        input_img[:, :, 0] = img_slice
        input_img[:, :, 1] = img_slice
        input_img[:, :, 2] = img_slice
        
        # recon_conf = configs.Config()
        # recon_conf.scale_factors = [[np.sqrt(target_resolution_fact[1]), 1]]

        if np.sum(input_img) > 0:
            print(Fore.YELLOW + f"input_img shape: {input_img.shape}" + Style.RESET_ALL)
            
            net = ZSSR.ZSSR(input_img = input_img, conf=recon_conf, ground_truth=None, kernels=None)
            int_img = net.run()
           
            recon_conf.scale_factors[0][0] = img_zssr_y.shape[1] / int_img.shape[0]
            net = ZSSR.ZSSR(input_img = int_img, conf=recon_conf, ground_truth=None, kernels=None)
            output_img = net.run()
            
            high_res_im = np.squeeze(output_img[:, :, 0])
            
            target_shape = img_zssr_y[slice, :, :].shape
            high_res_im = resize(high_res_im, target_shape, order=3, mode="reflect", anti_aliasing=True)
            img_zssr_y[slice,: ,:] = high_res_im

            print("high_res_im.shape:", high_res_im.shape)

            if visible == True:
                # Display input slice, intermediate image, and output high_res_im for y axis
                plt.figure(figsize=(15, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(img_slice, cmap='gray')
                plt.title(f"Y-axis Input Slice {slice}\nshape: {img_slice.shape}")
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.imshow(np.squeeze(int_img[:, :, 0]), cmap='gray')
                plt.title(f"Y-axis Intermediate int_img {slice}\nshape: {np.squeeze(int_img[:, :, 0]).shape}")
                plt.axis('off')
                plt.subplot(1, 3, 3)
                plt.imshow(high_res_im, cmap='gray')
                plt.title(f"Y-axis Output high_res_im {slice}\nshape: {high_res_im.shape}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            print(Fore.YELLOW + str(slice))
        else:
            print(Fore.RED + 'No data found!')
            output_img = input_img
            print(Style.RESET_ALL)

    return img_zssr_y
