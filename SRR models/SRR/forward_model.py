# -*- coding: utf-8 -*-
"""
Created on Wed March 27, 2024
@author: Sairam Geethanath
Input: 3D files with a isotropic resolutions
Purpose: convert to low resolution volumes for training
Output: Low resolution volumes
TODO: display, PEP8 header and formatting
"""

import numpy as np
import os
import nibabel as nib
from colorama import Fore, Back, Style
import glob
import nibabel.processing
from skimage.transform import resize
from nibabel.viewers import OrthoSlicer3D
from nifti_write import make_nifti


def process_data(fname:str=None, ds_resolution:np.ndarray=None, us_resolution_0:np.ndarray=None,
                 write_file:bool=False): #Load data subject wise
    for nii_name in glob.glob(fname, recursive=True): 
        if 'anat' in nii_name:      # Focusing only on anatomical scans
            if  not 'ds' in nii_name:
                img = nib.load(nii_name).get_fdata()
                ds0 = down_sample_skimage(img, native_res=[1, 1, 1], target_voxel_size=ds_resolution)
                ds1 = down_sample_skimage(img, native_res=[1, 1, 1], 
                                        target_voxel_size=[ds_resolution[1], ds_resolution[2], ds_resolution[0] ])
                ds2 = down_sample_skimage(img, native_res=[1, 1, 1], 
                                        target_voxel_size=[ds_resolution[2], ds_resolution[0], ds_resolution[1]])
                us0 = down_sample_skimage(img, native_res=[1, 1, 1], 
                                        target_voxel_size=us_resolution_0)
                
                if write_file is True:
                    fname_grep = nii_name.split('.nii.gz')
                    fname_ds0 = fname_grep[0] + '_ds0.nii.gz'
                    fname_ds1 = fname_grep[0] + '_ds1.nii.gz'
                    fname_ds2 = fname_grep[0] + '_ds2.nii.gz'
                    fname_us0 = fname_grep[0] +  '_us0.nii.gz'
                    
                    # Debug only
                    # data_vis(img, display_type='raw')
                    print(fname_ds0)
                    print(fname_ds1)
                    print(fname_ds2)
                    print(fname_us0)
                    
                    save_data(fname_ds0, ds0, save_format='nii', res = ds_resolution)
                    save_data(fname_ds1, ds1, save_format='nii', res = [ds_resolution[1], ds_resolution[2], ds_resolution[0]])
                    save_data(fname_ds2, ds2, save_format='nii', res = [ds_resolution[2], ds_resolution[0], ds_resolution[1]])
                    save_data(fname_us0, us0, save_format='nii', res = us_resolution_0)

                    
    return True

def down_sample_nibabel(input_img:float=None, native_res:np.ndarray=None, 
                        target_voxel_size:np.ndarray=None):
        data_ds0  = nibabel.processing.resample_to_output(input_img, 
                            [voxel_size[0],voxel_size[1],voxel_size[2]]).get_fdata()
    
        data_ds1  = (nibabel.processing.resample_to_output(input_img, 
                            [voxel_size[1],voxel_size[2],voxel_size[0]])).get_fdata()
        
        data_ds2  = nibabel.processing.resample_to_output(input_img, 
                            [voxel_size[2],voxel_size[0],voxel_size[1]])
 
        # print(data_ds1.shape)
        # print(data_ds2.shape)
        data_vis(data_im=data_ds0, display_type='raw')
        return data_ds0, data_ds1, data_ds2
   
def down_sample_skimage(input_img:float=None, native_res:np.ndarray=None, 
                        target_voxel_size:np.ndarray=None):
        out_matsize_fact = np.divide(target_voxel_size, native_res)
        out_matsize = np.divide(input_img.shape, out_matsize_fact)
        image_ds = resize(input_img, (int(out_matsize[0]), int(out_matsize[1]), 
                                      int(out_matsize[2])))
        return image_ds
    
def save_data(fname_source:str=None, data:np.ndarray=None, save_format:str=None, res: np.ndarray = None):
    if save_format == 'nii':
        make_nifti(data = data, fname = fname_source, mask = False,
               res = res,  norm=False)

def data_vis(data_im:float=None, display_type:str=None):
    if display_type=='raw':
        s=OrthoSlicer3D(data_im, affine=np.eye(4))
        s._clim = 2048
        s.show()
    
if __name__ == '__main__':
    # topdir = './Data'
    topdir = './QTAB/'
    subjects = os.listdir(topdir)
    ds_resolution = [2.0, 2.0, 5.0]
    us_resolution_0 = [2.0, 2.0, 2.0]     # to enable curriculum learning
    write_file = True
    debug = True
    
    for subject in subjects:
        if 'sub' in subject:
            if '._' in subject:
                 print(Fore.RED + 'some system file - not reading')
            else:
                session_dirs = os.listdir(os.path.join(topdir, subject))
                search_path = os.path.join(topdir, subject) + '/**/*_T2w.nii.gz'
                if '.DS' in session_dirs: # TODO
                     session_dirs-=1
                print(Fore.GREEN + 'Found subject: ' + subject + ' with ' +
                    str(len(session_dirs)) + ' sessions')
                print(Fore.YELLOW + 'Downsampling the data now')
                process_data(search_path, ds_resolution=ds_resolution, us_resolution_0 = us_resolution_0,
                             write_file=write_file)
                print(Fore.YELLOW + 'Completed data processing')
                print(Style.RESET_ALL)

    # Check a few written nifti's with visualization
    if debug is True:
        
        print('Got done with Today :)')
        fname = './QTAB/sub-0001/ses-01/anat/sub-0001_ses-01_T2w.nii.gz'
        data_im = nib.load(fname).get_fdata()
        data_vis(data_im, display_type='raw')
                
                

            
                
        
    
    
    
    
    
    
    