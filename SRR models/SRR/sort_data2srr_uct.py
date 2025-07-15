# This file performs ZSSR 2D SRR slice by slice for low field MRI
import dicom2nifti.convert_dicom
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
from os import listdir
from os.path import isfile, join
from nifti_write import make_nifti

def do_norm_im(im_slice:np.ndarray=0):
    max_slice = np.max(im_slice)
    min_slice = np.min(im_slice)
    im_slice_norm = (im_slice - min_slice) / (max_slice - min_slice)
    return im_slice_norm

def get_T2data(dir_name: str = None, sub_name: str = None, debug: bool = False):
    dir_num = 0
    for f in listdir(join(dir_name, sub_name)):
        if isfile(join(dir_name, sub_name, f)):
          if '.dcm' in f:
            ds = dicom.dcmread(join(dir_name, sub_name, f))  #Check for pixel data - "PixelData" in ds
            if 'Image Storage' in ds.SOPClassUID.name:
                if 'T2-Weighted' in ds.PulseSequenceName or 'T2 (AXI) (T2 Brain Segmentation)' in ds.PulseSequenceName:
                    print(Fore.RED + 'Pulse Sequence: ' + ds.PulseSequenceName)
                    print(Fore.GREEN + 'Slice Thickness: '+ str(ds.SliceThickness))
                    print(Fore.BLUE + 'Matrix size: ' + str(ds.pixel_array.shape))
                    dir_num  += 1
                    data = ds.pixel_array
                    make_nifti(data=data, fname=join(dir_name, sub_name, sub_name + '_orient'+str(dir_num)+'.nii.gz'), mask=False,
                            res=[ds.PixelSpacing[0], ds.PixelSpacing[1], ds.SliceThickness], dim_info=[0, 1, 2]) # slice, freq, phase      
                    # data[data<100]=0 # create a mask
                    # data[data>=100]=1 # create a mask
                    if debug is True:
                        im = nib.load(join(dir_name, sub_name, sub_name + '_orient'+str(dir_num)+'.nii.gz')).get_fdata()
                        s = OrthoSlicer3D(im)
                        s.show()
        print(Style.RESET_ALL)
        if dir_num == 0:
            status = 1
        else:
            status = 0
        return status

if __name__=='__main__':
    debug_mode = False
    recon_conf = configs.Config()
    dir_name = './Data/UCT_Hyperfine/T5/T5/'
    sub_name = '191_11561329_24M'
    debug = False
    get_T2data(dir_name=dir_name, sub_name=sub_name, debug=True)

    


    



   