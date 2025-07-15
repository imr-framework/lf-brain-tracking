import os
from sort_data2srr_uct import get_T2data
from os.path import isfile, join
import nibabel as nib
from colorama import Fore, Style
from prep4zssr_2D_uct import get_recon_config
from ZSSR_master import configs, configs_2, ZSSR
import numpy as np
from run_ZSSR_2D_ms_uct import run_zssr_workflow

# Workflow for SRR 
def run_uct_srr(dir_name:str=None):
    sub_names = os.listdir(dir_name)
# For a subject folder:
    for sub_name in sub_names:
    # Identify relevant T2 dicoms, read native acquisitions 
        status = get_T2data(dir_name=dir_name, sub_name=sub_name, debug=False)
        
        # For each, do ZSSR to match SliceThickness to PixelSpacing
        # Low-res is always 0 dim ---> slice, freq, phase
        # Good to target ---> freq, freq, freq in two steps
        if status ==0:
            for orient in range(1, 4):
                im_low_res = nib.load(join(dir_name, sub_name, sub_name + '_orient'+str(orient)+'.nii.gz'))
                im_low_res_data, recon_config  = get_recon_config(im_low_res)     
                im_high_res = run_zssr_workflow(im_low_res_data, recon_config)
            # Write niftis with masks

    # Align the three views - https://github.com/nipy/nireg/blob/master/examples/affine_registration.py

    # Save as niftis

    ## Run segmentation

    # Do NiftyMic

    ## Run segmentation

    # Do ZSSR to get 1mm isotropic

    # Run segmentation

if __name__=='__main__':
    debug_mode = False
    # recon_conf = configs.Config()
    dir_name = './Data/UCT_Hyperfine/T5/T5/'
    run_uct_srr(dir_name=dir_name)