from colorama import Fore, Style
import nibabel as nib 
from ZSSR_master import configs, configs_2, ZSSR

def get_recon_config(im_low_res:nib=None):
    im_low_res_data = im_low_res.get_fdata()
    im_low_res_header = im_low_res.header
    im_low_res_voxdim = im_low_res_header.get_zooms()

    target_res = [im_low_res_voxdim[1]/im_low_res_voxdim[0], im_low_res_voxdim[2]/im_low_res_voxdim[0]  ]
    print(Fore.GREEN + 'Upscale factors: ' + str(target_res[0]) + ' ,' +str(target_res[1]))
    recon_conf2 = configs_2.Config()
    recon_conf2.scale_factors = [[target_res[1]/2, target_res[0]]] # two steps 
    recon_conf2.base_change_sfs = [[1.0]]
    print(Style.RESET_ALL)
    
    return im_low_res_data, recon_conf2

if __name__=='__main__':
    im_low_res = nib.load('Data/UCT_Hyperfine/T5/T5/191_45931101_24M/191_45931101_24M_orient1.nii.gz')