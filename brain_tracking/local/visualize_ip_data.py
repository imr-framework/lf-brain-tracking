import numpy as np
from display_vlf_ni_data import plot_anatomy_nifti
import nibabel as nib

def make_2dpanel(im_axial, im_sag, im_cor, im_sr):
    plot_anatomy_nifti(im=im_axial, output_file='axial.png')
    plot_anatomy_nifti(im=im_sag, output_file='sag.png')
    plot_anatomy_nifti(im=im_cor, output_file='cor.png')
    plot_anatomy_nifti(im=im_sr, output_file='srr.png', coords=(110, 110, 110))
    return

if __name__ == '__main__':
    im_axial = nib.load('axial.nii.gz')
    im_sag = nib.load('sag.nii.gz')
    im_cor = nib.load('cor.nii.gz')
    im_sr = nib.load('srr.nii.gz')

    make_2dpanel(im_axial, im_sag, im_cor, im_sr)



