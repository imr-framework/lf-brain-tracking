import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import data, img_as_float
from display_vlf_ni_data import plot_anatomy_raw, plot_anatomy_nifti
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise



# TODO:
# 1. Denoise
# 2.

def non_local_means_denoising(data_im: np.ndarray = None):
    sigma_est = np.mean(estimate_sigma(data_im))
    patch_kw = dict(patch_size=5,  # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    channel_axis=-1)
    data_im_denoise = denoise_nl_means(data_im, h=0.8 * sigma_est, sigma=sigma_est,
                                       fast_mode=False, **patch_kw)
    return data_im_denoise


if __name__ == '__main__':
    dataFolder = r'./Data/In_vivo/1_avg'
    im_zy_folder = '1_zy_Axial_circshift.npy'  # axial
    im_xy_folder = '1_xy.npy'  # sagittal
    im_zx_folder = '1_zx_Coronal_circshift.npy'  # coronal

    fov = 120, 220, 220

    im_axial = np.abs(np.load(os.path.join(dataFolder, im_zy_folder)))
    im_sag = np.abs(np.load(os.path.join(dataFolder, im_xy_folder)))
    im_cor = np.abs(np.load(os.path.join(dataFolder, im_zx_folder)))

    im_axial_denoise = non_local_means_denoising(data_im=im_axial)
    plot_anatomy_raw(im_axial, clim = [0,  500])
    plot_anatomy_raw(im_axial_denoise, clim = [0, 500])
    plot_anatomy_raw(im_axial - im_axial_denoise, clim=[0, 100])

    #
