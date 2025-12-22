import numpy as np
from scipy.ndimage import gaussian_filter, zoom

def degrade_to_lf(vol):
    vol = gaussian_filter(vol, sigma=1.5)
    # print("After Gaussian filter shape:", vol.shape)
    vol = zoom(vol, 0.5, order=1)
    # print("After zoom 0.5 shape:", vol.shape)
    vol = zoom(vol, 2.0, order=1)
    # print("After zoom 2.0 shape:", vol.shape)
    noise = np.random.normal(0, 0.08, vol.shape)
    return vol + noise
