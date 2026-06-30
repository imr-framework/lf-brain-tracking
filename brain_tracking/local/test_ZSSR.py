import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./src')

from src.ZSSR_master import configs, configs_2, ZSSR
from src.ZSSR_master.ZSSR import *


ds_to_process = 4
target_resolution_fact = [1, 1, 2]
snr_component = False

max_iters = 6000
min_iters = 256
# Define parameter options
widths = [32, 32, 16]
depths = [8, 4, 4]
crop_sizes = [128, 256]
noise_stds = [0.0, 0.2]

import numpy as np
import nibabel as nib

# -----------------------------------------------
# Helper: run ZSSR on one 2D slice
# -----------------------------------------------
def run_zssr_slice(slice2d, configs):
    minv, maxv = slice2d.min(), slice2d.max()
    s = (slice2d - minv) / (maxv - minv + 1e-8)

    s3 = np.stack([s, s, s], axis=-1)

    out = ZSSR(input_img=s3, conf=configs,
                    ground_truth=None, kernels=None).run()

    out2d = out[:, :, 0]
    out2d = out2d * (maxv - minv) + minv
    return out2d


# -----------------------------------------------
# Pass 1: XY SR
# -----------------------------------------------
def zssr_xy(im_srr, configs):
    X, Y, Z = im_srr.shape
    vol_xy = np.zeros((X*2, Y*2, Z), dtype=float)

    for z in range(Z):
        print(f"[XY] Slice {z+1}/{Z}")
        vol_xy[:, :, z] = run_zssr_slice(im_srr[:, :, z], configs)

    return vol_xy


# -----------------------------------------------
# Pass 2: YZ SR
# -----------------------------------------------
def zssr_yz(vol_xy, configs):
    X2, Y2, Z2 = vol_xy.shape
    vol_yz = np.zeros((X2, Y2*2, Z2), dtype=float)

    for x in range(X2):
        print(f"[YZ] Slice {x+1}/{X2}")
        vol_yz[x, :, :] = run_zssr_slice(vol_xy[x, :, :], configs)

    return vol_yz


# -----------------------------------------------
# Pass 3: XZ SR
# -----------------------------------------------
def zssr_xz(vol_yz, configs):
    X3, Y3, Z3 = vol_yz.shape
    vol_xz = np.zeros((X3, Y3, Z3*2), dtype=float)

    for y in range(Y3):
        print(f"[XZ] Slice {y+1}/{Y3}")
        vol_xz[:, y, :] = run_zssr_slice(vol_yz[:, y, :], configs)

    return vol_xz


# -----------------------------------------------
# Full 3D Multi-Plane ZSSR (XY ➜ YZ ➜ XZ)
# -----------------------------------------------
def run_full_3d_zssr(im_srr, configs):
    print("\nRunning 3D ZSSR: XY → YZ → XZ\n")

    vol_xy = zssr_xy(im_srr, configs)
    vol_yz = zssr_yz(vol_xy, configs)
    vol_xz = zssr_xz(vol_yz, configs)

    return vol_xz
