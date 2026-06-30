import numpy as np
from scipy.interpolate import RegularGridInterpolator
from display_vlf_ni_data import plot_anatomy_raw
from yaml import warnings
from phantominator import shepp_logan


def create_object(res: float = 2, FOV: np.ndarray = [160.0, 220.0, 220.0], phantom: str = 'SL3D'):
    mat_obj = (np.divide(FOV, res)).astype(int)

    if phantom == 'SL3D':
      # MR phantom (returns proton density, T1, and T2 maps)
      M0, T1, T2 = shepp_logan((mat_obj[0], mat_obj[1], mat_obj[2]), MR=True)
      object_im = T2
    else:
      object_im = 0
    return object_im


def down_res(object_im: np.ndarray = 0, res: float = 2, dres: float = 6, axis_dres: int = 0):
    mat_im = object_im.shape
    mat_im_new = list(mat_im)
    dres_fact = np.divide(dres, res).astype(int)
    mat_im_new[axis_dres] = np.divide(mat_im[axis_dres], dres_fact).astype(int)

    steps = [res, res, res]  # original step sizes
    x, y, z = [steps[k] * np.arange(object_im.shape[k]) for k in range(3)]  # original grid
    f = RegularGridInterpolator((x, y, z), object_im)  # interpolator
    dx, dy, dz = 0, 0, 0

    if axis_dres == 0:
      dx, dy, dz = dres, res, res  # new step sizes
    elif axis_dres == 1:
      dx, dy, dz = res, dres, res  # new step sizes
    elif axis_dres == 2:
      dx, dy, dz = res, res, dres  # new step sizes
    else:
      warnings('Failed to identify axis')

    new_grid = np.mgrid[0:x[-1]:dx, 0:y[-1]:dy, 0:z[-1]:dz]  # new grid
    new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))  # reorder axes for evaluation
    object_im_dres = f(new_grid)
    return object_im_dres


if __name__ == '__main__':
    FOVx, FOVy, FOVz = 160.0, 220.0, 220.0
    object_im = create_object(res=2, FOV=[FOVx, FOVy, FOVz], phantom='SL3D')
    plot_anatomy_raw(object_im)
    im_ax = down_res(object_im, res=2, dres=6, axis_dres=0)  # case axial = X
    plot_anatomy_raw(im_ax)
