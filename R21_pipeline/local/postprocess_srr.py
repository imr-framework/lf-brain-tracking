import numpy as np
import cv2
import nibabel as nib
from display_vlf_ni_data import plot_anatomy_raw, plot_anatomy_nifti
from nibabel.viewers import OrthoSlicer3D
import scipy.ndimage
from kea2nifti import make_nifti
from nilearn.plotting import plot_anat
import matplotlib.pyplot as plt


def do_sharpen(im_srr: np.ndarray = None, kernel: np.ndarray = None):
    nx, ny, nz = im_srr.shape
    im_srr_sharp = np.zeros_like(im_srr)
    if kernel is None:
      kernel = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])

    for z in range(nz):
      # im_srr_sharp[:, :, z] = cv2.filter2D(np.squeeze(im_srr[:, :, z]), -1, kernel)
      lp_image = cv2.Laplacian(np.squeeze(im_srr[:, :, z]), cv2.CV_64F)
      im_srr_sharp[:, :, z] = np.squeeze(im_srr[:, :, z]) - 0.1 * lp_image

    return im_srr_sharp


def resample_axial(im: np.ndarray = None, resamp_fact: int = None):
    im_resampled = np.zeros_like(im, shape=(resamp_fact * im.shape[0], resamp_fact * im.shape[1], im.shape[2]))
    for sl in range(im.shape[2]):
      x = np.squeeze(im[:, :, sl])
      x_resampled = scipy.ndimage.zoom(x, 2, order=3)
      im_resampled[:, :, sl] = x_resampled
    return im_resampled


if __name__ == "__main__":
    fname = '../Data/In_vivo/2_avg/reoriented_skull_stripped_raw_srr7_reoriented.nii.gz'
    im_srr_all = nib.load(fname)
    im_srr = im_srr_all.get_fdata()
    print(im_srr.shape)

    # Resample by a factor of 2 in the axial plane
    resamp_fact = 2
    im_resampled = resample_axial(im=im_srr, resamp_fact=resamp_fact)
    print(im_resampled.shape)

    # Make nifti for future use
    make_nifti(im_resampled, fname='resampled_invivo_2mm.nii.gz', mask=False, res=[1, 1, 2],
               dim_info=[1, 2, 0])

    # Visualize the nifti file
    # plot_anatomy_nifti(im='resampled_invivo_2mm.nii.gz', output_file='test.png', )
    cut_coords = (range(38, 50, 5))  # Go up to 100; 38,50,54,68; hit 59 for axial
    plot_anat('resampled_invivo_2mm.nii.gz', display_mode='z', cut_coords=(cut_coords),
    output_file='z_stack_resampled_invivo_2mm_1.png', annotate=False, vmin=0, vmax=2048 + 384)
    plot_anat('resampled_invivo_2mm.nii.gz', display_mode='z', cut_coords=(cut_coords), vmin=0, vmax=2048 + 384)
    plt.show()

    # # Display
    # s = OrthoSlicer3D(im_resampled, affine=np.eye(4))
    # s.clim = [0, 2048]
    # s.show()
