from nilearn.plotting import plot_anat, plot_img, plot_stat_map, show
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# from skimage.util.montage import montage2d

def plot_anatomy_nifti(im:str='test.nii.gz', output_file: str = 'test.png', coords = None, annotate = True):  # Needs a nifti
    im_load = nib.load(im)
    if coords == None:
      nx, ny, nz = im_load.shape
      coords = tuple((int(nx/2), int(ny/2), int(nz/2)))
    plot_anat(im, colorbar=False, cbar_tick_format="%i", display_mode='ortho',
              vmin=0, vmax=1024+384, draw_cross=False, cut_coords=coords, output_file=output_file,
              annotate=annotate)
    plt.show()


def plot_anatomy_raw(im, clim = [0, 2048]):
    s = OrthoSlicer3D(im, affine=np.eye(4))
    s.clim = clim
    s.show()
    



# fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
# ax1.imshow(montage2d(test_image), cmap ='bone')
# fig.savefig('ct_scan.png')

if __name__ == "__main__":
    # nifti_name = 'Data/In_vivo/2_avg/circ-shifted/srr_7.nii.gz'
    nifti_name = 'cor.nii.gz'
    plot_anatomy_nifti(im = nifti_name, output_file='cor.png', coords=(46, 17, 56),  annotate=False)
    # plot_anat(nifti_name, display_mode='mosaic')
    # plt.show()

# ax - (17, 59, 57)
# sag - (56, 60, 17)
# cor - (46, 17, 56)
# srr - (122, 118, 118)
