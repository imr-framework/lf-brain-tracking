import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2


def convert2unit8(img2):
    img2 = 255 * (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    return img2


if __name__=='__main__':
    datadir = '/home/mri4all/Documents/Tools/Data/MW'
    
    nii_name = './MW/srr_7.nii.gz'

    ax_name = './MW/axial_circshift_1yz.npy'
    cor_name = './MW/coronal_circshift_1zx.npy'
    sag_name = './MW/sagittal_circshift_1yx.npy'

    # Read a nifti
    img = nib.load(nii_name).get_fdata()
    nx, ny, nz = img.shape

    img2 = convert2unit8(img)
    img2_nii = np.abs(np.reshape(img2, (nx,  ny * nz), order='C'))

    # Write them as color PNGs for now
    img2srr2 = np.zeros((nx, ny * nz, 3), dtype=int)
    img2srr2[:, :, 0] = img2_nii.astype(int) 
    img2srr2[:, :, 1] = img2_nii.astype(int) 
    img2srr2[:, :, 2] = img2_nii.astype(int) 
    print(img2srr2.shape)
    cv2.imwrite('./Code/ZSSR-master/test_data/brain_MW.png', img2srr2)


    # ax_img = convert2unit8(np.abs(np.load(ax_name)))
    # cor_img = convert2unit8(np.abs(np.load(cor_name)))
    # sag_img = convert2unit8(np.abs(np.load(sag_name)))

    # nx, ny, nz = ax_img.shape
    # print(nx, ny, nz)

    # # Reshape them to have one high res and one res dimension
    # ax_img2 = np.abs(np.reshape(ax_img, (nx,  ny * nz), order='C'))
    # cor_img2 = np.abs(np.reshape(cor_img, (nx,  ny * nz), order='C'))
    # sag_img2 = np.abs(np.reshape(sag_img, (nx,  ny * nz), order='C'))

    # # Visualize them before writing as a DEBUG step
    # plt.imshow(ax_img[:, :, 9])
    # plt.set_cmap('gray')
    # plt.show()

    # plt.imshow(np.squeeze(ax_img2[:, 990:990+110]))
    # plt.set_cmap('gray')
    # plt.show()

    
    # # Write them as color PNGs for now
    # img2srr = np.zeros((nx, ny * nz, 3), dtype=int)
    # img2srr[:, :, 0] = ax_img2.astype(int) 
    # img2srr[:, :, 1] = cor_img2.astype(int) 
    # img2srr[:, :, 2] = sag_img2.astype(int) 
    # print(img2srr.shape)
    # cv2.imwrite('./Code/ZSSR-master/test_data/brain_MW.png', img2srr)

   