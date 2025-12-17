import numpy as np
import nibabel as nib
from read_kea3d import kea3d
from kea2nifti import make_nifti
from nibabel.viewers import OrthoSlicer3D
import os

def read_lf_data(
    data_folder='Ajay_training/Ajay_training_01',
    output_folder='Ajay_training/Output_nifti',
    subject="34507",
    sub_folder='3DTSE/1',
    file_name='lf_mri.nii.gz'
):
    try:
        # print(f"Data folder: {data_folder}")
        # print(f"Output folder: {output_folder}")
        # print(f"Subject: {subject}")
        # print(f"Sub folder: {sub_folder}")
        # print(f"File name: {file_name}")

        subject_folder = os.path.join(output_folder, subject)
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)

        # Include subfolder name in the output filename for differentiation
        filename = f"lf_mri_{sub_folder.replace('/', '_')}.nii.gz"
        fname_nii = os.path.join(subject_folder, filename)
        print(f"Output NIfTI file will be saved as: {fname_nii}")

        sample_data = kea3d(data_folder=data_folder, sub_folder=sub_folder)
        kspace = sample_data.kspace_gauss_filter
        im = np.abs(np.fft.fftshift(np.fft.fftn((np.fft.fftshift(kspace)))))

        if im is None:
            print("No data found in the specified folder.")
            return None
        
        # s = OrthoSlicer3D(np.abs(im))
        # s.clim = [0, np.abs(1.5 * np.max(np.abs(im)))]
        # s.cmap = 'gray'
        # s.show()

        # print(np.max(np.abs(im)))
        # print("Min value:", np.min(np.abs(im)))
        # print("Data type of np.abs(im):", np.abs(im).dtype)
        # print("Shape of im:", im.shape)
        
        # Make nifti in case of need for further inputs to other software 
        make_nifti(
            im,
            fname=fname_nii,
            mask=False,
            res=[sample_data.res_dim1, sample_data.res_dim2, sample_data.res_dim3],
            dim_info=[0, 1, 2]
        )

        if im is None:
            sample_data = []

        return im, sample_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

if __name__ == "__main__":
    im = read_lf_data()