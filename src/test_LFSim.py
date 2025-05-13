import sys
sys.path.append('./LFsim')
import os
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
import pydicom
import numpy as np

from pydicom.filereader import dcmread
# import matplotlib
# matplotlib.use('GTkAgg')
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print(tf.config.list_physical_devices('GPU'))  # Check if GPU is visible
print(tf.config.list_physical_devices('CPU'))  # Check if CPU is visible

if tf.config.list_physical_devices('GPU'):
    print("CUDNN detected!")
else:
    print("No CUDNN detected!")

from LF_sim_QTAB import low_field_simulator
from do_zssr_collage import do_mask_image, do_ZSSR_steps, do_zssr_recon_slices
from nifti_write import make_nifti
from nibabel.viewers import OrthoSlicer3D
from LF_simulation_functions import read_nifti
from colorama import Fore, Back, Style

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define the path to the IRF_3T folder
irf_3t = './Data/IRF_3T'
subjects = os.listdir(irf_3t)

viewing = True
ds_to_process = 4
target_resolution_fact = [2, 2, 5]
snr_component = True
print   (subjects)

# load_dicom_slices
def load_dicom_slices(dicom_dir):
    slices = []
    slice_num = 0

    for filename in sorted(os.listdir(dicom_dir)):
        filepath = os.path.join(dicom_dir, filename)
        if os.path.isfile(filepath) and not filename.startswith("."):
            try:
                ds = dcmread(filepath)
                if hasattr(ds, 'SliceLocation') or hasattr(ds, 'ImagePositionPatient'):
                    slices.append(ds)
                    slice_num += 1
                    print(f"Loaded slice {slice_num}: {filename}")
            except Exception as e:
                print(f"Skipping file {filename}: {e}")
    # Sort by ImagePositionPatient[2] or SliceLocation
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') else float(x.SliceLocation))
    return slices

# Convert DICOM Slices to NIfTI
def dicom_to_nifti(dicom_dir, output_path):
    slices = load_dicom_slices(dicom_dir)
    image_shape = list(slices[0].pixel_array.shape)
    image_shape.append(len(slices))
    volume3d = np.zeros(image_shape, dtype=np.int16)

    for i, s in enumerate(slices):
        volume3d[:, :, i] = s.pixel_array

    # Get voxel spacing
    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = float(slices[0].SliceThickness)
    voxel_spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), slice_thickness)
    
    # Print out the DICOM metadata (header)
    print(f"Loaded {len(slices)} slices from {dicom_dir}")
    print(f"Voxel spacing: {voxel_spacing}")
    print(f"Image shape: {image_shape}")
    print(f"Volume shape: {volume3d.shape}")

    # Print DICOM metadata
    print(Fore.GREEN + 'PROCESSING SLICE DICOM METADATA' + Style.RESET_ALL)
    print(f"Patient Position: {slices[0].PatientPosition}")
    print(f"Slice Location: {slices[0].SliceLocation}")
    print(f"Image Position Patient: {slices[0].ImagePositionPatient}")
    print(f"Image Orientation Patient: {slices[0].ImageOrientationPatient}")
    print(f"Instance Number: {slices[0].InstanceNumber}")

    print(f"Study Date: {slices[0].StudyDate}")
    print(f"Study Time: {slices[0].StudyTime}")
    print(f"Modality: {slices[0].Modality}")
    print(f"Patient Name: {slices[0].PatientName}")
    print(f"Patient ID: {slices[0].PatientID}")
    print(f"Slice Thickness: {slices[0].SliceThickness} mm")
    print(f"Study Description: {slices[0].StudyDescription}")
    print(f"Series Instance UID: {slices[0].SeriesInstanceUID}")

    # Create affine matrix
    affine = np.diag(voxel_spacing + (1.0,))
    nifti_img = nib.Nifti1Image(volume3d, affine)
    nib.save(nifti_img, output_path)
    print(f"NIfTI file saved to {output_path}")
    return nifti_img

# For each subject and session in QTAB, the script will:
for subject in subjects:
    
    # 1. Read the dicom file        
    sessions = os.listdir(irf_3t + '/' + subject)

    for session in sessions:
        if 'T2.2D' in session:
            print(Fore.GREEN + 'Processing: ', subject, session + Style.RESET_ALL)
            
            # # Example usage
            dicom_dir = irf_3t + '/' + subject + '/' + session
            output_dir = "./Data/IRF_3T_LFsim" + '/' + subject
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_nifti = output_dir + '/' + session + ".nii.gz"

            nifti_data = dicom_to_nifti(dicom_dir, output_nifti)
            nifti_file = output_nifti
            

            if os.path.exists(nifti_file):
                # 2. Pass it through the low field simulator: target resolution - 2mm isotropic               

                # Read information from NIfTI data header
                nifti_data = nib.load(nifti_file)
                hdr = nifti_data.header
                pixdim  = hdr['pixdim']

                # Print information from NIfTI data header
                print(Fore.GREEN + 'PROCESSING NIFTI FILE METADATA' + Style.RESET_ALL)
                print("NIfTI Header Information:")
                print("Dimensions:", hdr.get_data_shape())
                print("Voxel Sizes:", hdr.get_zooms())
                print("Data Type:", hdr.get_data_dtype())
                print("Intent:", hdr.get_intent())
                print("Affine Matrix:")
                print(hdr.get_qform())

                im_lf_sim = low_field_simulator(nifti_file, save_nifti=True, 
                                                target_resolution_fact= target_resolution_fact,
                                                snr_component=False)  
                
                
                # if viewing == True:
                #     print(Fore.GREEN + 'Viewing: ', subject, session + Style.RESET_ALL)
                #     OrthoSlicer3D(im_lf_sim).show()
