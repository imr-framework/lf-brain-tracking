import sys
sys.path.append('./LFsim')
import os
import nibabel as nib
from nilearn import plotting
from nibabel.viewers import OrthoSlicer3D
import tensorflow as tf
# import matplotlib.pyplot as plt
import pydicom
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on what's installed

from pydicom.filereader import dcmread
# import matplotlib
# matplotlib.use('gtk4agg')
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print(tf.config.list_physical_devices('GPU'))  # Check if GPU is visible
print(tf.config.list_physical_devices('CPU'))  # Check if CPU is visible

if tf.config.list_physical_devices('GPU'):
    print("CUDNN detected!")
else:
    print("Display the data using OrthoSlicer3D")
    OrthoSlicer3D(img.dataobj).show()
    plotting.plot_anat(img, title="3D TSC Image")
    plt.show()

# Note: Undo CUDNN detected!")

from LF_sim_QTAB import low_field_simulator
from do_zssr_collage import do_mask_image, do_ZSSR_steps, do_zssr_recon_slices
from nifti_write import make_nifti
from LF_simulation_functions import read_nifti
from colorama import Fore, Back, Style

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define the path to the IRF_3T folder
irf_3t = './Data/IRF_3T'
subjects = os.listdir(irf_3t)
subjects.sort()
print(subjects[0:1])
viewing = True
ds_to_process = 4
target_resolution_fact = [1, 1, 2]
snr_component = True

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

# For each subject and session in IRF_3T, the script will:
for subject in subjects[0:1]:
    
    # 1. Read the dicom file        
    sessions = os.listdir(irf_3t + '/' + subject)
    for session in sessions:
        if 'T2_n100' in session:
            print(Fore.GREEN + 'Processing: ', subject, session + Style.RESET_ALL)
            
            # # Example usage
            dicom_dir = irf_3t + '/' + subject + '/' + session
            output_dir = "./Data/IRF_3T_LFsim" + '/' + subject
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_nifti = output_dir + '/' + session + ".nii.gz"

            if not os.path.exists(output_nifti):
                print(Fore.GREEN + f"NIfTI file does not exist. Creating: {output_nifti}" + Style.RESET_ALL)
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

                if viewing == True:
                    print(Fore.GREEN + 'Viewing: ', subject, session + Style.RESET_ALL)
                    # data = nifti_data.get_fdata()
                    # OrthoSlicer3D(data).show()
                    view = plotting.view_img(nifti_data, title="High Field Image")
                    view.open_in_browser()

                im_lf_sim = low_field_simulator(nifti_file, save_nifti=True, 
                                                target_resolution_fact= target_resolution_fact,
                                                snr_component=False)
                
                simulated_nifti = output_dir + '/' + session + "_LF_simulated.nii.gz"

                nifti_data_lf = nib.load(simulated_nifti)
                hdr = nifti_data_lf.header
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

                if viewing == True:
                    # print(Fore.GREEN + 'Viewing: ', subject, session + Style.RESET_ALL)
                    # OrthoSlicer3D(im_lf_sim).show()
                    # Display using interactive browser-based viewer
                    view = plotting.view_img(nifti_data_lf, title="LF Simulated Image")
                    view.open_in_browser()

            #     # 3. Pass it through the ZSSR algorithm
            #     print('Passing through ZSSR')
            #     im_lf_sim_zssr = do_zssr_recon_slices(img=im_lf_sim, fname=nifti_file, 
            #                         do_preprocess = True, do_postprocess=True, 
            #                         padding=False, mask_image=False, reuse_mask=False, 
            #                         target_resolution_fact= target_resolution_fact)
                
            #     # 4. Save the output as a nifti file
            #     zssr_fname= nifti_file[:-7] + '_zssr_noise_'+ str(snr_component) +'.nii.gz'
            #     make_nifti(im_lf_sim_zssr, fname =  zssr_fname, mask=False, 
            #     res=[pixdim[1], pixdim[2], pixdim[3]], dim_info=[0, 1, 2])

            #     if viewing == True:
            #         OrthoSlicer3D(im_lf_sim_zssr).show()
            # else:
            #     print(Fore.RED + "Did not find file")