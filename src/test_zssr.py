# Import necessary libraries
import sys
sys.path.append('./Code/LFsim')
import os
import re
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print(tf.config.list_physical_devices('GPU'))  # Check if GPU is visible
print(tf.config.list_physical_devices('CPU'))  # Check if CPU is visible

if tf.config.list_physical_devices('GPU'):
    print("CUDNN detected!")
else:
    print("No CUDNN detected!")



import nibabel as nib
import os
from LFsim.LF_sim_QTAB import low_field_simulator
from do_zssr_collage import do_mask_image, do_ZSSR_steps, do_zssr_recon_slices
from nifti_write import make_nifti
from nibabel.viewers import OrthoSlicer3D
from LFsim.LF_simulation_functions import read_nifti
from colorama import Fore, Back, Style

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))





# Define the path to the QTAB folder
qtab_path = '/Users/niyathigirish/Documents/JHU/LFTracking2/lf-brain-tracking/Data/QTAB'
subjects = os.listdir(qtab_path)
viewing = False
# ds_to_process = 12
target_resolution_fact = [1, 1, 2]
snr_component = True
# img = nib.load('/home/mri4all/Documents/Tools/SRR/Data/QTAB/sub-0011/ses-01/anat/sub-0011_ses-01_T2w_zssr.nii.gz').get_fdata()
# OrthoSlicer3D(img).show()

# For each subject and session in QTAB, the script will:
for subject in subjects:
    if re.fullmatch(r"sub-\d{4}", subject):
        subject_num = int(subject.split('-')[1])
        if 17 <= subject_num <= 25:

            subject_path = os.path.join(qtab_path, subject)
            sessions = sorted(os.listdir(subject_path))  # Ensures consistent ordering
            print(f"\n📁 Subject: {subject}")
            print("🔎 Sessions found:", sessions)

            for session in sessions:
                if 'ses' in session:
                    print("Checking session folder:", session)
                    print(Fore.GREEN + 'Processing: ', subject, session + Style.RESET_ALL)
                    fname = subject + '_'+ session + '_' + 'T2w.nii'
                    anat_dir = os.path.join(qtab_path, subject, session, 'anat')
                    if os.path.exists(anat_dir):
                        print(f"📂 Contents of {anat_dir}:", os.listdir(anat_dir))
                    else:
                        print(f"❌ anat folder does not exist: {anat_dir}")
                    nifti_file = os.path.join(anat_dir, fname)
                    print("Looking for NIfTI file at:", nifti_file)
                    if os.path.exists(nifti_file):
                        # 2. Pass it through the low field simulator: target resolution - 2mm isotropic

                        nifti_data = nib.load(nifti_file)
                        hdr        = nifti_data.header
                        pixdim  = hdr['pixdim'] 


                        im_lf_sim = low_field_simulator(nifti_file, save_nifti=True, 
                                                        target_resolution_fact= target_resolution_fact,
                                                        snr_component=False)    
                        if viewing == True:
                            OrthoSlicer3D(im_lf_sim).show()
                        # 3. Pass it through the ZSSR algorithm
                        print('Passing through ZSSR')
                        im_lf_sim_zssr = do_zssr_recon_slices(img=im_lf_sim, fname=nifti_file, 
                                            do_preprocess = True, do_postprocess=True, 
                                            padding=False, mask_image=False, reuse_mask=False, 
                                            target_resolution_fact= target_resolution_fact)
                        # 4. Save the output as a nifti file
                        zssr_fname= nifti_file[:-7] + '_zssr_noise_'+ str(snr_component) +'.nii.gz'
                        print("ZSSR output will be saved to:", zssr_fname)
                        print("ZSSR image shape:", None if im_lf_sim_zssr is None else im_lf_sim_zssr.shape)
                        make_nifti(im_lf_sim_zssr, fname =  zssr_fname, mask=False, 
                        res=[pixdim[1], pixdim[2], pixdim[3]], dim_info=[0, 1, 2])
                        if os.path.exists(zssr_fname):
                            print("✅ Successfully saved:", zssr_fname)
                        else:
                            print("❌ File not written:", zssr_fname)
                        if viewing == True:
                            OrthoSlicer3D(im_lf_sim_zssr).show()
                    else:
                        print("❌ NIfTI file does not exist:", nifti_file)

