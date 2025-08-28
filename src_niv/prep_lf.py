import numpy as np
import ants
import pydicom
import nibabel as nib
import cv2
import os
from skimage import exposure, restoration
import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on what's installed
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
from src_niv.prep_data import data_ops
sys.path.append('./data_read_code')
from demo_read_data import read_lf_data

def load_nifti(path):
    return nib.load(path).get_fdata()

def save_nifti(data, reference_path, output_path):
    ref_img = nib.load(reference_path)
    nib.save(nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header), output_path)

def extract_largest_component(volume):
    largest_mask = np.zeros_like(volume)
    for i in range(volume.shape[2]):
        slice_img = volume[:, :, i]
        slice_img_float = slice_img.astype(np.float32)
        slice_bin = np.uint8(slice_img_float > np.percentile(slice_img_float, 20))  # Simple thresholding
        contours, _ = cv2.findContours(slice_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(slice_img, dtype=np.uint8)
            mask = np.ascontiguousarray(mask)
            cv2.drawContours(mask, [largest], -1, 1, thickness=-1)
            largest_mask[:, :, i] = mask
    return volume * largest_mask

# def do_contrast_enhance(im:np.ndarray=None, method:str='custom', fact:float=1.10):
#     im_ce = np.zeros_like(im)
#     for slice in range(im.shape[2]):
#         im_slice = np.squeeze(im[:, :, slice])
#         if method is 'nc':
#             im_slice = 255 * do_norm_im(im_slice)
#             im_slice = im_slice.astype(np.uint8)

#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
#             im_cl = clahe.apply(im_slice)
            
#             # morphological
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
#             # Top Hat Transform
#             topHat = cv2.morphologyEx(im_slice, cv2.MORPH_TOPHAT, kernel)
#             # Black Hat Transform
#             blackHat = cv2.morphologyEx(im_slice, cv2.MORPH_BLACKHAT, kernel)
#             im_cl = im_slice + topHat - blackHat
#         else:
#             # im_slice[np.where(im_slice <1350)] = im_slice[np.where(im_slice <1350)] / fact
#             # im_slice[np.where(im_slice >1550)]  *= fact
#             im_slice[im_slice < 900] = im_slice[im_slice < 900] / fact
#             im_cl = im_slice
#         im_ce[:, :, slice] = im_cl

#     return im_ce

def do_norm_im(im_slice:np.ndarray=0):
    max_slice = np.max(im_slice)
    min_slice = np.min(im_slice)
    im_slice_norm = (im_slice - min_slice) / (max_slice - min_slice)
    return im_slice_norm

def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalizes an image volume to zero mean and unit variance.

    Parameters:
    - img (np.ndarray): Input image array.

    Returns:
    - np.ndarray: Normalized image as float32.
    """
    img = img.astype(np.float32)
    return (img - np.mean(img)) / (np.std(img) + 1e-5)


def resize_mri_volume(volume: np.ndarray, target_shape: tuple, interp_type: int = 1) -> np.ndarray:
    """
    Resizes a 3D MRI volume to a target shape using ANTsPy.

    Parameters:
    - volume (np.ndarray): Input 3D volume, shape (D, H, W) or (H, W, D)
    - target_shape (tuple): Desired shape (X, Y, Z)
    - interp_type (int): Interpolation type (0=nearest, 1=linear, 2=BSpline, etc.)

    Returns:
    - np.ndarray: Resized volume as float32.
    """
    if volume.ndim != 3:
        raise ValueError("Input volume must be a 3D array.")

    volume = volume.astype(np.float32)
    ants_img = ants.from_numpy(volume, spacing=(1.0, 1.0, 1.0))  # Ensure spacing is defined
    resized_img = ants.resample_image(ants_img, target_shape, use_voxels=True, interp_type=interp_type)
    return resized_img.numpy().astype(np.float32)

def bias_correction(volume):
    ants_img = ants.from_numpy(volume.astype(np.float32))
    corrected = ants.n4_bias_field_correction(ants_img)
    return corrected.numpy()

def denoise(volume, method='nlm'):
    if method == 'nlm':
        return ants.denoise_image(ants.from_numpy(volume.astype(np.float32))).numpy()
    elif method == 'tv':
        return restoration.denoise_tv_chambolle(volume, weight=0.1)
    else:
        return volume

def enhance_contrast(volume, method='clahe'):
    enhanced = np.zeros_like(volume)
    for i in range(volume.shape[2]):
        slice_img = volume[:, :, i]
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            norm_slice = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            enhanced[:, :, i] = clahe.apply(norm_slice)
        else:
            enhanced[:, :, i] = exposure.equalize_adapthist(slice_img)
    return enhanced.astype(np.float32)

def normalize(volume):
    volume = volume.astype(np.float32)
    return (volume - np.mean(volume)) / (np.std(volume) + 1e-5)

def register_to_hf(lf_volume, hf_volume):
    fixed = ants.from_numpy(hf_volume.astype(np.float32))
    moving = ants.from_numpy(lf_volume.astype(np.float32))
    reg = ants.registration(fixed, moving, type_of_transform="SyN")
    return reg['warpedmovout'].numpy()

def match_histogram(lf_volume, hf_volume):
    matched = exposure.match_histograms(lf_volume, hf_volume)
    return matched.astype(np.float32)

# === Main pipeline ===
def preprocess_lf_mri(lf_path, hf_path, out_path,
                      denoise_method='nlm', contrast_method='clahe'):

    lf_img = lf_path
    hf_img = hf_path

    lf_img = np.where(lf_img < 5, 0, lf_img)
    print(f"Type of lf_img: {type(lf_img)}")
    print(f"Type of hf_img: {type(hf_img)}")
    print("[1] Loading images...")

    # lf_img = load_nifti(lf_path)
    # hf_img = load_nifti(hf_path)

    print("[2] Extracting brain...")
    brain = extract_largest_component(lf_img)

    # print("[3] Bias field correction...")
    # brain_corrected = bias_correction(brain)

    # print(f"[4] Denoising using {denoise_method}...")
    # brain_denoised = denoise(brain_corrected, method=denoise_method)

    # print(f"[5] Contrast enhancement using {contrast_method}...")
    # brain_enhanced = enhance_contrast(brain_denoised, method=contrast_method)

    # print("[6] Normalizing intensity...")
    # brain_normalized = normalize(brain_enhanced)

    # print("[7] Registering to HF-MRI...")
    # registered = register_to_hf(brain_normalized, hf_img)

    # print("[8] Histogram matching with HF-MRI...")
    # matched = match_histogram(registered, hf_img)

    # print("[9] Saving output...")
    # save_nifti(matched, lf_path, out_path)

    print("✅ Preprocessing complete.")

    return brain

   
if __name__ == "__main__":
    
    # HF_MRI path

    # Define the path to the IRF_3T folder ( High Field Data)
    nhp_base_path = './Data/IRF_3T'
    subject = '26184'  # Example subject number, adjust as needed

    subjects = os.listdir(nhp_base_path)
    subjects = sorted(subjects)
    print(f"Available subjects: {subjects}")
    print(f"Selected subject: {subject}")

    nhp_data_path = f'{nhp_base_path}/{subject}' #truct the full path to the DICOM folder

    # Initialize data object and load data (IRF_3T)
    data_obj = data_ops(nhp_data_path)

    # Retrieve dictionary of 3D volumes (day1 to day5)
    all_volumes = data_obj.data

    # Select and visualize Day 1: 10 slices spaced 10 apart
    day_idx = 1
    volume_26184 = all_volumes[day_idx]

    #LF-MRI path

    data_folder = 'Data/LFMRI_DATA_IRF/IRF_071E_2_C1_20240709/34507_D_minus28'
    output_folder = '/home/ajay/Documents/lf-brain-tracking/Data/LFMRI_DATA_IRF_nifti'
    subject = subject
    sub_folder ='3DTSE/4'
    file_name ='20240829_day2_3DTSC_12.nii'
    name = file_name
    im = read_lf_data(data_folder, output_folder, subject, sub_folder, file_name)
    

    print(f"LF_MRI data processing  started .............")
    print("Max value:", np.max(np.abs(im)))
    print("Min value:", np.min(np.abs(im)))
    print("Data type of np.abs(im):", np.abs(im).dtype)
    print("Shape of im:", im.shape)
    
    num_slices = im.shape[2]
    fig, axes = plt.subplots(2, 8, figsize=(20, 8))
    # fig.suptitle(f'All Axial Slices for {name}\n{subject}\n{Visit_id}\n3DTSE/{subf}', fontsize=16)
    axes = axes.flatten()

    for i in range(16):
        if i < num_slices:
            slice_img = np.flipud(np.abs(lf_img[:, :, i]).T)
            axes[i].imshow(slice_img, cmap='gray')
            axes[i].set_title(f'Slice {i + 1}')
            axes[i].axis('off')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    # plt.savefig(f'Figures/{subject}/{fig_name}')
    plt.show()
    plt.close()    

    

    # low_res_im_ce = do_contrast_enhance(im)
    # low_res_im = low_res_im_ce


    # lf = preprocess_lf_mri(
    #     lf_path=im,
    #     hf_path=volume_26184,
    #     out_path="LF_enhanced_registered.nii.gz",
    #     denoise_method="nlm",         # or "tv"
    #     contrast_method="clahe"       # or "adapthist"
    # )

    num_slices = low_res_im.shape[2]
    fig, axes = plt.subplots(2, 8, figsize=(20, 8))
    # fig.suptitle(f'All Axial Slices for {name}\n{subject}\n{Visit_id}\n3DTSE/{subf}', fontsize=16)
    axes = axes.flatten()

    for i in range(16):
        if i < num_slices:
            slice_img = np.flipud(np.abs(low_res_im[:, :, i]).T)
            axes[i].imshow(slice_img, cmap='gray')
            axes[i].set_title(f'Slice {i + 1}')
            axes[i].axis('off')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    # plt.savefig(f'Figures/{subject}/{fig_name}')
    plt.show()
    plt.close()