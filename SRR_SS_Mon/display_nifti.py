import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# List of NIfTI files to display
nifti_files = [
    "Data/POCEMR001_T1_zssr_noise_newFalse.nii.gz",
    "Data/ULC_img enhancement/Training data/POCEMR001/64mT/POCEMR001_T1.nii.gz",
    "Data/ULC_img enhancement/Training data/POCEMR001/3T/POCEMR001_T1.nii.gz"
]

ref_file = nifti_files[2]  # Reference: 3T scan
ref_img = nib.load(ref_file)
ref_data = ref_img.get_fdata()

# Normalize each NIfTI file's data to [0, 255] and print min/max after normalization
for file_path in nifti_files:
    img = nib.load(file_path)
    data = img.get_fdata()
    norm_data = 255 * (data - data.min()) / (data.max() - data.min())
    print(f"{file_path}: min={norm_data.min():.4f}, max={norm_data.max():.4f}")

for idx, file_path in enumerate(nifti_files[:2]):
    print(f"\nComparing: {file_path} with reference: {ref_file}")
    img = nib.load(file_path)
    data = img.get_fdata()

    # Normalize both reference and current data to [0, 255]
    norm_ref_data = 255 * (ref_data - ref_data.min()) / (ref_data.max() - ref_data.min())
    norm_data = 255 * (data - data.min()) / (data.max() - data.min())

    # Print voxel size
    voxel_size = img.header.get_zooms()
    print(f"Voxel size: {voxel_size}")
    print(f"Data shape: {data.shape}")

    # Ensure shapes match
    if norm_data.shape != norm_ref_data.shape:
        print(f"Shape mismatch: {norm_data.shape} vs {norm_ref_data.shape}")
        continue

    # Compute PSNR
    psnr_value = psnr(norm_ref_data, norm_data, data_range=255)
    print(f"PSNR: {psnr_value:.2f}")

    # Compute SSIM (multichannel=False for 3D)
    ssim_value = ssim(norm_ref_data, norm_data, data_range=255)
    print(f"SSIM: {ssim_value:.4f}")

    # Display current file using OrthoSlicer3D
    print(f"\nDisplaying: {file_path}")
    OrthoSlicer3D(data).show()

# Display reference 3D image using OrthoSlicer3D
print(f"\nDisplaying reference: {ref_file}")
OrthoSlicer3D(ref_data).show()