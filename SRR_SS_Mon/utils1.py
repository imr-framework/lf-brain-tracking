import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def resample_nifti(in_file, out_file, target_spacing=(1.6, 1.6, 1.0)):
    # Load image
    img = nib.load(in_file)
    data = img.get_fdata()
    affine = img.affine
    header = img.header.copy()

    # Original voxel spacing
    original_spacing = header.get_zooms()[:3]

    # Compute zoom factors
    zoom_factors = np.array(original_spacing) / np.array(target_spacing)

    # Resample
    resampled_data = zoom(data, zoom_factors, order=3)  # cubic interpolation

    # Update affine
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = target_spacing[i] * np.sign(affine[i, i])

    # Create new header with updated zooms
    new_header = header.copy()
    new_header.set_zooms(target_spacing)

    # Create new image
    new_img = nib.Nifti1Image(resampled_data, new_affine, header=new_header)

    # Save
    nib.save(new_img, out_file)
    print(f"✅ Resampled image saved: {out_file}")
    print(f"Original spacing: {original_spacing}, New spacing: {target_spacing}")

# Example
resample_nifti(
    "Data/ULC_img enhancement/Training data/POCEMR001/64mT/POCEMR001_T1.nii.gz",
    "resampled_1p6_1p6_1.nii.gz"
)