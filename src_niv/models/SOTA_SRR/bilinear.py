import SimpleITK as sitk
import matplotlib as plt

def bicubic_resize_3d(volume, new_spacing=(1.0, 1.0, 1.0)):
    """
    Resample a 3D volume with tricubic (bicubic extension) interpolation.
    
    volume: SimpleITK.Image (3D MRI/CT)
    new_spacing: desired voxel spacing (dx, dy, dz)
    """
    # original spacing
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    
    # new size = old_size * (old_spacing / new_spacing)
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)   # ~tricubic
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(volume.GetDirection())
    resampler.SetOutputOrigin(volume.GetOrigin())
    
    return resampler.Execute(volume)

def train_bilinear(lf_image_volume, save_path="lf_mri_bicubic.nii.gz"):
    # Load low-field MRI
    # img = sitk.ReadImage(lf_image_path)

    # Upsample with bicubic (B-spline) interpolation
    up_img = bicubic_resize_3d(lf_image_volume, new_spacing=(1.0, 1.0, 2.0))

    # Save result
    # sitk.WriteImage(up_img, save_path)
    print(f"Upsampled image saved at: {save_path}")

    # Convert to numpy for visualization (pick middle slice in Z)
    img_np = sitk.GetArrayFromImage(img)
    up_img_np = sitk.GetArrayFromImage(up_img)

    mid_z_orig = img_np.shape[0] // 2
    mid_z_up = up_img_np.shape[0] // 2

    # Plot original vs upsampled slice
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img_np[mid_z_orig,:,:], cmap="gray")
    plt.title("Original LF MRI")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(up_img_np[mid_z_up,:,:], cmap="gray")
    plt.title("Upsampled (Bicubic)")
    plt.axis("off")

    plt.show()

    return up_img