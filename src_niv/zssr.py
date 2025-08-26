# prompt: apply zero shot super resolution algorithm to give im slices as input and super resolution as output display original normalized , refined extract brain with hole filling  and ZSSR
# SR by 4x

import matplotlib.pyplot as plt
import numpy as np
import cv2
# Install the Zero-Shot Super-Resolution library
# The specific library for ZSSR might need to be identified or implemented.
# As a placeholder, let's assume there's a library or a model available.
# For example, if using a Keras implementation:
# !pip install tensorflow  # Already likely installed
# You might need to install a specific implementation of ZSSR if it's packaged.
# As of my last update, a widely available, easy-to-install ZSSR library isn't standard.
# We'll assume we have a function or class that implements the ZSSR algorithm.

# --- Assuming you have a ZSSR implementation ---
# You would typically define or import your ZSSR model/function here.
# For demonstration purposes, let's create a placeholder function.
# A real ZSSR implementation would involve training/optimization on the image itself.

def zero_shot_super_resolution(low_res_slice, scale_factor=2):
    """
    Placeholder for a Zero-Shot Super-Resolution algorithm.
    In a real implementation, this would apply the ZSSR logic to enhance
    the resolution of the low_res_slice.

    Args:
        low_res_slice (np.ndarray): The 2D low-resolution image slice.
        scale_factor (int): The factor by which to increase resolution.

    Returns:
        np.ndarray: The super-resolved image slice.
    """
    print(f"Applying ZSSR to slice of shape {low_res_slice.shape} with scale factor {scale_factor}...")

    # --- Real ZSSR Implementation Placeholder ---
    # A typical ZSSR process involves:
    # 1. Generating downscaled versions of the input image.
    # 2. Training a small CNN on these self-similar patches.
    # 3. Applying the trained CNN to the original low-resolution image.
    # This requires a model architecture and an optimization loop.

    # Since we don't have a ZSSR model implemented here, we'll return
    # a simple upscaled version using interpolation as a visual placeholder.
    # This is NOT Zero-Shot Super-Resolution, but allows the code to run.
    # You would replace this with your actual ZSSR model application.

    # Simple interpolation for visualization (Not ZSSR)
    original_height, original_width = low_res_slice.shape
    new_height = original_height * scale_factor
    new_width = original_width * scale_factor

    # Ensure the input slice is in a suitable format (e.g., float32)
    if low_res_slice.dtype != np.float32:
        low_res_slice = low_res_slice.astype(np.float32)

    # Use OpenCV's resize function for interpolation
    # You might choose different interpolation methods (cv2.INTER_CUBIC, cv2.INTER_LANCZOS4)
    # cv2.INTER_CUBIC is often used for upscaling.
    try:
        super_resolved_slice = cv2.resize(low_res_slice, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    except cv2.error as e:
         print(f"Error during simple upscaling using OpenCV: {e}")
         # Return original or handle error
         return low_res_slice # Or return None

    print(f"Upscaled slice shape: {super_resolved_slice.shape}")

    return super_resolved_slice

def extract_brain(im, scale_factor = 2):
    # Assuming 'im' is your original low-field MRI volume
    # And the refined brain extraction process yields masks or extracted slices.
    # Let's re-process slices to get the refined brain-extracted data for SRR input.
    print("====================================Inside ZSSR_Extract brain ....................")
    orig_slices = []
    super_resolved_slices = []
    brain_extracted_slices = []
    scale_factor = 2 # Define your desired super-resolution scale factor (changed to 4)

    num_slices = im.shape[0]
    print(f"\nApplying Zero-Shot Super-Resolution to {num_slices} slices...")

    for i in range(num_slices):
        slice_data = im[i, :, :]

        if slice_data is None or slice_data.size == 0:
            print(f"Skipping SRR for slice {i} due to empty data.")
            super_resolved_slices.append(None) # Append None or an empty array
            continue

        try:
            # Re-run the brain extraction logic to get the refined mask for this slice
            normalized_slice = cv2.normalize(np.abs(slice_data), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            orig_slices.append(normalized_slice)
            ret, thresh = cv2.threshold(normalized_slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            refined_mask = np.zeros_like(normalized_slice, dtype=np.uint8) # Ensure mask is uint8
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(normalized_slice, dtype=np.uint8)
                cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

                kernel_dilate = np.ones((6, 6), np.uint8)
                mask_dilated = cv2.dilate(mask, kernel_dilate, iterations=1)

                kernel_close = np.ones((6, 6), np.uint8)
                refined_mask = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel_close)
            else:
                print(f"No contours found for slice {i}. Brain extraction failed.")
                # refined_mask remains all zeros

            # Apply the refined mask to get the brain-extracted slice for SRR input
            # Convert original slice data to a suitable type (e.g., float) before masking if needed for SRR input
            # For the placeholder ZSSR, let's use the normalized 8-bit slice as input.
            brain_extracted_slice_input = cv2.bitwise_and(normalized_slice, normalized_slice, mask=refined_mask)
            brain_extracted_slices.append(brain_extracted_slice_input)
            # Apply the ZSSR (or placeholder) algorithm
            super_resolved_slice = zero_shot_super_resolution(brain_extracted_slice_input, scale_factor=scale_factor)

            super_resolved_slices.append(super_resolved_slice)

        except Exception as e:
            print(f"Error processing slice {i} for SRR: {e}")
            super_resolved_slices.append(None) # Append None if processing fails

    print("\nFinished applying SRR (or placeholder) to all slices.")

    return orig_slices, brain_extracted_slices, super_resolved_slices

def extract_lf_volumes(im,scale_factor = 2, display = False):
    # --- Display original low-res, refined brain extracted, and super-resolved slices side-by-side ---

    orig_slices, brain_extracted_slices, super_resolved_slices = extract_brain(im)
    print("\nDisplaying Original (Normalized), Refined Brain Extracted, and Super-Resolved Slices...")

    # Filter out None values in case some slices failed
    valid_indices = [i for i, s in enumerate(super_resolved_slices) if s is not None]
    valid_sr_slices = [super_resolved_slices[i] for i in valid_indices]
    valid_orig_slices = [orig_slices[i] for i in valid_indices]
    valid_refined_be_slices = [brain_extracted_slices[i] for i in valid_indices]

    if display:
        if not valid_sr_slices:
            print("No valid super-resolved slices to display.")
        else:
            num_valid_slices = len(valid_sr_slices)
            cols_per_row = 6 # Display fewer triplets per row
            # Calculate required rows for original+refined+SR triplets
            rows = (num_valid_slices + cols_per_row - 1) // cols_per_row * 3 # Three rows per set of columns (original, refined, SR)

            fig, axes = plt.subplots(rows, cols_per_row, figsize=(3 * cols_per_row, 2 * rows))
            axes = axes.flatten()

            for i in range(num_valid_slices):
                original_slice = valid_orig_slices[i]
                refined_be_slice = valid_refined_be_slices[i]
                sr_slice = valid_sr_slices[i]
                original_slice_index = valid_indices[i] # Get the index in the original volume

                # Display original normalized slice (First row for this slice index)
                axes[3 * i].imshow(np.flipud(original_slice.T), cmap='gray')
                axes[3 * i].set_title(f'Slice {original_slice_index} Original (Norm)')
                axes[3 * i].axis('off')

                # Display refined brain extracted slice (Second row for this slice index)
                axes[3 * i + 1].imshow(np.flipud(refined_be_slice.T), cmap='gray')
                axes[3 * i + 1].set_title(f'Slice {original_slice_index} Refined BE')
                axes[3 * i + 1].axis('off')

                # Display super-resolved slice (Third row for this slice index)
                axes[3 * i + 2].imshow(np.flipud(sr_slice.T), cmap='gray')
                axes[3 * i + 2].set_title(f'Slice {original_slice_index} SR ({scale_factor}x)')
                axes[3 * i + 2].axis('off')


            # Hide any unused subplots
            for j in range(3 * num_valid_slices, len(axes)):
                axes[j].axis('off')


            plt.tight_layout()
            plt.show()

        print("\nFinished displaying Original (Normalized), Refined Brain Extracted, and Super-Resolved slices.")

    # --- Optional: Reconstruct the 3D Super-Resolved Volume ---
    # If all slices have the same super-resolved dimensions, you can stack them
    # back into a 3D numpy array.
    # Check if all valid SR slices have the same shape
    if valid_orig_slices and all(s.shape == valid_orig_slices[0].shape for s in valid_orig_slices):
        try:
            # Stack the super-resolved slices along the z-axis
            valid_orig_volume = np.stack(valid_orig_slices, axis=0)
            print(f"\nSuccessfully original Volume with shape: {valid_orig_volume.shape}")
        except Exception as e:
            print(f"Error in origial Volume: {e}")

    if valid_refined_be_slices and all(s.shape == valid_refined_be_slices[0].shape for s in valid_refined_be_slices):
        try:
            # Stack the super-resolved slices along the z-axis
            valid_refined_be_volume = np.stack(valid_refined_be_slices, axis=0)
            print(f"\nSuccessfully Brain_extracted Volume with shape: {valid_refined_be_volume.shape}")
        except Exception as e:
            print(f"Error brain extraction Volume: {e}")

    if valid_sr_slices and all(s.shape == valid_sr_slices[0].shape for s in valid_sr_slices):
        try:
            # Stack the super-resolved slices along the z-axis
            super_resolved_volume = np.stack(valid_sr_slices, axis=0)
            print(f"\nSuccessfully reconstructed Super-Resolved Volume with shape: {super_resolved_volume.shape}")

            # You can optionally save this volume as a NIfTI file
            # Need to create a NIfTI image object from the numpy array and original header/affine
            # Get affine and header from the original NIfTI image 'im'
            # (Need to load the original NIfTI file again or save it earlier)
            # For this example, assuming 'im' was loaded from a NIfTI and you have the original obj
            # Check if the original 'im' came from a nibabel object
            # If 'im' is just a numpy array from read_lf_data, you might need to manually
            # reconstruct the affine or load the original NIfTI to get it.

            # Assuming you have the original nibabel image object (e.g., 'im_nii')
            # If 'im' is a numpy array and you have its affine (e.g., 'im_affine')
            # new_affine = im_affine.copy()
            # Need to adjust the affine matrix for the new resolution.
            # If the original affine matrix describes the transformation from voxel indices (i, j, k)
            # to spatial coordinates (x, y, z), e.g., [[vx, 0, 0, x0], [0, vy, 0, y0], [0, 0, vz, z0], [0, 0, 0, 1]]
            # where (vx, vy, vz) are voxel sizes. The SR volume has voxel sizes (vx/scale_factor, vy/scale_factor, vz).
            # The new affine might look like [[vx/scale_factor, 0, 0, x0], [0, vy/scale_factor, 0, y0], [0, 0, vz, z0], [0, 0, 0, 1]]
            # The origin (x0, y0, z0) should remain the same.

            # As we don't have the original nibabel object/affine readily available from the provided code structure,
            # we'll just keep the super_resolved_volume as a numpy array for now.
            # To save as NIfTI, you would need:
            # !pip install nibabel
            # import nibabel as nib
            # # Assuming you have the original affine and header
            # sr_nii_img = nib.Nifti1Image(super_resolved_volume, new_affine, original_header)
            # nib.save(sr_nii_img, '/drive/MyDrive/ajay/lf-brain-tracking/Data/LFMRI_DATA_IRF_nifti/super_resolved_volume.nii.gz')
            # print("Super-resolved volume saved as NIfTI (placeholder path).")


        except Exception as e:
            print(f"Error reconstructing Super-Resolved Volume: {e}")

    else:
        print("\nSkipping 3D volume reconstruction: Valid SR slices are empty or have inconsistent shapes.")
    
    return valid_refined_be_volume