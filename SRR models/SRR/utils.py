import nibabel as nib
import numpy as np
from skimage.restoration import denoise_wavelet
from skimage.filters import unsharp_mask
import matplotlib.pyplot as plt
from nifti_write import make_nifti

def preprocess_img_nhp(data,  debug=False):
    # Load the NIfTI file
   
    # data = np.float64(input_data) / np.max(input_data)
    # Process each slice
    processed_slices = []
    for i in range(data.shape[2]):
        slice_ = data[:, :, i]
        
        # Denoise the slice
        # denoised_slice = denoise_wavelet(slice_, 
                                        #    sigma=0.1,
                                        #     method='BayesShrink',
                                        #     mode='soft',
                                        #     )

            
        # Apply unsharp mask
        
        sharpened_slice = unsharp_mask(slice_, radius=2, amount=1, preserve_range=False, channel_axis=None)
        # processed_slice = denoise_wavelet(sharpened_slice, 
                                        #    sigma=0.01,
                                        #     method='BayesShrink',
                                        #     mode='soft',
                                        #     )
        processed_slice = sharpened_slice
        
        if debug:
            plt.subplot(1, 3, 1)
            plt.imshow(slice_, cmap='gray')
            plt.subplot(1, 3, 2)
            plt.imshow(denoised_slice, cmap='gray')
            plt.subplot(1, 3, 3)
            plt.imshow(sharpened_slice, cmap='gray')
            plt.show()
        
        
        processed_slices.append(processed_slice)
    
    # Stack the processed slices back into a 3D array
    processed_data = np.stack(processed_slices, axis=2)
    print(np.max(processed_data))
    
    return processed_data
  
  
def mosaic_all_slices(processed_data, debug=False, 
                      filename='mosaic.png', savefig=False, num_rows = 5, num_cols = 5):
    # Reshape all slices into a single image collage
    # if np.max(processed_data) < 1.2:
    #     processed_data = processed_data * 2047
        
    num_slices = processed_data.shape[2]
    # num_cols = int(np.ceil(np.sqrt(num_slices)))
    # num_rows = int(np.ceil(num_slices / num_cols))
    # if num_cols ** 2 > num_slices:
    #     num_cols = 5
    #     num_rows = num_slices // num_cols
    
    
    # Create an empty array to hold the collage
    collage_height = num_rows * processed_data.shape[0]
    collage_width = num_cols * processed_data.shape[1]
    collage = np.zeros((collage_height, collage_width))
    
    # Fill the collage with slices
    for i in range(num_slices):
        row = i // num_cols
        col = i % num_cols
        slice_ = processed_data[:, :, i]
        collage[row * processed_data.shape[0]:(row + 1) * processed_data.shape[0],
                col * processed_data.shape[1]:(col + 1) * processed_data.shape[1]] = slice_
    
    collage =  np.rot90(collage) 

    if debug:
        plt.imshow(collage, cmap='gray')
        plt.axis('off')
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.show()
            
    return collage

def threshold_image(image, threshold):
    image[image < threshold] = 0
    return image

def mosaic_to_3D(collage_img, orig_dim1=64, orig_dim2=16, orig_dim3=64):
    # Initialize the 3D array
    im_3D = np.zeros((orig_dim3,  orig_dim1, orig_dim2))
    collage_img = np.rot90(collage_img, k=-1) # Undo by 270 degrees
     
    # Calculate the number of slices
    collage_img_shape = collage_img.shape

    # Extract each slice from the collage image
    slice_num = 0
    
    for row in range(collage_img_shape[0] // orig_dim1):
        for col in range(collage_img_shape[1] // orig_dim2):
            slice_ = collage_img[row * orig_dim1:(row + 1) * orig_dim1,
                                 col * orig_dim2:(col + 1) * orig_dim2]
            # plt.imshow(slice_, cmap='gray')
            # plt.show()
            im_3D[slice_num, :, :] = slice_
            slice_num += 1
    
    print(slice_num)
    return im_3D
        
# if __name__ == '__main__':
#     # Example usage
#     # File paths and initialization
#     input_file = '/Users/sairamgeethanath/Documents/Projects/Infectious_diseases/Work/Data/Selected_to_process/WIP/nhp_59228_D_22.nii.gz'
#     output_file_slices = '/Users/sairamgeethanath/Documents/Projects/Infectious_diseases/Work/Data/Selected_to_process/WIP/nhp_59228_D_22_preprocessed_slices.nii.gz'
#     output_file_collage = '/Users/sairamgeethanath/Documents/Projects/Infectious_diseases/Work/Data/Selected_to_process/WIP/nhp_59228_D_22_preprocessed_collage.nii.gz'
#     debug = True
    
#     # Load data and threshold
#     image_data = (nib.load(input_file).get_fdata()) / np.max(nib.load(input_file).get_fdata())
#     T = 0.2 *  (np.max(image_data))  
#     masked_image_data = np.copy(image_data)
#     masked_image_data[masked_image_data < T] = 0

#     # Change axis for through-plane input
#     masked_image_data = np.swapaxes(masked_image_data, 1, 2)
#     masked_image_data_shape = masked_image_data.shape
    
#     # Preprocess the data and write files to nifti
#     processed_data = preprocess_img_nhp(masked_image_data, debug=False)
#     input2zssr_img = mosaic_all_slices(processed_data, debug=True,filename='nhp_59228_D_22_mosaic_preprocess.png', savefig=True)
    
#     # Make nifti files
#     make_nifti(processed_data, fname = output_file_slices, mask=False, res=[2, 5, 2], dim_info=[0, 1, 2])
#     make_nifti(input2zssr_img, fname = output_file_collage, mask=False, res=[2, 5, 2], dim_info=[0, 1, 2])

#     if debug is True:
#         processed_data = mosaic_to_3D(input2zssr_img, orig_dim1=masked_image_data_shape[0], 
#                                       orig_dim2=masked_image_data_shape[1], orig_dim3=masked_image_data_shape[2])
#         # masked_image_data = np.swapaxes(processed_data, 1, 2)
#         mosaic_all_slices(processed_data, debug=True, filename='nhp_59228_D_22_mosaic_input.png', savefig=True)

def get_num_cols_rows(processed_coronal_img):
    num_slices = processed_coronal_img.shape[2]
    num_cols = int(np.ceil(np.sqrt(num_slices)))
    num_rows = int(np.ceil(num_slices / num_cols))
    if num_cols ** 2 > num_slices:
        num_cols = get_factors(num_slices)
        num_rows = num_slices // num_cols
    return num_cols, num_rows

def get_factors(num):
    factors = []
    for i in range(1,num+1):
        if num%i==0:
            factors.append(i)
    factor = factors[-1]
    return factor
