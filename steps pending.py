# Physics informed noise simulation step for denoising; SRR and Domain adoption for image enhancement and correction;
# Noise mask and discard outside noise (all steps at 1,1,2)
# Noise mask and correct all lF-MRI data with parameters retrieve for noise mask.
# Apply noise to reduce noise from images;
# Then apply domain adoption using CycleGAN to correct the images to be more like the target domain (e.g. 3T MRI).
# Finally, evaluate the performance of the corrected images using metrics such as PSNR and SSIM
# Finally Image enhancement method of SRR to train the images;

# Improvements for the models;
# Evaluation pipeline and volume save results of nifti and test with denoising and enhancement;

1. Domain adoption model is okay with CycleGAN, but can be improved with Context.

2. Image denoising model is okay with very lightweight architecture; however, it can be enhanced with more advanced techniques such as attention mechanisms or transformer-based models.

3. Image Enhancement model enhanced with more UNet variants adding context and multiscale context.

# Steps for today:
1. Train test-split problem resolve; based on Domain adoption; (7 subjects and 3 subjects)

2. Evalyation pipeline and volume save results of nifti and test with denoising and enhancement;
    
3. CycleGAN with registration for domain adoption;

4. Prepration of all subjects in low-field fortraining and testing; (7 for training and 3 for testing)
