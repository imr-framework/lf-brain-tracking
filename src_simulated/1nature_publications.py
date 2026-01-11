# To do list for 1nature_publications.py
# Create and save dataset HF and LF in .npy format
# Identify of correct LF slices to start and compare with hf slices and save them
# Train using supervised and physics informed methods
# Implement evaluation metrics (PSNR, SSIM, VIF, etc.)
# Visualize results and create comparison plots
# Make different folder structures for different experiments

# Domain adaptation; Denoising; Super-resolution

# Denoising/Total variation
# ZSSR (inplane super-resolution)
# Diffusion based denoising
# Transformer based super-resolution
# Figure 2 update -- reference aware
# Loss terms (GRAM matrix and feature based losses)
# NIQE PIQE AES

# LF-MRI save actual in .npy format
# HF-MRI save actual (all)in .npy format


# Increase augmentation variability on contrast as compared to LF-MRI
# Use advanced augmentation techniques like elastic deformation, random cropping, rotation, scaling, intensity variations,
# and adding noise to simulate real-world variations in LF-MRI scans
# 
# Domain adaptation model  -- CYCLEGAN 3D and data prepration
# SRR architecture as hybrid architecture [CNN, Transformer]
# Diffusion denoising model for LF-MRI enhancement


# LF-MRI to simulated HF-MRI translation using CycleGAN
#1. Data Preparation: --- Prepare LFMRI by center correction
#2. Check domain adaptation at 2,2,5 mm resolution first, then denoising then SRR model.

# Final Inplementation details
# Data prepration
    1. Save all HF in the same folder with same resolution in nii.gz -- [DATA/NIPAH_IRF_DATA]
    2. Save all lf in the same folder with the same resolution in nii.gz -- [Data/Nipah_IRF_data/LFMRI_DATA_IRF_nifti_all]
    3. Save best visits of lf in the same folder in nii.gz  [Data/Nipah_IRF_data/LFMRI_DATA_IRF_best]
    4. Perform circshift to all the nii.gz files in the same folder
    4. Save t1 type-lf in the same folder
    5. Save t2 type-lf in the same fodler


Input (real LR): (2 × 2 × 5) mm

Output (HR): (1 × 1 × 2) mm --> 1x1x1 mm3

TRAINING PIPELINE (Simulation + Real Data)

    High-Frequency (HF) Reference Data
        HF GT volume
        Resolution: (1 × 1 × 2) mm   ← ground truth

    Physics-Based Degradation (Simulation)
    HF (1,1,2)
    ↓  B (slice PSF, anisotropic blur)
    ↓  D (downsampling)
    ↓  N (noise model)

    Simulated LR
    Resolution: (2 × 2 × 5) mm

# Stage 1 - Domain Adaptation Training Data
    
    Simulated LR (2,2,5)  ↔  Real LR (2,2,5)


    Purpose:

    Stage 0 — Input
    Real LR NIfTI
    Resolution: (2 × 2 × 5) mm

    Stage 1 — Domain Adaptation (2D / 2.5D)

    Real LR (2,2,5)
    ↓ 2.5D CycleGAN / CUT (axial slices)
    Domain-adapted LR
    Resolution: (2 × 2 × 5) mm  ← UNCHANGED


# Stage 2 — Physics-Informed Denoising (3D)

    Domain-adapted LR (2,2,5)
    ↓ 3D physics-informed denoiser
    Denoised LR
    Resolution: (2 × 2 × 5) mm  ← UNCHANGED

# Stage 3 — Physics-Informed Super-Resolution (3D)

    Denoised LR (2,2,5)
    ↓ 3D SRR (anisotropic upsampling)
    Super-resolved HR
    Resolution: (1 × 1 × 2) mm

# Stage 4 — Optional Consistency Refinement

    SR output (1,1,2)
    ↓ forward-projection check (optional)
    Final HR output
    Resolution: (1 × 1 × 2) mm

# SINGLE-LINE SUMMARY

    (2,2,5) ── DA ──▶ (2,2,5) ── DENOISE ──▶ (2,2,5) ── SR ──▶ (1,1,2)

# TRAINING VS INFERENCE (SIDE-BY-SIDE)

    Stage	Training Resolution	Inference Resolution
    Domain Adaptation	Sim LR ↔ Real LR (2,2,5)	Real LR (2,2,5)
    Denoising	Noisy LR → Clean LR (2,2,5)	DA LR → Clean LR (2,2,5)
    SRR	Clean LR → HF (1,1,2)	Clean LR → HR (1,1,2)

# FINAL TAKEAWAY

    Domain adaptation: (2,2,5) → (2,2,5)

    Denoising: (2,2,5) → (2,2,5)

    Super-resolution: (2,2,5) → (1,1,2)

#Ingredients to reach there

1. Correct all Low-field data to center allignment
2. Decide training and testing of Domain adaptation (Seperate testing samples or subjects from DA)
3. Design pipeline to get results on the four NHPs or LF T2-weighted imaging
4. Increase physics informed model (More data for variability of different contrast, including original imaging as well)