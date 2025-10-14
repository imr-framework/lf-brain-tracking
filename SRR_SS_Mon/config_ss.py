



class Config:
    # Model setup
    model_type = 'residual_srr_unet_subjects'
    model_fn = residual_srr_unet

    # Training setup
    steps_per_epoch = 50
    epochs = 50
    batch_size = 2

    # Data setup
    patch_size = None
    full_slices = True
    augment = False
    noise_sigma = 0.02
    visualize_pairs = False
    day_idx = 0

    # Loss/metrics
    composite_loss = 'mse'
    psnr = None
    ssim = None

    # Output
    results_dir = './Data/ULC_results'
