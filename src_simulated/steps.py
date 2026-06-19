# Steps for Improvement and training of the model;

    # 1. Improvement in the CycleGAn for hallucination and domain.
    # 2. Training of ResUNet for image quality enhancement for the task.

    # Go to CycleGAN models and use file - src_simulated/cyclegan_models/cyclegan2d_lfmri_downsampledhf112_params_parallel.py for task training and improvement.
    # Test using the files inside the test scripts for the above task and check for improvements in the hallucination and domain adaptation task.
    # Create synthetic data for the above models and retrain file DA for domain adaptation to the HF to the domain adopted network; then generate data and check for improvements;
    # Check for the DA improvements using the test scripts for the above task.
    # Retrain the ResUNet for the image quality enhancement task using the models trained on the synthetic data task and check for improvements in the image quality enhancement task.
    # Test using the file inside the test scripts;