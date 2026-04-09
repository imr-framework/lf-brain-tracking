# config.py
# from src_niv.models.ResUNet import residual_srr_unet
# from src_niv.models.DenseUNet import build_dense_unet_3d
# from src_niv.models.Inception import build_res_inception_unet_3d
import os

class Config:
    # -----------------------------
    # 🔧 DATA PARAMETERS
    # -----------------------------
    data_folder = "Data/data_sim_check/35528simulated_LF/train_test"
    subjects = ["26184", "30366", "35528", "34507", "35547", "59228", "59877", "59233"]
    train_day = 1
    val_day = 2
    test_days = [3, 4, 5]

    # -----------------------------
    # 🧠 MODEL PARAMETERS
    # -----------------------------
    model_name = "residual_srr_unet"
    # model_type = residual_srr_unet  # symbolic name; loaded dynamically if needed
    output_path = "src_simulated/outputs/Output_patch_noise_transformer"
    os.makedirs(output_path, exist_ok=True)
    
    # -----------------------------
    # ⚙️ TRAINING PARAMETERS
    # -----------------------------
    patch_xy = 64
    patch_z = 32
    input_shape = (64,64,32,1)
    batch_size = 32
    steps_per_epoch = 24
    epochs = 3
    learning_rate = 0.001
    loss_type_denoise = "mse_ssim_edge"

    # ----------------------------
    # 🔁 REFINEMENT PARAMETERS
    # -----------------------------
    retrain_loss_type = "mse_ssim_edge"
    retrain_batch_size = 2
    retrain_steps_per_epoch = 24
    retrain_epochs = 2
    visualize = False

    # -----------------------------
    # 📈 AUGMENTATION SETTINGS
    # -----------------------------
    train_augment = True
    val_augment = False
    augmentations_per_patch = 3
    angles = [0,10,15, 20, 25]
    jitter = 5
    intensity_aug = False
    noise_std = 0.01

    # -----------------------------
    # 🧩 PATCH SETTINGS
    # -----------------------------
    patches_per_volume = 16
    overlap = 0.5

    # Combine dynamically
    model_name = f"{model_name}_{loss_type_denoise}_{retrain_loss_type}"
    checkpoint_path = os.path.join(output_path, f"{model_name}_checkpoint.keras")
    refined_model_name = f"{model_name}_retrained"
    # -----------------------------
    # 🧾 LOGGING
    # -----------------------------
    csv_log_suffix = "_training_log.csv"

# Instantiate a global config
config = Config()
