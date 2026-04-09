# develop configuration file for CycleGAN model
# define parameters such as image size, batch size, learning rate, etc. 

# Check t1_t2 with the mae in cycle loss as well

import sys
sys.path.insert(0, './')
from src_simulated.cyclegan_models.losses_cyclegan import *
from functools import partial
import os

class CycleGANConfig:


    path_lf = "niv_raw_data/Nipah_IRF_data/data_niv/Low_field_data_DA/LFMRI_DATA_T2w"

    path_lf_t1w = "niv_raw_data/Nipah_IRF_data/data_niv/Low_field_data_DA/LFMRI_DATA_T1w"
    path_lf_t2w = "niv_raw_data/Nipah_IRF_data/data_niv/Low_field_data_DA/LFMRI_DATA_T2w"
    path_lf_all = "niv_raw_data/Nipah_IRF_data/data_niv/Low_field_data_DA/LFMRI_DATA_ALL"

    path_lf_test = "niv_raw_data/Nipah_IRF_data/data_niv/Low_field_data_DA/LFMRI_DATA_T1w_denoise"

    # Image parameters
    BATCH_SIZE = 1

    # code testing mode
    TEST = True  # Whether to run in test mode
    SLICES_TEST = False  # Whether to limit number of slices for quick testing of code

    # Descriminator parameters
    # Discriminator parameters
    DISC_LOSS = 'mse'
    DISC_LEARNING_RATE = 0.0002
    DISC_BETA_1 = 0.5
    DISC_LOSS_WEIGHTS = [0.25]  # weights for [real, fake]

    # Generator parameters

    GEN_LEARNING_RATE = 0.0002 
    GEN_BETA_1 = 0.5
    N_SLICES = 10

    # -----------------------
    # Generator Losses
    # -----------------------

    # Adversarial loss
    GEN_LOSS_1 = 'mse'
    
    # Identity loss
    GEN_LOSS_2 = 'mae' 
    # GEN_LOSS_2 = partial(mae_ssim_id, ssim_weight_id=0.5)

    # Cycle consistency losses
    # GEN_LOSS_3 = 'mae'
    # GEN_LOSS_4 = 'mae'
    GEN_LOSS_3 = partial(mae_ssim_cycle, ssim_weight_cycle=0.2) 
    GEN_LOSS_4 = partial(mae_ssim_cycle, ssim_weight_cycle=0.2)

    # Loss weights (CycleGAN paper style)
    GEN_LOSS_WEIGHTS = [0.25, 1, 30, 30] # weights for [adv, identity, cycle_A, cycle_B]

    #train parameters
    EPOCHS = 1000 # total number of epochs
    INITIAL_LR = 0.002 # initial learning rate
    N_ITER = 300 # number of epochs with initial learning rate
    N_ITER_DECAY = 0 # number of epochs with linearly decaying learning rate

    # Output directories
    OUTPUT_DIR = 'niv_results/outputs_src_simulated_context/cyclegan_lfmri20t2w_2_lfsimulated_context_1000_denoisedT2w' # directory to save outputs
    
    # Visualization parameters
    VISUALIZE = True  # Whether to visualize test examples during training

    #Model Evaluation loading
    MODEL_EVAL_PATH = 'niv_results/outputs_src_simulated/cyclegan_lfmri20t2w_1_lfsimulated/' # path to load model for evaluation

    # Specific model names to load using MODEL_EVAL_PATH
    MODEL_NAME_D_A = os.path.join(MODEL_EVAL_PATH, 'd_A_030000.keras')
    MODEL_NAME_D_B = os.path.join(MODEL_EVAL_PATH, 'd_B_030000.keras')
    MODEL_NAME_G_A2B = os.path.join(MODEL_EVAL_PATH, 'g_AtoB_030000.keras')
    MODEL_NAME_G_B2A = os.path.join(MODEL_EVAL_PATH, 'g_BtoA_030000.keras')

    # # With latest models
    # MODEL_NAME_D_A = os.path.join(MODEL_EVAL_PATH, 'd_A_latest.keras')
    # MODEL_NAME_D_B = os.path.join(MODEL_EVAL_PATH, 'd_B_latest.keras')
    # MODEL_NAME_G_A2B = os.path.join(MODEL_EVAL_PATH, 'g_A2B_latest.keras')
    # MODEL_NAME_G_B2A = os.path.join(MODEL_EVAL_PATH, 'g_B2A_latest.keras')

    # output_path_test
    OUTPUT_PATH_TEST = 'src_simulated/outputs/cyclegan_t1_t2_evaluate_outputs/'

config_lf = CycleGANConfig()