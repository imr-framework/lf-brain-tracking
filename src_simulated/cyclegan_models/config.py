# develop configuration file for CycleGAN model
# define parameters such as image size, batch size, learning rate, etc. 
import sys
sys.path.insert(0, './')
from src_simulated.cyclegan_models.losses_cyclegan import *
from functools import partial
import os

class CycleGANConfig:

    # Image parameters
    BATCH_SIZE = 1

    # code testing mode
    TEST = False  # Whether to run in test mode
    SLICES_TEST = False  # Whether to limit number of slices for quick testing of code

    #Descriminator parameters
    # Discriminator parameters
    DISC_LOSS = 'mse'
    DISC_LEARNING_RATE = 0.0001
    DISC_BETA_1 = 0.5
    DISC_LOSS_WEIGHTS = [0.20]  # weights for [real, fake]

    # Generator parameters

    GEN_LEARNING_RATE = 0.0002 
    GEN_BETA_1 = 0.5

    # -----------------------
    # Generator Losses
    # -----------------------

    # Adversarial loss
    GEN_LOSS_1 = 'mse'
    
    # Identity loss
    GEN_LOSS_2 = 'mae' 
    # GEN_LOSS_2 = partial(mae_ssim_id, ssim_weight_id=0.5)

    # Cycle consistency losses
    GEN_LOSS_3 = partial(mae_ssim_cycle, ssim_weight_cycle=0.1) 
    GEN_LOSS_4 = partial(mae_ssim_cycle, ssim_weight_cycle=0.1) 

    # Loss weights (CycleGAN paper style)
    GEN_LOSS_WEIGHTS = [0.10, 1, 15, 15] # weights for [adv, identity, cycle_A, cycle_B]

    #train parameters
    EPOCHS = 150 # total number of epochs
    INITIAL_LR = 0.0002 # initial learning rate
    N_ITER = 80 # number of epochs with initial learning rate
    N_ITER_DECAY = 100 # number of epochs with linearly decaying learning rate

    # Output directories
    OUTPUT_DIR = 'src_simulated/outputs/cyclegan_t1_t2_upsample14' # directory to save outputs

    # Visualization parameters
    VISUALIZE = False  # Whether to visualize test examples during training

    #Model Evaluation loading
    MODEL_EVAL_PATH = 'src_simulated/outputs/cyclegan_t1_t2_upsample13/' # path to load model for evaluation

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

config = CycleGANConfig()