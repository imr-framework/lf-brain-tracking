# Train 35, valid 5, test 10
# Free surfer 10 subjects

import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from typing import Dict, Tuple
from SRR_SS_Mon.data_read import PairedMRI
from SRR_SS_Mon.supervised_train import train_and_evaluate
from src_niv.models.ResUNet import residual_srr_unet

# model_type = 
# model_ = 
steps_per_epoch = 50
epochs = 50
batch_size = 2
visualize_pairs = False
# calling the residual_srr_unet model

model_type = 'residual_srr_unet_subjects'
model_ = residual_srr_unet

dataset = PairedMRI("Data/ULC_img enhancement/Training data")

# List subjects
print(dataset.subjects)

# Multi-channel (default) → shape (N, 2, H, W, D)
(x_train, y_train), (x_val, y_val) = dataset.make_train_val_split("T1", train_size=1, val_size=1, mode="multi")
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

train_and_evaluate(x_train, y_train, x_val, y_val, model_type, model_, steps_per_epoch = 50,
     epochs = 50, batch_size = 1, visualize_pairs = False)
    



