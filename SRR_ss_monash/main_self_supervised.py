import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from typing import Dict, Tuple
from SRR_ss_monash.data_read import PairedMRI

dataset = PairedMRI("Data/ULC_img enhancement/Training data")

# List subjects
print(dataset.subjects)











# # Multi-channel (default) → shape (N, 2, H, W, D)
# (x_train, y_train), (x_val, y_val) = dataset.make_train_val_split("T1", train_size=2, val_size=0, mode="multi")
# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)
# # Display some samples from x_train
# num_samples = min(2, x_train.shape[0])  # Show up to 3 samples
# slice_idx = x_train.shape[3] // 2  # Middle slice in W dimension

# for i in range(num_samples):
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     # Display the middle slice for sample i
#     ax.imshow(x_train[i, :, :, slice_idx], cmap='gray')
#     ax.set_title(f'Sample {i} - Slice {slice_idx}')
#     ax.axis('off')
#     plt.tight_layout()
#     plt.show()

