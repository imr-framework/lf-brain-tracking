import numpy as np
import tensorflow as tf
from model import DiffusionUNet
from diffusion import beta, alpha
from dataset import normalize
import nibabel as nib

H, W = 256, 256

model = DiffusionUNet((H, W, 3))
model.load_weights("checkpoints/ddpm.h5")

lf = nib.load("data/LF/test.nii.gz").get_fdata()
lf = normalize(lf)

x = np.random.normal(size=lf.shape)

for t in reversed(range(len(beta))):
    eps = model.predict([x, lf], verbose=0)
    x = (x - beta[t] * eps) / np.sqrt(alpha[t])

nib.save(nib.Nifti1Image(x[...,1], np.eye(4)), "outputs/pred_hf.nii.gz")
nib.save(nib.Nifti1Image(lf[...,1], np.eye(4)), "outputs/input_lf.nii.gz")