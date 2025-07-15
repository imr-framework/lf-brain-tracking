import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import cv2


img0 = cv2.imread('./Code/ZSSR-master/test_data/charlie.png')
img = cv2.imread('./Code/ZSSR-master/test_data/brain_T2w_GT.png')
x, y, channels = img.shape
print(img.shape)
img_srr = cv2.imread('./Code/ZSSR-master/results/test_May_01_14_54_16/brain_T2w_zssr_X2.00X2.00.png')
print(img_srr.shape)

x_new = int(x * 0.5)
y_new = int(y * 0.5)
img2 = np.zeros((x_new, y_new,3))
print(img2.shape)

for ch in range(channels):
    img2[:, :, ch] = cv2.resize(np.squeeze(img[:, :, ch]), dsize=((x_new, y_new)))
    
cv2.imwrite('./Code/ZSSR-master/test_data/brain_T2w_gt.png', img)
cv2.imwrite('./Code/ZSSR-master/test_data/brain_T2w.png', img2)







