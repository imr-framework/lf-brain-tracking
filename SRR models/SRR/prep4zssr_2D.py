import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import cv2


img = cv2.imread('./Code/ZSSR-master/test_data/Brain_T2_GT.jpeg')
x, y, channels = img.shape
print(img.shape)

x_new = int(x * 0.5)
y_new = int(y * 0.5)
img2 = np.zeros((x_new, y_new,3))
print(img2.shape)

for ch in range(channels):
    img2[:, :, ch] =  cv2.resize((img[:, :, ch]), dsize=((y_new, x_new)))

    
# cv2.imwrite('./Code/ZSSR-master/test_data/brain_T2w_gt.png', img)
cv2.imwrite('./Code/ZSSR-master/test_data/brain_T2w.png', img2)







