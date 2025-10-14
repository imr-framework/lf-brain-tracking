import sys
sys.path.insert(0, './')
import numpy as np
# -----------------------------
# VISUALIZATION
# -----------------------------
def visualize_random_slice(X, y, num_samples=3):
    for _ in range(num_samples):
        subj_idx = np.random.randint(0, X.shape[0])
        slice_idx = np.random.randint(0, X.shape[3])
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(X[subj_idx,:,:,slice_idx], cmap='gray')
        plt.title(f"Subject {subj_idx} - X slice {slice_idx}")
        plt.subplot(1,2,2)
        plt.imshow(y[subj_idx,:,:,slice_idx], cmap='gray')
        plt.title(f"Subject {subj_idx} - Y slice {slice_idx}")
        plt.show()

# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize_volume(vol, method='minmax'):
    if method=='minmax':
        vol_min, vol_max = vol.min(), vol.max()
        if vol_max - vol_min > 0:
            vol = (vol - vol_min) / (vol_max - vol_min)
        else:
            vol = np.zeros_like(vol)
    elif method=='zscore':
        mean, std = vol.mean(), vol.std()
        if std>0:
            vol = (vol - mean) / std
        else:
            vol = np.zeros_like(vol)
    return vol

def normalize_dataset(X, y, method='minmax'):
    X_norm = np.array([normalize_volume(vol, method) for vol in X])
    y_norm = np.array([normalize_volume(vol, method) for vol in y])
    return X_norm, y_norm