import numpy as np

import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

def evaluate_model(model, x_data, y_data, batch_size=1, patch_size=None, full_slices=True,
                   save_path=None, dataset_name='val'):
    """
    Evaluate a trained 3D SRR model on validation or test data.

    Args:
        model: trained keras model
        x_data: LF volumes (N,X,Y,Z,1)
        y_data: HF volumes (N,X,Y,Z,1)
        batch_size: int
        patch_size: tuple or None
        full_slices: bool, whether to use full slices or patches
        save_path: str, path to save predicted volumes (optional)
        dataset_name: str, 'val' or 'test' for naming files

    Returns:
        metrics_dict: dictionary with PSNR and SSIM per volume and mean
        preds: list of predicted volumes
    """

    os.makedirs(save_path, exist_ok=True) if save_path else None
    N = x_data.shape[0]

    # Generator (no augmentation)
    gen = srr_generator(x_data, y_data,
                        batch_size=batch_size,
                        patch_size=patch_size,
                        full_slices=full_slices,
                        augment=False,
                        noise_sigma=0.0)

    psnr_list = []
    ssim_list = []
    preds = []

    print(f"\nEvaluating model on {dataset_name} set ({N} volumes)...")

    for i in range(N):
        x_batch, y_batch = next(gen)
        pred_batch = model.predict(x_batch)
        preds.append(pred_batch[0])  # Assuming batch_size=1

        # Compute PSNR/SSIM slice-wise along Z
        psnr_volume = []
        ssim_volume = []
        for z in range(y_batch.shape[1]):
            y_slice = y_batch[0, z, ..., 0]
            pred_slice = pred_batch[0, z, ..., 0]

            psnr_volume.append(sk_psnr(y_slice, pred_slice, data_range=y_slice.max()-y_slice.min()))
            ssim_volume.append(sk_ssim(y_slice, pred_slice, data_range=y_slice.max()-y_slice.min()))

        psnr_list.append(np.mean(psnr_volume))
        ssim_list.append(np.mean(ssim_volume))

        # Save predicted volume if path provided
        if save_path:
            save_file = os.path.join(save_path, f"{dataset_name}_pred_{i}.npy")
            np.save(save_file, pred_batch[0])
            print(f"Saved predicted volume {i} at {save_file}")

    metrics_dict = {
        'psnr_per_volume': psnr_list,
        'ssim_per_volume': ssim_list,
        'mean_psnr': np.mean(psnr_list),
        'mean_ssim': np.mean(ssim_list)
    }

    print(f"\nEvaluation complete. Mean PSNR: {metrics_dict['mean_psnr']:.2f}, "
          f"Mean SSIM: {metrics_dict['mean_ssim']:.4f}")

    return metrics_dict, preds


def srr_generator(lf_vols, hf_vols, batch_size=2, patch_size=None, full_slices=False,
                  augment=True, noise_sigma=0.02):
    """
    Flexible SRR generator (LF -> HF).

    Args:
        lf_vols: numpy array (N, X, Y, Z, 1)
        hf_vols: numpy array (N, X, Y, Z, 1)
        batch_size: int
        patch_size: tuple (px, py, pz) -> if given, split volume into non-overlapping patches
        full_slices: bool, if True yields full slices along Z instead of patches
        augment: bool, apply augmentations
        noise_sigma: float, Gaussian noise sigma (LF only)

    Yields:
        batch_lf: (B, X, Y, Z, 1)
        batch_hf: (B, X, Y, Z, 1)
    """

    N, X, Y, Z = lf_vols.shape

    while True:
        batch_lf, batch_hf = [], []

        # Shuffle volumes
        vol_indices = np.random.permutation(N)

        for idx in vol_indices:
            lf = lf_vols[idx]
            hf = hf_vols[idx]

            if full_slices:
                # Yield full 3D volume as one sample
                samples_lf, samples_hf = [lf], [hf]

            elif patch_size is not None:
                px, py, pz = patch_size
                # Split into non-overlapping patches
                samples_lf, samples_hf = [], []
                for x0 in range(0, X, px):
                    for y0 in range(0, Y, py):
                        for z0 in range(0, Z, pz):
                            lf_patch = lf[x0:x0+px, y0:y0+py, z0:z0+pz, :]
                            hf_patch = hf[x0:x0+px, y0:y0+py, z0:z0+pz, :]

                            if lf_patch.shape[:3] == (px, py, pz):  # ensure complete patch
                                samples_lf.append(lf_patch)
                                samples_hf.append(hf_patch)

            else:
                raise ValueError("Either patch_size must be given or full_slices=True")

            # Apply augmentations
            for lf_patch, hf_patch in zip(samples_lf, samples_hf):

                if augment:
                    # Random flips (3D)
                    if np.random.rand() > 0.5:
                        lf_patch = np.flip(lf_patch, axis=0)
                        hf_patch = np.flip(hf_patch, axis=0)
                    if np.random.rand() > 0.5:
                        lf_patch = np.flip(lf_patch, axis=1)
                        hf_patch = np.flip(hf_patch, axis=1)
                    if np.random.rand() > 0.5:
                        lf_patch = np.flip(lf_patch, axis=2)
                        hf_patch = np.flip(hf_patch, axis=2)

                    # LF-only Gaussian noise
                    if np.random.rand() > 0.5:
                        lf_patch = lf_patch + np.random.normal(0, noise_sigma, lf_patch.shape)

                    # LF-only intensity scaling
                    if np.random.rand() > 0.5:
                        factor = np.random.uniform(0.9, 1.1)
                        lf_patch = lf_patch * factor

                batch_lf.append(lf_patch)
                batch_hf.append(hf_patch)

                if len(batch_lf) == batch_size:
                    yield np.array(batch_lf), np.array(batch_hf)
                    batch_lf, batch_hf = [], []
