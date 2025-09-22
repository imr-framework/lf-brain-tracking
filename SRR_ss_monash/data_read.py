import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from typing import Dict, Tuple
from skimage.filters import threshold_otsu

class PairedMRI:
    def __init__(self, root_dir: str):
        """
        Args:
            root_dir: Path to training folder containing subject folders.
        """
        self.root_dir = root_dir
        self.subjects = sorted(os.listdir(root_dir))

    def _load_modality(self, path: str) -> nib.Nifti1Image:
        return nib.load(path)

    def _get_voxel_size(self, nii: nib.Nifti1Image) -> Tuple[float, float, float]:
        return tuple(np.round(nii.header.get_zooms(), 3))

    def _resample(self, data: np.ndarray, original_spacing: Tuple[float, float, float], target_spacing: Tuple[float, float, float]) -> np.ndarray:
        zoom_factors = [o/t for o, t in zip(original_spacing, target_spacing)]
        return zoom(data, zoom_factors, order=1)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        min_val, max_val = np.min(data), np.max(data)
        if max_val - min_val > 0:
            data = (data - min_val) / (max_val - min_val)
        return data

    def get_subject_data(self, subject_id: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns all modalities for given subject as dict:
        {
            "HF": {"t2w": np.ndarray, "flair": np.ndarray, "t1w": np.ndarray},
            "LF": {"t2w": np.ndarray, "flair": np.ndarray, "t1w": np.ndarray}
        }
        """
        subj_path = os.path.join(self.root_dir, subject_id)

        hf_path = os.path.join(subj_path, "3T")
        lf_path = os.path.join(subj_path, "64mT")

        hf_data, lf_data = {}, {}
        for seq in ["FLAIR", "T1", "T2"]:
            hf_img = self._load_modality(os.path.join(hf_path, f"{subject_id}_{seq}.nii.gz"))
            lf_img = self._load_modality(os.path.join(lf_path, f"{subject_id}_{seq}.nii.gz"))

            hf_data[seq] = hf_img
            lf_data[seq] = lf_img

        return {"HF": hf_data, "LF": lf_data}

    def get_voxel_sizes(self, subject_id: str) -> Dict[str, Dict[str, Tuple[float]]]:
        data = self.get_subject_data(subject_id)
        voxel_sizes = {
            "HF": {seq: self._get_voxel_size(img) for seq, img in data["HF"].items()},
            "LF": {seq: self._get_voxel_size(img) for seq, img in data["LF"].items()},
        }
        return voxel_sizes

    def get_resampled_normalized(self, subject_id: str, target_spacing=(1.0, 1.0, 2.0)) -> Dict[str, Dict[str, np.ndarray]]:
        data = self.get_subject_data(subject_id)
        processed = {"HF": {}, "LF": {}}
        for field in ["HF", "LF"]:
            for seq, img in data[field].items():
                voxel_size = self._get_voxel_size(img)
                arr = img.get_fdata()
                resampled = self._resample(arr, voxel_size, target_spacing)
                normed = self._normalize(resampled)
                processed[field][seq] = normed
        return processed


    def describe_subject(self, subject_id: str):
      """
      Prints details of all HF and LF modalities for the given subject:
      - Image shape
      - Voxel size
      - Min/Max values
      - Mean/Standard Deviation
      """
      data = self.get_subject_data(subject_id)

      print(f"\n--- Subject: {subject_id} ---")
      for field in ["HF", "LF"]:
          print(f"\n{field} MRI Data:")
          for seq, img in data[field].items():
              voxel_size = self._get_voxel_size(img)
              arr = img.get_fdata()
              min_val, max_val = np.min(arr), np.max(arr)
              mean_val, std_val = np.mean(arr), np.std(arr)
              print(f"  {seq}:")
              print(f"    Shape       : {arr.shape}")
              print(f"    Voxel size  : {voxel_size} mm")
              print(f"    Min/Max     : ({min_val:.3f}, {max_val:.3f})")
              print(f"    Mean/Std    : ({mean_val:.3f}, {std_val:.3f})")

    def analyze_noise_distribution(self, subject_id: str, hf_seq: str, lf_seq: str, mask=None, bins=200):
        """
        Compare pixel distributions of HF, LF, and residual (HF - LF) MRI for a given subject & sequence.
        """
        data = self.get_subject_data(subject_id)

        hf_img = data["HF"][hf_seq]
        lf_img = data["LF"][lf_seq]

        # Convert to arrays
        hf = hf_img.get_fdata() if hasattr(hf_img, "get_fdata") else np.array(hf_img)
        lf = lf_img.get_fdata() if hasattr(lf_img, "get_fdata") else np.array(lf_img)

        # Residual
        residual = hf - lf

        # Auto-generate mask if not provided
        if mask is None:
            thr = threshold_otsu(hf)
            mask = hf > thr

        fg_idx = mask
        bg_idx = ~mask

        def stats(arr):
            return {
                "mean": np.mean(arr),
                "std": np.std(arr),
                "min": np.min(arr),
                "max": np.max(arr),
                "SNR": np.mean(arr[fg_idx]) / np.std(arr[bg_idx])
            }

        hf_stats = stats(hf)
        lf_stats = stats(lf)
        res_stats = stats(residual)

        print("\n--- Intensity Statistics ---")
        for name, s in zip(["HF", "LF", "Residual"], [hf_stats, lf_stats, res_stats]):
            print(f"{name}: mean={s['mean']:.3f}, std={s['std']:.3f}, "
                  f"min={s['min']:.3f}, max={s['max']:.3f}, SNR={s['SNR']:.2f}")

        # Plot histograms
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        datasets = [hf, lf, residual]
        titles = ["HF MRI", "LF MRI", "Residual (HF - LF)"]

        for i, data_arr in enumerate(datasets):
            axs[0, i].hist(data_arr[fg_idx].flatten(), bins=bins, color='blue', alpha=0.7)
            axs[0, i].set_title(f"{titles[i]} - Foreground")
            axs[0, i].set_xlabel("Intensity")
            axs[0, i].set_ylabel("Count")

            axs[1, i].hist(data_arr[bg_idx].flatten(), bins=bins, color='red', alpha=0.7)
            axs[1, i].set_title(f"{titles[i]} - Background (Noise)")
            axs[1, i].set_xlabel("Intensity")
            axs[1, i].set_ylabel("Count")
            axs[1, i].set_yscale('log')


        plt.tight_layout()
        plt.show()


    def make_train_val_split(
    self,
    seq: str = "T2",   # "T1", "T2", "FLAIR", "all", or comma-separated like "T1,T2"
    train_size: int = 5,
    val_size: int = 5,
    target_spacing=(1.0, 1.0, 2.0),
    random_state: int = 42,
    mode: str = "multi"  # "multi" = stack channels, "append" = treat as separate samples
  ):
        """
        Creates train/validation splits for paired LF/HF MRI.

        Args:
            seq (str):
                - "T1", "T2", "FLAIR" -> single sequence
                - "all" -> all three [T1, T2, FLAIR]
                - "T1,T2", "T1,FLAIR", "T2,FLAIR" -> custom multi-sequence
            train_size (int): number of subjects for training
            val_size (int): number of subjects for validation
            target_spacing (tuple): voxel spacing to resample both HF and LF
            random_state (int): reproducibility
            mode (str):
                - "multi": stack sequences into channels (N, C, H, W, D)
                - "append": treat each sequence as independent (N*C, H, W, D)

        Returns:
            (x_train, y_train), (x_val, y_val)
        """
        assert train_size + val_size <= len(self.subjects), "Not enough subjects!"

        # Normalize seq argument
        if seq.lower() == "all":
            seqs = ["T1", "T2", "FLAIR"]
        else:
            seqs = [s.strip().upper() for s in seq.split(",")]

        print(f"Creating train/val split for {seqs} MRI in {mode} mode...")
        np.random.seed(random_state)
        indices = np.random.permutation(len(self.subjects))
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]

        def collect(indices):
            x, y = [], []
            for i in indices:
                subj_id = self.subjects[i]
                data = self.get_resampled_normalized(subj_id, target_spacing=target_spacing)

                if len(seqs) == 1:
                    # Single sequence
                    x.append(data["LF"][seqs[0]])
                    y.append(data["HF"][seqs[0]])
                else:
                    if mode == "multi":
                        # Stack as channels → (C, H, W, D)
                        lf_stack = np.stack([data["LF"][s] for s in seqs], axis=0)
                        hf_stack = np.stack([data["HF"][s] for s in seqs], axis=0)
                        x.append(lf_stack)
                        y.append(hf_stack)
                    elif mode == "append":
                        # Append as separate samples → increases N
                        for s in seqs:
                            x.append(data["LF"][s])
                            y.append(data["HF"][s])
                    else:
                        raise ValueError("mode must be 'multi' or 'append'")

            return np.array(x), np.array(y)

        x_train, y_train = collect(train_idx)
        x_val, y_val = collect(val_idx)

        return (x_train, y_train), (x_val, y_val)


    def get_self_supervised_data(
        self,
        seq: str = "all",
        train_size: int = 5,
        val_size: int = 5,
        target_spacing=(1.0, 1.0, 2.0),
        mode: str = "multi"
    ):
        """
        Returns only LF scans for self-supervised or zero-shot setups.
        Supports single, multi, or all sequences.
        mode:
            - "multi": stacked channels (N, C, H, W, D)
            - "append": independent samples (N*C, H, W, D)
        """
        (x_train, _), (x_val, _) = self.make_train_val_split(
            seq, train_size, val_size, target_spacing=target_spacing, mode=mode
        )
        return x_train, x_val

    def display_pair(self, subject_id: str, seq: str, slice_index: int = None, save_path: str = None):
        data = self.get_resampled_normalized(subject_id)
        hf_img = data["HF"][seq]
        lf_img = data["LF"][seq]

        if seq == "T1":
            s_seq = "T1w"
        elif seq == "T2":
            s_seq = "T2w"
        elif seq == "FLAIR":
            s_seq = "FLAIR"
        else:
            s_seq = seq

        if slice_index is None:
            slice_index = hf_img.shape[2] // 2

        # Flip the images horizontally (left-right)
        hf_img_flipped = np.rot90(hf_img[:, :, slice_index])
        lf_img_flipped = np.rot90(lf_img[:, :, slice_index])

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(lf_img_flipped, cmap="gray")
        axes[0].set_title(f"LF 0.064 T - {s_seq}")
        axes[0].axis("off")

        axes[1].imshow(hf_img_flipped, cmap="gray")
        axes[1].set_title(f"HF 3T - {s_seq}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{subject_id}_{seq}_slice{slice_index}.png"))
        plt.show()

        # if save_path:
        #     # Create directory if it doesn't exist
        #     os.makedirs(save_path, exist_ok=True)
        #     plt.savefig(os.path.join(save_path, f"{subject_id}_{seq}_slice{slice_index}.png"))
        #     print(f"Saved images to {save_path}")

if __name__ == "__main__":
        dataset = PairedMRI("Data/ULC_img enhancement/Training data")

        # List subjects
        print(dataset.subjects)

        # Get voxel sizes
        # voxel_info = dataset.get_voxel_sizes(dataset.subjects[0])
        # print(voxel_info)
        dataset.describe_subject(dataset.subjects[0])

        # dataset.display_pair(dataset.subjects[49], "T2", save_path= "Data/")

        # dataset.analyze_noise_distribution("POCEMR003", hf_seq="T1", lf_seq="T1")