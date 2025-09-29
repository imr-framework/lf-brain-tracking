# T1 and T2 weighted images

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
                # voxel_size = self._get_voxel_size(normed)
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
    
    def resample_nifti(self, img, target_spacing=(1.6, 1.6, 1.0)):
        
        # Load image
        # img = nib.load(in_file)
        data = img.get_fdata()
        affine = img.affine
        header = img.header.copy()

        # Original voxel spacing
        original_spacing = header.get_zooms()[:3]

        # Compute zoom factors
        zoom_factors = np.array(original_spacing) / np.array(target_spacing)

        # Resample
        resampled_data = zoom(data, zoom_factors, order=3)  # cubic interpolation

        # Update affine
        new_affine = affine.copy()
        for i in range(3):
            new_affine[i, i] = target_spacing[i] * np.sign(affine[i, i])

        # Create new header with updated zooms
        new_header = header.copy()
        new_header.set_zooms(target_spacing)

        # Create new image
        new_img = nib.Nifti1Image(resampled_data, new_affine, header=new_header)

        # import nibabel.viewers

        # # Display the resampled image using OrthoSlicer3D
        # nibabel.viewers.OrthoSlicer3D(new_img.get_fdata()).show()

        return new_img

    def get_subject_image(self, subject_id: str, seq: str, modality: str = "HF", slice_index: int = None, cmap="gray", visible=True):
        """
        Retrieve an MRI volume from a subject for a given sequence.
        Optionally display a single slice.

        Args:
            subject_id (str): Subject identifier.
            seq (str): Sequence key (e.g., "T1", "T2", etc.).
            modality (str): "HF" or "LF" (default: "HF").
            slice_index (int): Slice index to visualize if visible=True (default: center slice).
            cmap (str): Colormap for visualization (default: "gray").
            visible (bool): If True, display the chosen slice.

        Returns:
            np.ndarray: The full 3D MRI volume as a NumPy array.
        """
        # Load subject data
        data = self.get_subject_data(subject_id)
        img = data[modality][seq]

        new_img = self.resample_nifti(img, target_spacing=(1.6, 1.6, 1.0))

        #Perform operations here
        # Normalize

        return new_img

    def compare_hf_lf_alignment(self, subject_id: str, hf_seq: str, lf_seq: str, slice_index: int = None, cmap_hf="gray", cmap_lf="hot", alpha=0.5):
        
        """
        Compare HF and LF MRI alignment by showing HF, LF, and Overlay slices side-by-side.

        Args:
            subject_id (str): Subject identifier.
            hf_seq (str): HF sequence key (e.g., "T1", "T2").
            lf_seq (str): LF sequence key (same sequence type as HF, e.g., "T1", "T2").
            slice_index (int): Slice index to display (default: middle slice).
            cmap_hf (str): Colormap for HF image (default: "gray").
            cmap_lf (str): Colormap for LF overlay (default: "hot").
            alpha (float): Transparency for LF overlay (0=transparent, 1=opaque).
            
        Returns:
            tuple: (hf_volume, lf_volume) as NumPy arrays.
        """
        # Load subject data
        data = self.get_subject_data(subject_id)
        hf_img = data["HF"][hf_seq]
        lf_img = data["LF"][lf_seq]

        # Convert to numpy
        hf_vol = hf_img.get_fdata() if hasattr(hf_img, "get_fdata") else np.array(hf_img)
        lf_vol = lf_img.get_fdata() if hasattr(lf_img, "get_fdata") else np.array(lf_img)

        # Pick slice (default = middle slice)
        if slice_index is None:
            slice_index = hf_vol.shape[2] // 2

        hf_slice = hf_vol[:, :, slice_index]
        lf_slice = lf_vol[:, :, slice_index]

        # Plot side-by-side
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # HF
        axs[0].imshow(hf_slice.T, cmap=cmap_hf, origin="lower")
        axs[0].set_title(f"HF - {hf_seq} (slice {slice_index})")
        axs[0].axis("off")

        # LF
        axs[1].imshow(lf_slice.T, cmap=cmap_hf, origin="lower")
        axs[1].set_title(f"LF - {lf_seq} (slice {slice_index})")
        axs[1].axis("off")

        # Overlay
        axs[2].imshow(hf_slice.T, cmap=cmap_hf, origin="lower")
        axs[2].imshow(lf_slice.T, cmap=cmap_lf, origin="lower", alpha=alpha)
        axs[2].set_title("Overlay")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

        return hf_vol, lf_vol


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

    def show_hf_lf_difference(self, subject_id: str, hf_seq: str, lf_seq: str, slice_index: int = None, cmap="bwr"):
        
        """
        Show the difference image (HF - LF) for a given subject and sequence.

        Args:
            subject_id (str): Subject identifier.
            hf_seq (str): HF sequence key (e.g., "T1", "T2").
            lf_seq (str): LF sequence key (same sequence type as HF).
            slice_index (int): Slice index to display (default: middle slice).
            cmap (str): Colormap for difference visualization (default: "bwr" -> blue=negative, red=positive).
            
        Returns:
            np.ndarray: Difference volume (HF - LF).
        """
        # Load subject data
        data = self.get_subject_data(subject_id)
        hf_img = data["HF"][hf_seq]
        lf_img = data["LF"][lf_seq]

        # Convert to numpy
        hf_vol = hf_img.get_fdata() if hasattr(hf_img, "get_fdata") else np.array(hf_img)
        lf_vol = lf_img.get_fdata() if hasattr(lf_img, "get_fdata") else np.array(lf_img)

        # Compute difference
        diff_vol = hf_vol - lf_vol

        # Pick slice
        if slice_index is None:
            slice_index = hf_vol.shape[2] // 2

        diff_slice = diff_vol[:, :, slice_index]

        # Plot
        plt.figure(figsize=(6, 6))
        plt.imshow(diff_slice.T, cmap="Reds", origin="lower")
        plt.colorbar(label="HF - LF Intensity")
        plt.title(f"Difference Image (slice {slice_index})")
        plt.axis("off")
        plt.show()

        return diff_vol

if __name__ == "__main__":
        
        # Improvement in difference image discard outside things
        
        dataset = PairedMRI("Data/ULC_img enhancement/Training data")

        # List subjects
        print(dataset.subjects)
        # Get voxel sizes
        # voxel_info = dataset.get_voxel_sizes(dataset.subjects[0])
        # print(voxel_info)
        dataset.describe_subject(dataset.subjects[0])
        # dataset.get_resampled_normalized((dataset.subjects[0]))
        dataset.display_pair(dataset.subjects[49], "T1", save_path= "Data/")
        # dataset.analyze_noise_distribution("POCEMR003", hf_seq="T1", lf_seq="T1")
        
        dataset.get_subject_image(dataset.subjects[0], "T1", 'LF')
        # dataset.compare_hf_lf_alignment(dataset.subjects[4], "T1", "T1")
        # dataset.show_hf_lf_difference(dataset.subjects[4], "T1", "T1")
        # # Multi-channel (default) → shape (N, 2, H, W, D)
        # (x_train, y_train), (x_val, y_val) = dataset.make_train_val_split("T1", train_size=1, val_size=1, mode="multi")
        # print(x_train.shape, y_train.shape)
        # print(x_val.shape, y_val.shape)

        # subject = dataset.get_subject_image(dataset.subjects[0], "T1", 'LF', visible=True)
        # subject = dataset.get_subject_image(dataset.subjects[0], "T1", 'HF', visible=True)

        # print(f"Shape: {subject.shape}")
        # print(f"Data type: {subject.dtype}")
        # print(f"Min: {np.min(subject)}, Max: {np.max(subject)}")
        # print(f"Mean: {np.mean(subject):.3f}, Std: {np.std(subject):.3f}")
        