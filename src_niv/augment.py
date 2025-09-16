import numpy as np
import tensorflow as tf

# --------------------------
# 2D augmentations
# --------------------------

def rotate2d(img, angle_deg):
    angle_rad = angle_deg * np.pi / 180
    frac = angle_rad / (2.0 * np.pi)
    rr = tf.keras.layers.RandomRotation(factor=(frac, frac), fill_mode="nearest")
    img = tf.expand_dims(img, axis=0)
    out = rr(img)
    return out[0].numpy()

def shear2d(img, shear_x=0.1, shear_y=0.1):
    img = tf.expand_dims(img, axis=0)
    layer = tf.keras.layers.RandomTranslation(height_factor=shear_y, width_factor=shear_x, fill_mode="nearest")
    out = layer(img)
    return out[0].numpy()

def add_gaussian_noise(img, sigma=0.01):
    noise = np.random.normal(0, sigma, img.shape)
    return img + noise

def adjust_intensity(img, factor_range=(0.9, 1.1)):
    factor = np.random.uniform(*factor_range)
    return img * factor

# --------------------------
# 3D SRR patch generator
# --------------------------
def srr_generator(lf_vol, hf_vol, batch_size=2, patch_z=32, patch_xy=128, augment=True, extra_slices=0, noise_sigma=0.02):
    """
    Generate 3D SRR patches for LF-MRI -> HF-MRI training.
    Outputs:
        batch_lf: (batch_size, Z, X, Y)
        batch_hf: (batch_size, Z, X, Y)
    """

    # print("inside _ generator ......................")
    if lf_vol.ndim == 3:
        lf_vols = [lf_vol]
        hf_vols = [hf_vol]
    elif lf_vol.ndim == 4:
        lf_vols = [lf_vol[i] for i in range(lf_vol.shape[0])]
        hf_vols = [hf_vol[i] for i in range(hf_vol.shape[0])]
    else:
        raise ValueError("lf_vol must be (Z,X,Y) or (A,Z,X,Y)")

    N = len(lf_vols)
    while True:
        batch_lf, batch_hf = [], []

        volume_indices = np.random.permutation(N)
        for idx_vol in volume_indices:
            lf_current = lf_vols[idx_vol]
            hf_current = hf_vols[idx_vol]

            Z, X, Y = lf_current.shape

            # Random start along Z
            start = np.random.randint(0, max(1, Z - patch_z + 1))
            lf_patch = lf_current[start:start+patch_z].copy()
            hf_patch = hf_current[start:start+patch_z].copy()

            # Resize XY to patch_xy
            lf_patch = tf.image.resize(lf_patch[..., np.newaxis], (patch_xy, patch_xy)).numpy()[..., 0]
            hf_patch = tf.image.resize(hf_patch[..., np.newaxis], (patch_xy, patch_xy)).numpy()[..., 0]

            # Stack LF + HF for augmentation
            vol_stack = np.stack([lf_patch, hf_patch], axis=0)  # (2, Z, XY, XY)

            # Extra augmented slices
            if augment and extra_slices > 0:
                augmented_slices = []
                for _ in range(extra_slices):
                    idx = np.random.randint(0, vol_stack.shape[1])
                    slice_aug = vol_stack[:, idx].copy()  # (2, XY, XY)

                    lf_slice, hf_slice = slice_aug[0], slice_aug[1]

                    # -----------------------------
                    # 1. Global augmentations (apply to both LF & HF)
                    # -----------------------------
                    if np.random.rand() > 0.5:
                        angle = np.random.uniform(-15, 15)
                        lf_slice = rotate2d(lf_slice, angle)
                        hf_slice = rotate2d(hf_slice, angle)

                    if np.random.rand() > 0.5:
                        shear_x = np.random.uniform(-0.1, 0.1)
                        shear_y = np.random.uniform(-0.1, 0.1)
                        lf_slice = shear2d(lf_slice, shear_x, shear_y)
                        hf_slice = shear2d(hf_slice, shear_x, shear_y)

                    # -----------------------------
                    # 2. LF-only augmentations
                    # -----------------------------
                    if np.random.rand() > 0.6:
                        lf_slice = add_gaussian_noise(lf_slice, sigma=noise_sigma)

                    if np.random.rand() > 0.5:
                        lf_slice = adjust_intensity(lf_slice, factor_range=(0.85, 1.15))

                    # -----------------------------
                    # Put slices back
                    # -----------------------------
                    slice_aug[0], slice_aug[1] = lf_slice, hf_slice
                    augmented_slices.append(slice_aug[:, None, ...])  # add Z dim

                vol_stack = np.concatenate([vol_stack] + augmented_slices, axis=1)

            # Randomly select patch_z slices if needed
            if vol_stack.shape[1] > patch_z:
                idxs = np.random.choice(vol_stack.shape[1], patch_z, replace=False)
                vol_stack = vol_stack[:, idxs]

            batch_lf.append(vol_stack[0])
            batch_hf.append(vol_stack[1])

            # Inside your generator, replace the yield line with:
            if len(batch_lf) == batch_size:
                batch_lf_out = np.stack(batch_lf, axis=0)[..., np.newaxis]  # (B, Z, X, Y, 1)
                batch_hf_out = np.stack(batch_hf, axis=0)[..., np.newaxis]  # (B, Z, X, Y, 1)
                yield batch_lf_out, batch_hf_out
                batch_lf, batch_hf = [], []

def srr_generator_single(lf_vol, hf_vol, batch_size=2, patch_z=32, patch_xy=128, augment=True, extra_slices=0, noise_sigma=0.02):
    """
    Generate 3D SRR patches for LF-MRI -> HF-MRI training.
    Outputs:
        batch_lf: (batch_size, Z, X, Y)
        batch_hf: (batch_size, Z, X, Y)
    """

    print("inside _ generator ......................")
    if lf_vol.ndim == 3:
        lf_vols = [lf_vol]
        hf_vols = [hf_vol]
    elif lf_vol.ndim == 4:
        lf_vols = [lf_vol[i] for i in range(lf_vol.shape[0])]
        hf_vols = [hf_vol[i] for i in range(hf_vol.shape[0])]
    else:
        raise ValueError("lf_vol must be (Z,X,Y) or (A,Z,X,Y)")

    N = len(lf_vols)
    while True:
        batch_lf, batch_hf = [], []

        volume_indices = np.random.permutation(N)
        for idx_vol in volume_indices:
            lf_current = lf_vols[idx_vol]
            hf_current = hf_vols[idx_vol]

            Z, X, Y = lf_current.shape

            # Random start along Z
            start = np.random.randint(0, max(1, Z - patch_z + 1))
            lf_patch = lf_current[start:start+patch_z].copy()
            hf_patch = hf_current[start:start+patch_z].copy()

            # Resize XY to patch_xy
            lf_patch = tf.image.resize(lf_patch[..., np.newaxis], (patch_xy, patch_xy)).numpy()[..., 0]
            hf_patch = tf.image.resize(hf_patch[..., np.newaxis], (patch_xy, patch_xy)).numpy()[..., 0]

            # Stack LF + HF for augmentation
            vol_stack = np.stack([lf_patch, hf_patch], axis=0)  # (2, Z, XY, XY)

            # Extra augmented slices
            if augment and extra_slices > 0:
                augmented_slices = []
                for _ in range(extra_slices):
                    idx = np.random.randint(0, vol_stack.shape[1])
                    slice_aug = vol_stack[:, idx].copy()  # (2, XY, XY)

                    lf_slice, hf_slice = slice_aug[0], slice_aug[1]

                    # -----------------------------
                    # 1. Global augmentations (apply to both LF & HF)
                    # -----------------------------
                    if np.random.rand() > 0.5:
                        angle = np.random.uniform(-15, 15)
                        lf_slice = rotate2d(lf_slice, angle)
                        hf_slice = rotate2d(hf_slice, angle)

                    if np.random.rand() > 0.5:
                        shear_x = np.random.uniform(-0.1, 0.1)
                        shear_y = np.random.uniform(-0.1, 0.1)
                        lf_slice = shear2d(lf_slice, shear_x, shear_y)
                        hf_slice = shear2d(hf_slice, shear_x, shear_y)

                    # -----------------------------
                    # 2. LF-only augmentations
                    # -----------------------------
                    if np.random.rand() > 0.6:
                        lf_slice = add_gaussian_noise(lf_slice, sigma=noise_sigma)

                    if np.random.rand() > 0.5:
                        lf_slice = adjust_intensity(lf_slice, factor_range=(0.85, 1.15))

                    # -----------------------------
                    # Put slices back
                    # -----------------------------
                    slice_aug[0], slice_aug[1] = lf_slice, hf_slice
                    augmented_slices.append(slice_aug[:, None, ...])  # add Z dim

                vol_stack = np.concatenate([vol_stack] + augmented_slices, axis=1)

            # Randomly select patch_z slices if needed
            if vol_stack.shape[1] > patch_z:
                idxs = np.random.choice(vol_stack.shape[1], patch_z, replace=False)
                vol_stack = vol_stack[:, idxs]

            batch_lf.append(vol_stack[0])
            batch_hf.append(vol_stack[1])

            # Inside your generator, replace the yield line with:
            if len(batch_lf) == batch_size:
                batch_lf_out = np.stack(batch_lf, axis=0)[..., np.newaxis]  # (B, Z, X, Y, 1)
                batch_hf_out = np.stack(batch_hf, axis=0)[..., np.newaxis]  # (B, Z, X, Y, 1)
                yield batch_lf_out, batch_hf_out
                batch_lf, batch_hf = [], []



def srr_generator_dual(lf_vol_input, hf_vol_input, hf_vol_target,
                       batch_size=2, patch_z=32, patch_xy=128,
                       augment=True, extra_slices=0, noise_sigma=0.02):
    """
    Dual-encoder generator for LF+HF -> HF SRR network.

    Args:
        lf_vol_input: LF input volumes (train, 1st visit), shape (N,Z,X,Y) or (Z,X,Y)
        hf_vol_input: HF input volumes (train, 1st visit), shape (N,Z,X,Y) or (Z,X,Y)
        hf_vol_target: HF target volumes (train, 1st visit), shape (N,Z,X,Y) or (Z,X,Y)
        batch_size: number of patches per batch
        patch_z: number of slices per patch (depth)
        patch_xy: XY size of patch
        augment: whether to apply augmentation
        extra_slices: number of augmented extra slices
        noise_sigma: gaussian noise sigma for LF augmentation

    Yields:
        ([batch_lf, batch_hf_in], batch_hf_target)
            batch_lf:  (B, Z, X, Y, 1)
            batch_hf:  (B, Z, X, Y, 1)
            batch_hf_target: (B, Z, X, Y, 1)
    """

    # Handle single or multiple volumes
    def to_list(vol):
        if vol.ndim == 3:
            return [vol]
        elif vol.ndim == 4:
            return [vol[i] for i in range(vol.shape[0])]
        else:
            raise ValueError("Volume must be (Z,X,Y) or (N,Z,X,Y)")

    lf_vols = to_list(lf_vol_input)
    hf_vols_in = to_list(hf_vol_input)
    hf_vols_tgt = to_list(hf_vol_target)

    N = len(lf_vols)
    while True:
        batch_lf, batch_hf_in, batch_hf_tgt = [], [], []

        volume_indices = np.random.permutation(N)
        for idx_vol in volume_indices:
            lf_current = lf_vols[idx_vol]
            hf_current_in = hf_vols_in[idx_vol]
            hf_current_tgt = hf_vols_tgt[idx_vol]

            Z, X, Y = lf_current.shape

            # Random start along Z
            start = np.random.randint(0, max(1, Z - patch_z + 1))
            lf_patch = lf_current[start:start+patch_z].copy()
            hf_patch_in = hf_current_in[start:start+patch_z].copy()
            hf_patch_tgt = hf_current_tgt[start:start+patch_z].copy()

            # Resize XY to patch_xy
            lf_patch = tf.image.resize(lf_patch[..., np.newaxis], (patch_xy, patch_xy)).numpy()[..., 0]
            hf_patch_in = tf.image.resize(hf_patch_in[..., np.newaxis], (patch_xy, patch_xy)).numpy()[..., 0]
            hf_patch_tgt = tf.image.resize(hf_patch_tgt[..., np.newaxis], (patch_xy, patch_xy)).numpy()[..., 0]

            # Stack LF + HF-in + HF-target for augmentation
            vol_stack = np.stack([lf_patch, hf_patch_in, hf_patch_tgt], axis=0)  # (3, Z, XY, XY)

            # Augment extra slices
            if augment and extra_slices > 0:
                augmented_slices = []
                for _ in range(extra_slices):
                    idx = np.random.randint(0, vol_stack.shape[1])
                    slice_aug = vol_stack[:, idx].copy()  # (3, XY, XY)

                    lf_slice, hf_in_slice, hf_tgt_slice = slice_aug

                    # --- Global augmentations (same for LF, HF-in, HF-target) ---
                    if np.random.rand() > 0.5:
                        angle = np.random.uniform(-15, 15)
                        lf_slice = rotate2d(lf_slice, angle)
                        hf_in_slice = rotate2d(hf_in_slice, angle)
                        hf_tgt_slice = rotate2d(hf_tgt_slice, angle)

                    if np.random.rand() > 0.5:
                        shear_x = np.random.uniform(-0.1, 0.1)
                        shear_y = np.random.uniform(-0.1, 0.1)
                        lf_slice = shear2d(lf_slice, shear_x, shear_y)
                        hf_in_slice = shear2d(hf_in_slice, shear_x, shear_y)
                        hf_tgt_slice = shear2d(hf_tgt_slice, shear_x, shear_y)

                    # --- LF-only augmentations ---
                    if np.random.rand() > 0.6:
                        lf_slice = add_gaussian_noise(lf_slice, sigma=noise_sigma)

                    if np.random.rand() > 0.5:
                        lf_slice = adjust_intensity(lf_slice, factor_range=(0.85, 1.15))

                    slice_aug = np.stack([lf_slice, hf_in_slice, hf_tgt_slice], axis=0)
                    augmented_slices.append(slice_aug[:, None, ...])  # add Z dim

                vol_stack = np.concatenate([vol_stack] + augmented_slices, axis=1)

            # Ensure patch_z slices
            if vol_stack.shape[1] > patch_z:
                idxs = np.random.choice(vol_stack.shape[1], patch_z, replace=False)
                vol_stack = vol_stack[:, idxs]

            # Collect patches
            batch_lf.append(vol_stack[0])       # LF input
            batch_hf_in.append(vol_stack[1])    # HF input
            batch_hf_tgt.append(vol_stack[2])   # HF target

            # Yield batch
            if len(batch_lf) == batch_size:
                batch_lf_out = np.stack(batch_lf, axis=0)[..., np.newaxis]        # (B, Z, X, Y, 1)
                batch_hf_in_out = np.stack(batch_hf_in, axis=0)[..., np.newaxis]  # (B, Z, X, Y, 1)
                batch_hf_tgt_out = np.stack(batch_hf_tgt, axis=0)[..., np.newaxis]# (B, Z, X, Y, 1)

                yield [batch_lf_out, batch_hf_in_out], batch_hf_tgt_out
                batch_lf, batch_hf_in, batch_hf_tgt = [], [], []

# import numpy as np
# import tensorflow as tf

# # Rotate 2D slice
# def rotate2d(img, angle_rad):
#     frac = angle_rad / (2.0 * np.pi)
#     rr = tf.keras.layers.RandomRotation(factor=(frac, frac), fill_mode="nearest")
#     img = tf.expand_dims(img, axis=0)  # add batch
#     out = rr(img)                      # (1,H,W,C)
#     return out[0]                      # (H,W,C)

# # Shear 2D slice using ImageProjectiveTransformV3 (no TFA)
# def shear2d(img, shear_x, shear_y):
#     X, Y, C = img.shape
#     # Shear matrix
#     transform = [1.0, shear_x, 0.0,
#                  shear_y, 1.0, 0.0,
#                  0.0,     0.0]
#     transform = tf.convert_to_tensor([transform], dtype=tf.float32)

#     # ImageProjectiveTransformV3 expects (N,H,W,C)
#     img = tf.expand_dims(img, axis=0)
#     out = tf.raw_ops.ImageProjectiveTransformV3(
#         images=img,
#         transforms=transform,
#         output_shape=[X, Y],
#         interpolation="BILINEAR"
#     )
#     return out[0]  # (H,W,C)

# def rotate2d(img, angle_deg):
#     """Rotate a single 2D slice."""
#     angle_rad = angle_deg * np.pi / 180
#     frac = angle_rad / (2.0 * np.pi)
#     rr = tf.keras.layers.RandomRotation(factor=(frac, frac), fill_mode="nearest")
#     img = tf.expand_dims(img, axis=0)
#     out = rr(img)
#     return out[0]

# def shear2d(img, shear_x=0.1, shear_y=0.1):
#     """Approximate shear with RandomTranslation."""
#     img = tf.expand_dims(img, axis=0)
#     layer = tf.keras.layers.RandomTranslation(height_factor=shear_y, width_factor=shear_x, fill_mode="nearest")
#     out = layer(img)
#     return out[0]

# def srr_generator(lf_vol, hf_vol, batch_size=1, patch_z=32, augment=True, num_augmented_copies=1):
    
#     """
#     Generator for 3D SRR patches: original patch first, then augmented copies as separate batches.
    
#     Parameters
#     ----------
#     lf_vol, hf_vol : np.ndarray
#         Single volume (Z,X,Y) or multiple volumes (A,Z,X,Y)
#     batch_size : int
#         Number of patches per batch
#     patch_z : int
#         Number of slices per patch
#     augment : bool
#         Whether to apply augmentation
#     num_augmented_copies : int
#         Number of augmented copies per slice
#     """
#     import tensorflow as tf
#     import numpy as np

#     # Helper: translate slice using tf
#     def translate_slice(slice_img, tx, ty):
#         slice_tf = tf.keras.preprocessing.image.apply_affine_transform(
#             slice_img,
#             tx=tx,
#             ty=ty,
#             row_axis=0,
#             col_axis=1,
#             channel_axis=2,
#             fill_mode='nearest'
#         )
#         return slice_tf

#     # Handle single vs multiple volumes
#     if lf_vol.ndim == 3:
#         lf_vols = [lf_vol]
#         hf_vols = [hf_vol]
#     elif lf_vol.ndim == 4:
#         lf_vols = [lf_vol[i] for i in range(lf_vol.shape[0])]
#         hf_vols = [hf_vol[i] for i in range(hf_vol.shape[0])]
#     else:
#         raise ValueError("lf_vol must be (Z,X,Y) or (A,Z,X,Y)")

#     N = len(lf_vols)

#     while True:
#         volume_indices = np.random.permutation(N)

#         for idx_vol in volume_indices:
#             lf_current = lf_vols[idx_vol]
#             hf_current = hf_vols[idx_vol]
#             Z, X, Y = lf_current.shape

#             # Random start along Z
#             start = np.random.randint(0, max(1, Z - patch_z + 1))
#             lf_patch = lf_current[start:start+patch_z].copy()
#             hf_patch = hf_current[start:start+patch_z].copy()

#             vol_stack = np.stack([lf_patch, hf_patch], axis=-1)  # (patch_z, X, Y, 2)

#             # ----- Yield original patch first -----
#             lf_patch_out = vol_stack[..., 0][..., np.newaxis]
#             hf_patch_out = vol_stack[..., 1][..., np.newaxis]
#             yield np.expand_dims(lf_patch_out, 0), np.expand_dims(hf_patch_out, 0)

#             # ----- Generate augmented copies as separate batches -----
#             if augment and num_augmented_copies > 0:
#                 for _ in range(num_augmented_copies):
#                     augmented_slices = []
#                     for slice_idx in range(vol_stack.shape[0]):
#                         slice_aug = vol_stack[slice_idx].copy()

#                         # rotations
#                         if np.random.rand() > 0.5:
#                             slice_aug = rotate2d(slice_aug, np.random.uniform(-15, 15)).numpy()
#                         # shear
#                         if np.random.rand() > 0.5:
#                             sx = np.random.uniform(-0.2, 0.2)
#                             sy = np.random.uniform(-0.2, 0.2)
#                             slice_aug = shear2d(slice_aug, sx, sy).numpy()
#                         # crop + resize
#                         if np.random.rand() > 0.5:
#                             crop_frac = np.random.uniform(0.7, 0.95)
#                             cx, cy = int(X*crop_frac), int(Y*crop_frac)
#                             x0, y0 = (X - cx)//2, (Y - cy)//2
#                             cropped = slice_aug[x0:x0+cx, y0:y0+cy, :]
#                             slice_aug = tf.image.resize(cropped, (X, Y)).numpy()
#                         # shifting
#                         if np.random.rand() > 0.5:
#                             tx = np.random.uniform(-0.1, 0.1) * X
#                             ty = np.random.uniform(-0.1, 0.1) * Y
#                             slice_aug = translate_slice(slice_aug, tx, ty)
#                         # zooming
#                         if np.random.rand() > 0.5:
#                             zoom_factor = np.random.uniform(0.8, 1.2)
#                             new_x, new_y = int(X*zoom_factor), int(Y*zoom_factor)
#                             slice_resized = tf.image.resize(slice_aug, (new_x, new_y))
#                             if zoom_factor > 1.0:
#                                 x0, y0 = (new_x - X)//2, (new_y - Y)//2
#                                 slice_aug = slice_resized[x0:x0+X, y0:y0+Y, :]
#                             else:
#                                 pad_x = (X - new_x)//2
#                                 pad_y = (Y - new_y)//2
#                                 slice_aug = tf.image.pad_to_bounding_box(slice_resized, pad_x, pad_y, X, Y)
#                             slice_aug = slice_aug.numpy()

#                         augmented_slices.append(slice_aug[None, ...])

#                     aug_patch = np.concatenate(augmented_slices, axis=0)
#                     lf_aug_out = aug_patch[..., 0][..., np.newaxis]
#                     hf_aug_out = aug_patch[..., 1][..., np.newaxis]

#                     # Yield augmented patch as separate batch
#                     yield np.expand_dims(lf_aug_out, 0), np.expand_dims(hf_aug_out, 0)


# import numpy as np

# def systematic_crops_3d(volume, target_shape=(32, 64, 64), stride=(32, 64, 64)):
#     """
#     Systematically crop a 4D volume (Z, X, Y, C) into non-overlapping or overlapping patches.
#     target_shape: patch size (Z, X, Y)
#     stride: step size for cropping (Z, X, Y)
#     """
#     z, x, y, c = volume.shape
#     tz, tx, ty = target_shape
#     sz, sx, sy = stride
    
#     assert z == tz, f"Z mismatch: got {z}, expected {tz}"

#     patches = []
#     for i in range(0, x - tx + 1, sx):
#         for j in range(0, y - ty + 1, sy):
#             patch = volume[:, i:i+tx, j:j+ty, :]
#             patches.append(patch)
#     return patches


# def systematic_crop_generator(base_gen, target_shape=(32,64,64), stride=(32,64,64)):
#     """
#     Wraps a base generator (like srr_generator) and produces systematic crops.
#     """
#     while True:
#         X, Y = next(base_gen)  # shapes: (B, Z, X, Y, C)
#         X_crops, Y_crops = [], []
        
#         for i in range(X.shape[0]):  # loop over batch
#             X_patches = systematic_crops_3d(X[i], target_shape, stride)
#             Y_patches = systematic_crops_3d(Y[i], target_shape, stride)
#             X_crops.extend(X_patches)
#             Y_crops.extend(Y_patches)
        
#         yield np.array(X_crops), np.array(Y_crops)