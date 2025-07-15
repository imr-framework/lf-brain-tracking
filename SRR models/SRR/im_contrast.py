





def do_contrast_enhance(im):
    im_ce = np.zeros_like(im)
    for slice in range(im.shape[2]):
        im_slice = 255 * do_norm_im(np.squeeze(im[:, :, slice]))
        im_slice = im_slice.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        im_cl = clahe.apply(im_slice)
        im_ce[:, :, slice] = im_cl

    return im_ce