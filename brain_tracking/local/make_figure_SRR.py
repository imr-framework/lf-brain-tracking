from display_vlf_ni_data import plot_anatomy_raw, plot_anatomy_nifti


if __name__ == '__main__':
    plot_anatomy_nifti(im='axial.nii.gz', output_file='axial_ortho.png')
    plot_anatomy_nifti(im='sag.nii.gz', output_file='sag_ortho.png')
    plot_anatomy_nifti(im='cor.nii.gz', output_file='cor_ortho.png')
    # plot_anatomy_nifti(im='srr_1cycles.nii.gz', output_file='srr1_ortho.png')
    # plot_anatomy_nifti(im='srr_2cycles.nii.gz', output_file='srr2_ortho.png')
    # plot_anatomy_nifti(im='srr_3cycles.nii.gz', output_file='srr3_ortho.png')
    # plot_anatomy_nifti(im='srr_6cycles.nii.gz', output_file='srr6_ortho.png')
