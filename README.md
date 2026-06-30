![LF Brain Tracking](assets/logo2.jpeg)

# Need for accessible imaging

Changes in brain morphology provide critical insights into a wide range of neuropsychiatric disorders. Magnetic Resonance Imaging (MRI) has been the primary tool to investigate these disorders, highlighting structural, functional, and metabolic changes. For example, 3T structural MRI data have significantly improved our understanding of the youths’ developing brain in health and disease. Recent studies have highlighted the need for densely sampled temporal neuroimaging data to maximize clinical insight in patients with mental health challenges. However, according to the World Health Organization, two-thirds of the global population does not have access to MRI. The cost, power, local expertise, and siting requirements of high-field MR systems (1.5T) impede longitudinal imaging in large populations, especially those in low-resource settings. The recent resurgence of portable, very low-field MRI (<0.1T) has provided an alternative to obtaining imaging data in an accessible and scalable manner. Currently, the major obstacle is that these portable scanners suffer from lower signal-to-noise ratio (SNR), impacting the volumetric accuracy required to monitor brain changes using structural imaging. These limitations render these scanners supplementary devices to high-field systems. For meaningful use, there is a critical need to develop novel methods to produce very low-field, structural MRI data statistically equivalent to or better than 3T data.

<br>
<h1 align="center">Low-field brain tracking using 0.05T MRI</h3>
<br>
Our image currently looks like ....

![LF Brain pipeline](assets/lf-mri-pipeline.jpeg)

# Development of lf-brain-tracking tools.
<br>

![Status](https://img.shields.io/badge/status-under%20development-orange)

This repository is currently under active development. Some modules are subject to frequent updates.

## 🚧 Description
This is the development branch for the low-field-brain tracking project. It integrates development or investigation of advanced lf-mri tools for motion correction, lf-simulations, zero-shot denoising, super-resolution reconstruction, and brain segmentation to enable robust brain tracking under low-SNR and low-resolution conditions.



# Registration and Orthogonal acquisiton combination

NiftyMIC is a research-focused toolkit for motion correction and volumetric image reconstruction. NiftyMIC tool is used for motion correction and three-orthogonal acquisition combination [NiftyMIC](https://github.com/gift-surg/niftymic)

# Brain Parcellation

Brain parcellation tools are publicly available [SynthSeg](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg), [SuperSynth](https://surfer.nmr.mgh.harvard.edu/fswiki/SuperSynth).

    SuperSynth:

# References 

# Aim1: Track youth brain changes at 0.05T using densely sampled neuroimaging data and DL-SRR

<h3 align="center">Conference proceddings</h3>
<br>

1. Girish, N., Sharma, A., and Geethanath, S., "Zero-shot self-supervised super-resolution reconstruction of MRI to track brain changes using volumetry: application to high and low-field data," SPIE Medical Imaging, 2026.
2. Sharma, A., & Geethanath, S. (2026). Learning beyond interpolation: Zero-shot resolution enhancement for low-field MRI. International Society for Magnetic Resonance in Medicine (ISMRM 2026) Annual Meeting.
3. Sharma, A., Oiye, I. E., Byrum, R., Holbrook, M., Cong, Y., Calcagno, C., Mani, V., & Geethanath, S. (2026). Enhancing low-field MRI image quality for Nipah virus infection imaging using deep learning. International Society for Magnetic Resonance in Medicine (ISMRM) Annual Meeting. 
4. Sharma, A., Oiye, I. E., Byrum, R., Holbrook, M., Cong, Y., Calcagno, C., Mani, V., & Geethanath, S. (2025). Physics-informed low-field Nipah virus MRI image reconstruction of non-human primates in a BSL-4 facility. International Society for Magnetic Resonance in Medicine (ISMRM) Annual Meeting.

<h3 align="center">Preprints</h3>
<br>

1. Oiye, I. E., Sharma, A., Mohanta, Z., Sankaralayam, D. S., Uchida, Y., Akinwale, T., ... & Geethanath, S. (2025). A Hands-On Workshop for Constructing a Low-Field MRI System in Three Days. arXiv preprint arXiv:2511.20979.

<h3 align="center">Revised submission</h3>
<br>

1. Sharma, A., Lu, H., Greenspan, H., Lin, D. D., & Geethanath, S. (2025). Enhancing low-field magnetic resonance image quality using deep learning: Challenges, opportunities, and resources. Manuscript under revision in NMR in Biomedicine.

# Aim 2: Automate 0.05T MRI (auto-MRI) to deliver consistent scanner operation and image quality for robust deployment

Open-source repository for Digital Twin [Auto-MRI](https://github.com/imr-framework/virtual-scanner-adt)

<h3 align="center">Conference proceddings</h3>
<br>

1. Kinyera.L, Oiye, I. E., Geethanath.A, Obungoloch. J, & Geethanath, S. (2026).  An Autonomous Digital Twin Agent for the Parametric Design and Validation of Halbach Array Magnets for Low-Field MRI. International Society for Magnetic Resonance in Medicine (ISMRM) Annual Meeting.

# Readings


# Other open-source tools for MRI education.

DELTA DIY MRI: Learning through building and playing [DELTA DIY project](https://github.com/delta-diy-mri/delta-diy-mri.github.io).

Virtual Scanner Tabletop Games [Virtual Scanner](https://github.com/imr-framework/virtual-scanner/).

