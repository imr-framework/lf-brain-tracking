![LF Brain Tracking](assets/logo2.jpeg)

# 🧠 Low-Field Brain Tracking Using 0.05T MRI

> **Developing open-source tools for motion correction, image enhancement, super-resolution reconstruction, and brain analysis for accessible low-field MRI.**

![Status](https://img.shields.io/badge/status-under%20development-orange)
![Version](https://img.shields.io/badge/version-v1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌍 Why Accessible MRI?

Brain morphology provides critical biomarkers for understanding neurodevelopment, aging, and neuropsychiatric disorders. Although **3T MRI** has substantially advanced brain imaging research, access to conventional MRI remains limited worldwide.

According to the **World Health Organization (WHO)**, nearly **two-thirds of the global population lacks access to MRI** because of the high cost, infrastructure requirements, power consumption, and specialized expertise needed to operate conventional high-field scanners.

Recent advances in **portable very-low-field MRI (<0.1T)** provide a promising alternative for longitudinal neuroimaging in community and resource-limited settings. However, these systems suffer from:

- Low signal-to-noise ratio (SNR)
- Lower spatial resolution
- Motion artifacts
- Reduced volumetric accuracy
- Limited diagnostic quality

To enable reliable longitudinal brain tracking, new computational methods are required to generate low-field MRI with image quality approaching that of conventional **3T MRI**.

---

## 🧠 Project Overview

**lf-brain-tracking** is an open-source framework that develops AI-powered tools for improving very-low-field MRI acquired at **0.05T**.

The project focuses on:

- Motion correction
- Orthogonal acquisition fusion
- Low-field MRI simulation
- Zero-shot denoising
- Super-resolution reconstruction
- Brain segmentation
- Brain parcellation
- Longitudinal brain volumetry
- Image quality assessment

---

## 🖼️ Pipeline

<p align="center">
  <img src="assets/lf-mri-pipeline.jpeg" width="900">
</p>

---

# 🚧 Development Status

This repository is **under active development**.

New modules, documentation, and examples are continuously being added as the project evolves.

Current development areas include:

- Motion correction
- Low-field MRI simulation
- Zero-shot denoising
- Super-resolution reconstruction
- Brain segmentation
- Brain parcellation
- Volumetric brain tracking
- Benchmark datasets
- Evaluation metrics

> **Note:** APIs and directory structures may change before the first stable release.

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



# Registration and Orthogonal Acquisition Combination

[![NiftyMIC](https://img.shields.io/badge/NiftyMIC-Motion%20Correction-blue)](https://github.com/gift-surg/NiftyMIC)

NiftyMIC is a research-focused toolkit for motion correction and volumetric image reconstruction. In **lf-brain-tracking**, NiftyMIC is used for motion correction and the combination of three orthogonal MRI acquisitions into a single high-quality reconstructed volume.


# Brain Parcellation

[![SynthSeg](https://img.shields.io/badge/SynthSeg-Low--Field%20Parcellation-green)](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg)
[![SuperSynth](https://img.shields.io/badge/SuperSynth-Low--Field%20Parcellation-orange)](https://github.com/shuohan-cao/SuperSynth)

Brain parcellation tools such as **SynthSeg** and **SuperSynth** are integrated to enable automated segmentation and anatomical parcellation of low-field MRI scans.


# 📚 Publications

This repository supports research on **low-field MRI**, **deep learning**, **super-resolution reconstruction**, **brain tracking**, and **MRI automation**.

---

## 🎯 Aim 1: Track Youth Brain Changes at 0.05T Using Densely Sampled Neuroimaging Data and DL-SRR

### Conference Proceedings

- **Girish, N., Sharma, A., & Geethanath, S.** (2026). *Zero-shot self-supervised super-resolution reconstruction of MRI to track brain changes using volumetry: Application to high and low-field data.* SPIE Medical Imaging.
- **Sharma, A., & Geethanath, S.** (2026). *Learning beyond interpolation: Zero-shot resolution enhancement for low-field MRI.* ISMRM Annual Meeting.
- **Sharma, A., Oiye, I. E., Byrum, R., Holbrook, M., Cong, Y., Calcagno, C., Mani, V., & Geethanath, S.** (2026). *Enhancing low-field MRI image quality for Nipah virus infection imaging using deep learning.* ISMRM Annual Meeting.
- **Sharma, A., Oiye, I. E., Byrum, R., Holbrook, M., Cong, Y., Calcagno, C., Mani, V., & Geethanath, S.** (2025). *Physics-informed low-field Nipah virus MRI image reconstruction of non-human primates in a BSL-4 facility.* ISMRM Annual Meeting.

<details>
<summary><strong>Preprints</strong></summary>

- **Oiye, I. E., Sharma, A., et al.** (2025). *A Hands-On Workshop for Constructing a Low-Field MRI System in Three Days.* arXiv.

</details>

<details>
<summary><strong>Journal Submission</strong></summary>

- **Sharma, A., Lu, H., Greenspan, H., Lin, D. D., & Geethanath, S.** (2025). *Enhancing Low-Field Magnetic Resonance Image Quality Using Deep Learning: Challenges, Opportunities, and Resources.* Under revision, **NMR in Biomedicine**.

</details>

<details>
<summary><strong>Ongoing Work</strong></summary>

- **Sharma, A., Oiye, I. E., Aggarwal, K., Mani, V., Calcagno, C., Laux, J., Cong, Y., Holbrook, M. R., & Geethanath, S.**
  *Feasibility of Very-Low-Field MRI for Imaging Nipah Virus-Infected Non-Human Primates in a Biosafety Level-4 Facility.*

</details>

---

## 🤖 Aim 2: Automate 0.05T MRI (Auto-MRI)

**Digital Twin Repository**

[![Auto-MRI](https://img.shields.io/badge/GitHub-Auto--MRI-blue?logo=github)](https://github.com/imr-framework/virtual-scanner-adt)

### Conference Proceedings

- **Kinyera, L., Oiye, I. E., Geethanath, A., Obungoloch, J., & Geethanath, S.** (2026). *An Autonomous Digital Twin Agent for the Parametric Design and Validation of Halbach Array Magnets for Low-Field MRI.* ISMRM Annual Meeting.

---

# 📖 Recommended Reading

| Topic | Reference |
|-------|-----------|
| Zero-shot Super-Resolution | Shocher et al., CVPR 2018 |
| Brain Parcellation | Billot et al., *SynthSeg*, Medical Image Analysis, 2023 |
| Very-Low-Field MRI | Aggarwal et al., ISMRM 2023 |
| Native Noise Modeling | Ssentamu et al., Frontiers in Neuroimaging, 2025 |

---

# 🌍 Related Open-Source Projects

| Project | Description |
|----------|-------------|
| 🛠 **[DELTA DIY MRI](https://github.com/delta-diy-mri/delta-diy-mri.github.io)** | Learn MRI through hands-on system construction, experiments, and interactive educational resources. |
| 🎮 **[Virtual Scanner](https://github.com/imr-framework/virtual-scanner)** | An educational MRI simulation platform featuring virtual scanner tabletop games and MRI concepts. |
| 🤖 **[Auto-MRI](https://github.com/imr-framework/virtual-scanner-adt)** | Autonomous digital twin framework for low-field MRI scanner design, optimization, and validation. |

---

## Citation

If you use **lf-brain-tracking** in your research, please cite the relevant publications listed above.