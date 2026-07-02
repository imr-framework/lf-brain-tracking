# Low‑Field Brain Tracking Using 0.05 T MRI — *High‑School Projects*

> **Branch:** `high_school_projects`   |   *Focused feasibility studies that extend the [`lf-brain-tracking`](https://github.com/imr-framework/lf-brain-tracking) framework to very‑low‑field (0.05 T) data.*

<div align="center">
<img src="docs/img/lowfield_banner.png" width="80%" alt="0.05 T low‑field scanner and example ZSSR recon"/>
</div>

---

## Aims

| # | Title | Goal |
|---|-------|------|
| **1** | Dense temporal sampling + ZSSR | Acquire frequent 0.05 T scans and apply **Z**ero‑**S**hot **S**elf‑super‑**R**esolution to approach 3 T volumetric accuracy. |
| **2** | Autonomous 0.05 T MRI | Provide containerised, on‑scanner pipelines for immediate QC and brain‑metric extraction. |

---

## Mini‑Projects

| Project | Lead | Objective | Status |
|---------|------|-----------|--------|
| **ZSSR‑Volumetry** | Gabriel | Can ZSSR‑enhanced single‑orientation scans yield accurate volumes (ICV, GM, WM, etc.)? | 🟡 Data analysis |
| **ZSSR‑Growth‑Rates** | Niyathi Girish | Use ZSSR reconstructions to compute longitudinal brain‑growth slopes across dense timepoints. | 🟡 Data prep |

> Want to contribute? See [Contributing](#-contributing) below.

---

## Pipeline Overview

```
┌──────────────┐     ┌────────────────────┐     ┌──────────────────┐
│  Raw 0.05 T  │ →  │  Motion & Noise    │ →  │    ZSSR‑SR        │
│   NIfTI      │     │  Correction (Nifty)│     │  (slice‑wise)    │
└──────────────┘     └────────────────────┘     └────────┬─────────┘
                                                         │
                                   ⭢  FreeSurfer  recon‑all‑clinical + SynthSeg/SynthSR
                                                         │
                              📊 Volumetry • Growth rates • Segment QC
```

---

## Installation & Environment

<details>
<summary><strong>1. Install FreeSurfer ( macOS )</strong></summary>

1. Download v8.0.0 from the [FreeSurfer site](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall).
2. Extract to `/Applications/freesurfer/8.0.0/`.
3. Add to shell profile:

```bash
export FREESURFER_HOME=/Applications/freesurfer/8.0.0
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=$FREESURFER_HOME/subjects
```
4. Reload: `source ~/.zshrc`  or `source ~/.bash_profile`.

Validate with `echo $FREESURFER_HOME` and `freeview --version`. fileciteturn2file1
</details>

<details>
<summary><strong>2. Clone the repo & install Python deps</strong></summary>

```bash
git clone --branch high_school_projects https://github.com/imr-framework/lf-brain-tracking.git
cd lf-brain-tracking
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_high_school.txt
```

Core libs: *numpy, scipy, nibabel, torch, monai, scikit‑image, tqdm*. fileciteturn2file0
</details>

<details>
<summary><strong>3. Optional: Docker</strong></summary>
A Dockerfile (`docker/Dockerfile.zssr`) yields a fully reproducible environment:

```bash
docker build -t lowfield_zssr -f docker/Dockerfile.zssr .
```
</details>

---

## Quick‑Start Demo

```bash
# Enhance & segment a sample scan
python tools/zssr_run.py \
  --input  sample_data/low_field.nii.gz \
  --config configs/zssr_tumor.yaml \
  --out    outputs/low_field_zssr.nii.gz

# Optional volumetry via FreeSurfer wrapper
recon-all-clinical.sh \
  outputs/low_field_zssr.nii.gz demo_zssr 4 ./outputs/demo_subject
```

*Ensure GPU‑related hangs are avoided on macOS/Metal:*
```bash
export CUDA_VISIBLE_DEVICES="-1"
export TF_ENABLE_ONEDNN_OPTS=0
```

---

## Command Cheat‑Sheet

| Task | Command | Notes |
|------|---------|-------|
| **Set environment** | `export FREESURFER_HOME=...`<br>`source $FREESURFER_HOME/SetUpFreeSurfer.sh` | Run once per shell. |
| **Run ZSSR** | `python tools/zssr_run.py --config ...` | Outputs `*_zssr_noise_True.nii.gz`. |
| **Clinical pipeline** | `recon-all-clinical.sh <in.nii.gz> subjID.tp1 4 <outdir>` | SynthSeg + SynthSR integrated. |
| **Visualise** | `freeview -v $SUBJECTS_DIR/subj/mri/brain.mgz ...` | Overlay segmentations. |

See *`docs/Freesurfer_Commands_Reference.md`* for the full list. fileciteturn2file0

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| *Terminal hangs during TensorFlow steps* | `export CUDA_VISIBLE_DEVICES="-1"` and `export TF_ENABLE_ONEDNN_OPTS=0` |
| `freeview: command not found` | Source FreeSurfer setup again. |
| Permission errors in `Processed/` | `chmod -R u+w <path>` or use `sudo` sparingly. |
| Missing subject after recon | Check input path & BIDS naming. |

More issues & fixes are documented in **docs/wiki_freesurfer_zssr.md**. fileciteturn2file1

---

## Comparing Volumes

The pipeline outputs `synthseg.vol.csv` per session. Use the helper notebook `notebooks/compare_volumes.ipynb` or any spreadsheet to review differences between **Regular** and **ZSSR** volumes.

| Structure | Regular (mm³) | ZSSR (mm³) | Δ (%) |
|-----------|--------------|------------|-------|
| ICV | — | — | — |
| GM | — | — | — |
| WM | — | — | — |
| … | … | … | … |

---

## Contributing

1. Fork & create a feature branch: `feat/<brief_desc>`.
2. Adhere to **PEP‑8** and run `pre‑commit`.
3. Include/modify unit tests in `tests/`.
4. Open a PR and request review.

---

## Contributors

| Name | Role |
|------|------|
| **Sairam Geethanath** | Principal Investigator |
| **Ajay Sharma** | ?? |
| **Niyathi Girish** | ?? |

---

## Cite Us

```bibtex
@inproceedings{girish2026_zssr_lowfield,
  title     = {Zero-Shot Super‑Resolution Reconstruction for 0.05 T MRI},
  author    = {Girish, N. and Sharma, A. and Geethanath, S.},
  booktitle = {Proceedings of <Conference>},
  year      = {2026}
}
```

---

## 📄 License

This branch is released under the **MIT License** (see `LICENSE`).

