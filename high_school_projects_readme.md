# 0.05 T Brain Tracking – **Setup & Usage Guide**

*Branch: `high_school_projects`*

This README is a **how‑to** for installing and running the low‑field (0.05 T) MRI processing pipeline that combines **Zero‑Shot Self‑Super‑Resolution (ZSSR)** with the **recon‑all‑clinical** FreeSurfer wrapper.  No results or study findings are included here—just the essentials to get you up and running.

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

## Why 0.05 T + ZSSR?
Conventional 3 T MRI offers high resolution but is expensive and scarce in many regions. Portable very‑low‑field scanners (< 0.1 T) are cheaper and easier to deploy, yet suffer from low signal‑to‑noise and limited spatial resolution.

**ZSSR** enhances each slice without needing an external training set, making it ideal for:
* boosting SNR/resolution on‑scanner
* enabling downstream segmentation & volumetry with FreeSurfer 8

> The commands below are distilled from our full workflow docs.  See the referenced cheat‑sheets for advanced options. fileciteturn2file0 fileciteturn2file1

---

## Prerequisites
| Requirement | Notes |
|-------------|-------|
| **macOS 12+** or **Ubuntu 20.04+** | Tested on Apple Silicon and x86_64. |
| **Python ≥ 3.9** | Use a virtualenv or Conda. |
| **FreeSurfer 8.0.0** | Install to `/Applications/freesurfer/8.0.0` (mac) or `$HOME/freesurfer/8.0.0` (Linux). |
| **Git** & **wget/curl** | Standard CLI tools. |

### 1 — Install FreeSurfer (macOS example)
```bash
# download & extract
mkdir -p /Applications/freesurfer && cd /Applications/freesurfer
curl -LO https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/8.0.0/freesurfer-macos-8.0.0.tar.gz
 tar -xzf freesurfer-macos-8.0.0.tar.gz

# add to shell profile (~/.zshrc or ~/.bash_profile)
export FREESURFER_HOME=/Applications/freesurfer/8.0.0
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=$FREESURFER_HOME/subjects
```
Reload the shell and verify:
```bash
echo $FREESURFER_HOME && freeview --version
```

### 2 — Clone & install Python deps
```bash
git clone --branch high_school_projects https://github.com/imr-framework/lf-brain-tracking.git
cd lf-brain-tracking
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_high_school.txt
```
---

## Typical Workflow

### A. Enhance an image with ZSSR
```bash
python tools/zssr_run.py \
  --input  data/sub-0001_T2w.nii.gz \
  --config configs/zssr_default.yaml \
  --out    outputs/sub-0001_T2w_zssr.nii.gz
```
*Outputs* a new NIfTI ending in `_zssr.nii.gz`.

### B. Segment & QC with recon‑all‑clinical
```bash
recon-all-clinical.sh \
  outputs/sub-0001_T2w_zssr.nii.gz \
  sub-0001.tp1 \
  4 \
  outputs/fs_sub-0001
```
*Arguments* – `<input> <subjectID> <#threads> <output_dir>`

### C. Visualise results (optional)
```bash
freeview -v \
  $SUBJECTS_DIR/sub-0001.tp1/mri/brain.mgz \
  $SUBJECTS_DIR/sub-0001.tp1/mri/aseg.mgz:colormap=lut:opacity=0.4
```
---

## Common Issues & Fixes
| Issue | Fix |
|-------|-----|
| **Terminal hang during TensorFlow step** | Force CPU: `export CUDA_VISIBLE_DEVICES="-1"` & `export TF_ENABLE_ONEDNN_OPTS=0` |
| `freeview: command not found` | Re‑source FreeSurfer setup script. |
| Permissions errors in output dir | `chmod -R u+w <dir>` or adjust `$SUBJECTS_DIR`. |
| Recon fails with *MRInormFindControlPoints* | Use higher‑quality input or run SynthSR beforehand. |

More troubleshooting tips live in `docs/wiki_freesurfer_zssr.md`.

---

## Directory Layout (this branch)
```
high_school_projects/
├── data/              # sample NIfTI files
├── tools/             # CLI helpers (zssr_run, volumetry, etc.)
├── configs/           # YAML configs for ZSSR + recon
├── docs/              # setup wikis & command refs
└── notebooks/         # optional Jupyter demos
```

---

## Contributors
* **Sairam Geethanath — Principal Investigator
* **Ajay Sharma** — ?? 
* **Niyathi Girish** — ??

---

## License
Released under the **MIT License**. See `LICENSE` for details.

