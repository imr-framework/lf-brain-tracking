import os
import shutil
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import tkinter as tk

# -----------------------------
# Settings
# -----------------------------
source_dir = 'Data/Nipah_IRF_data/LFMRI_DATA_IRF_NIFTI_best_Corrected'

target_t2w = 'Data/Nipah_IRF_data/Low_field_data_DA/LFMRI_DATA_T2w'
target_t1w = 'Data/Nipah_IRF_data/Low_field_data_DA/LFMRI_DATA_T1w'

os.makedirs(target_t2w, exist_ok=True)
os.makedirs(target_t1w, exist_ok=True)

# -----------------------------
# Collect all NIfTI files
# -----------------------------
nii_files = []
for root, _, files in os.walk(source_dir):
    for f in files:
        if f.endswith('.nii.gz'):
            nii_files.append(os.path.join(root, f))

nii_files.sort()
print(f"🔍 Found {len(nii_files)} NIfTI files")

# -----------------------------
# GUI Controller Class
# -----------------------------
class MRIReviewer:
    def __init__(self, files):
        self.files = files
        self.index = 0
        self.slicer = None

        self.root = tk.Tk()
        self.root.title("MRI Classification")
        self.root.geometry("300x160")

        tk.Label(self.root, text="Classify Current Scan", font=("Arial", 12)).pack(pady=10)

        tk.Button(self.root, text="✅ T2w", width=20, command=self.save_t2w).pack(pady=3)
        tk.Button(self.root, text="📁 T1w", width=20, command=self.save_t1w).pack(pady=3)
        tk.Button(self.root, text="⏭ Skip", width=20, command=self.skip).pack(pady=3)
        tk.Button(self.root, text="🛑 Quit", width=20, command=self.quit).pack(pady=5)

        self.load_current()

        self.root.mainloop()

    # -----------------------------
    def load_current(self):
        if self.index >= len(self.files):
            print("🎉 Review complete.")
            self.root.destroy()
            return

        fpath = self.files[self.index]
        print(f"\n[{self.index+1}/{len(self.files)}] {fpath}")

        try:
            img = nib.load(fpath)
            data = img.get_fdata()
            self.slicer = OrthoSlicer3D(data)
            self.slicer.show()
        except Exception as e:
            print(f"⚠️ Failed to open: {e}")
            self.index += 1
            self.load_current()

    # -----------------------------
    def close_slicer(self):
        try:
            if self.slicer:
                self.slicer.close()
        except Exception:
            pass

    def save_t2w(self):
        fpath = self.files[self.index]
        shutil.copy2(fpath, target_t2w)
        print(f"✅ Saved to T2w: {os.path.basename(fpath)}")
        self.next()

    def save_t1w(self):
        fpath = self.files[self.index]
        shutil.copy2(fpath, target_t1w)
        print(f"📁 Saved to T1w: {os.path.basename(fpath)}")
        self.next()

    def skip(self):
        print("⏭ Skipped")
        self.next()

    def quit(self):
        print("🛑 Quit by user")
        self.close_slicer()
        self.root.destroy()
        exit(0)

    def next(self):
        self.close_slicer()
        self.index += 1
        self.load_current()

# -----------------------------
# Run reviewer
# -----------------------------
MRIReviewer(nii_files)
