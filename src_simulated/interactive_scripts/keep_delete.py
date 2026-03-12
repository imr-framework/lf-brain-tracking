import os
import nibabel as nib
import tkinter as tk
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D
from PIL import Image

# -----------------------------
# Settings
# -----------------------------
source_dir = 'Data/Nipah_IRF_data/Low_field_data_DA/LFMRI_DATA_T2w'
valid_ext = ('.nii', '.nii.gz', '.png')

# -----------------------------
# Collect files
# -----------------------------
files = []
for root, _, fnames in os.walk(source_dir):
    for f in fnames:
        if f.lower().endswith(valid_ext):
            files.append(os.path.join(root, f))

files.sort()
print(f"🔍 Found {len(files)} files")

# -----------------------------
# Reviewer Class
# -----------------------------
class FileReviewer:
    def __init__(self, files):
        self.files = files
        self.idx = 0
        self.viewer = None
        self.fig = None

        self.root = tk.Tk()
        self.root.title("Keep / Delete Review")
        self.root.geometry("280x200")

        tk.Label(self.root, text="Review file", font=("Arial", 12)).pack(pady=10)

        tk.Button(self.root, text="✅ KEEP", width=20, command=self.keep).pack(pady=4)
        tk.Button(self.root, text="❌ DELETE", width=20, command=self.delete).pack(pady=4)
        tk.Button(self.root, text="⏭ SKIP", width=20, command=self.skip).pack(pady=4)
        tk.Button(self.root, text="🛑 QUIT", width=20, command=self.quit).pack(pady=6)

        self.load_current()
        self.root.mainloop()

    # -----------------------------
    def load_current(self):
        if self.idx >= len(self.files):
            print("🎉 Review complete")
            self.root.destroy()
            return

        self.close_viewer()
        fpath = self.files[self.idx]
        print(f"\n[{self.idx+1}/{len(self.files)}] {fpath}")

        try:
            if fpath.endswith(('.nii', '.nii.gz')):
                img = nib.load(fpath)
                data = img.get_fdata()
                self.viewer = OrthoSlicer3D(data)
                self.viewer.show()

            elif fpath.endswith('.png'):
                img = Image.open(fpath)
                self.fig, ax = plt.subplots(figsize=(5,5))
                ax.imshow(img, cmap='gray')
                ax.set_title(os.path.basename(fpath))
                ax.axis('off')
                plt.show(block=False)

        except Exception as e:
            print(f"⚠️ Failed to open: {e}")
            self.idx += 1
            self.load_current()

    # -----------------------------
    def close_viewer(self):
        try:
            if self.viewer:
                self.viewer.close()
            if self.fig:
                plt.close(self.fig)
        except Exception:
            pass
        self.viewer = None
        self.fig = None

    def keep(self):
        print("✅ Kept (no action taken)")
        self.next()

    def delete(self):
        f = self.files[self.idx]
        try:
            os.remove(f)
            print(f"❌ Deleted: {os.path.basename(f)}")
        except Exception as e:
            print(f"⚠️ Failed to delete {f}: {e}")
        self.next()

    def skip(self):
        print("⏭ Skipped")
        self.next()

    def quit(self):
        print("🛑 Quit by user")
        self.close_viewer()
        self.root.destroy()
        exit(0)

    def next(self):
        self.close_viewer()
        self.idx += 1
        self.load_current()

# -----------------------------
# Run
# -----------------------------
FileReviewer(files)
