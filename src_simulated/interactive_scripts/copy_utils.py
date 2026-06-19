import os
import shutil

source_dir = "Data/Nipah_IRF_data/Low_field_data_DA/LFMRI_DATA_T2w"
target_dir = "Data/Nipah_IRF_data/Low_field_data_DA/test_da_lf"

os.makedirs(target_dir, exist_ok=True)

copied = 0

for root, dirs, files in os.walk(source_dir):
    # recreate directory structure
    rel_path = os.path.relpath(root, source_dir)
    dest_root = os.path.join(target_dir, rel_path)
    os.makedirs(dest_root, exist_ok=True)

    for f in files:
        src_file = os.path.join(root, f)
        dst_file = os.path.join(dest_root, f)
        shutil.copy2(src_file, dst_file)  # copy with metadata
        copied += 1

print(f"📁 Copied {copied} files from {source_dir} → {target_dir}")
