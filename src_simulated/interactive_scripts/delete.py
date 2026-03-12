import os

source_dir = "Data/Nipah_IRF_data/Low_field_data_DA/test_da_lf"

deleted_count = 0

for root, dirs, files in os.walk(source_dir):
    for f in files:
        fpath = os.path.join(root, f)
        try:
            os.remove(fpath)
            deleted_count += 1
        except Exception as e:
            print(f"⚠️ Failed to delete {fpath}: {e}")

print(f"🗑️ Deleted {deleted_count} files from {source_dir}")
