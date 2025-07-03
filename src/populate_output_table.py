import pandas as pd
import os
from pathlib import Path

# === CONFIGURATION ===
base_dir = "/Users/niyathigirish/Documents/JHU/Processed"  # <-- UPDATE THIS
output_excel = "Output_Combined.xlsx"

# === REGION MAPPING ===
region_map = {
    "ICV": ["total intracranial"],
    "Gray Matter": ["left cerebral cortex", "right cerebral cortex"],
    "White Matter": ["left cerebral white matter", "right cerebral white matter"],
    "CSF": [
        "left lateral ventricle", "right lateral ventricle",
        "left inferior lateral ventricle", "right inferior lateral ventricle", "csf" , "3rd ventricle", "4th ventricle"
    ],
    "Amygdala": ["left amygdala", "right amygdala"],
    "Hippocampus": ["left hippocampus", "right hippocampus"],
    "Caudate": ["left caudate", "right caudate"],
    "Accumbens": ["left accumbens area", "right accumbens area"],
    "Putamen": ["left putamen", "right putamen"],
    "Pallidum": ["left pallidum", "right pallidum"],
    "Thalamus": ["left thalamus", "right thalamus"]
}

def extract_volumes(csv_path):
    try:
        print(f"📥 Reading: {csv_path}")
        df = pd.read_csv(csv_path)
        volumes = {}
        for region, cols in region_map.items():
            found_cols = [col for col in cols if col in df.columns]
            if found_cols:
                value = df[found_cols].sum(axis=1).values[0]
                print(f"   ✅ {region}: {value} from {found_cols}")
                volumes[region] = value
            else:
                print(f"   ⚠️ Columns missing for region '{region}'")
                volumes[region] = None
        return volumes
    except Exception as e:
        print(f"   ❌ Error reading {csv_path}: {e}")
        return {}

def build_subject_row(subj_id, base_path):
    print(f"\n🔍 Building row for {subj_id}")
    row = {"Subject": subj_id}
    tp_map = {"ses-01": "tp1", "ses-02": "tp2"}

    for session in ["ses-01", "ses-02"]:
        tp = tp_map[session]
        for method in ["T2W", "ZSSR"]:
            folder_name = f"{subj_id}.{tp}" if method == "T2W" else f"{subj_id}.zssr.{tp}"
            csv_path = Path(base_path) / subj_id / session / folder_name / "stats" / "synthseg.vol.csv"
            key_prefix = f"{method}_{session}"

            if csv_path.exists():
                print(f"👉 Found {key_prefix} for {subj_id}")
                volumes = extract_volumes(csv_path)
                for region, value in volumes.items():
                    row[f"{key_prefix}_{region}"] = value
            else:
                print(f"🚫 Missing: {csv_path}")
                for region in region_map:
                    row[f"{key_prefix}_{region}"] = None
    return row

# === LOAD OR INITIALIZE EXCEL FILE ===
if os.path.exists(output_excel):
    try:
        df_existing = pd.read_excel(output_excel)
        print(f"📂 Loaded existing file: {output_excel}")
        if "Subject" not in df_existing.columns:
            print("⚠️ No 'Subject' column found. Reinitializing DataFrame.")
            df_existing = pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Failed to read existing Excel: {e}")
        df_existing = pd.DataFrame()
else:
    print("📄 Creating new Excel file.")
    df_existing = pd.DataFrame()

# === FIND SUBJECT FOLDERS ===
subject_folders = sorted([d.name for d in Path(base_dir).glob("sub-*") if d.is_dir()])
print(f"\n📁 Found {len(subject_folders)} subject folders")

# === PROCESS EACH SUBJECT ===
for subj in subject_folders:
    if "Subject" in df_existing.columns and subj in df_existing["Subject"].values:
        print(f"🔁 Skipping {subj} — already processed.")
        continue
    print(f"🧠 Processing subject: {subj}")
    row_data = build_subject_row(subj, base_dir)
    df_existing = pd.concat([df_existing, pd.DataFrame([row_data])], ignore_index=True)

# === SAVE TO EXCEL ===
df_existing.to_excel(output_excel, index=False)
print(f"\n✅ Data saved to '{output_excel}' with {len(df_existing)} total subjects.")
