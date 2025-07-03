import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# Define regions and their corresponding columns
region_map = {
    "total_gray_matter": ["left cerebral cortex", "right cerebral cortex"],
    "total_white_matter": ["left cerebral white matter", "right cerebral white matter"],
    "cerebrospinal_fluid": ["left lateral ventricle", "right lateral ventricle",
                             "left inferior lateral ventricle", "right inferior lateral ventricle"],
    "amygdala": ["left amygdala", "right amygdala"],
    "hippocampus": ["left hippocampus", "right hippocampus"],
    "caudate": ["left caudate", "right caudate"],
    "accumbens": ["left accumbens area", "right accumbens area"],
    "putamen": ["left putamen", "right putamen"],
    "pallidum": ["left pallidum", "right pallidum"],
    "thalamus": ["left thalamus", "right thalamus"],
    "hypothalamus": ["hypothalamus"]
}

def extract_metadata(path_str):
    """Infer Subject, Session, and Method from file path."""
    subject = re.search(r"sub-\d+", path_str)
    session = re.search(r"ses-\d+", path_str)
    method = "TW" if "tw" in path_str.lower() else "ZSSR" if "zssr" in path_str.lower() else "Unknown"
    return subject.group() if subject else "sub-unknown", session.group() if session else "ses-unknown", method

def extract_volumes(csv_file):
    """Read CSV and sum volumes per region."""
    try:
        df = pd.read_csv(csv_file)
    except:
        return []

    subject, session, method = extract_metadata(str(csv_file))
    volumes = []
    for region, cols in region_map.items():
        available_cols = [col for col in cols if col in df.columns]
        if available_cols:
            vol = df[available_cols].sum(axis=1).values[0]
            volumes.append({
                "Subject": subject,
                "Session": session,
                "Method": method,
                "Region": region,
                "Volume": vol
            })
    return volumes

def collect_all_data(base_dir):
    """Traverse all subfolders for synthseg.vol.csv files."""
    all_data = []
    for file_path in Path(base_dir).rglob("synthseg.vol.csv"):
        data = extract_volumes(file_path)
        all_data.extend(data)
    return pd.DataFrame(all_data)

def plot_volumes(df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    for subject in df['Subject'].unique():
        df_sub = df[df['Subject'] == subject]
        for region in region_map:
            df_region = df_sub[df_sub['Region'] == region]
            if df_region.empty:
                continue
            plt.figure(figsize=(8,6))
            sns.barplot(data=df_region, x="Session", y="Volume", hue="Method", palette="Set2")
            plt.title(f"{subject} - {region.replace('_', ' ').title()}")
            plt.ylabel("Volume (mm³)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{subject}_{region}.png"))
            plt.close()

# ==== USAGE ====

# Set your recon-all output root directory here:
base_directory = "/Documents/JHU/Processed" 

# Step 1: Collect data
df_all = collect_all_data(base_directory)

# Step 2: Plot and save figures
plot_volumes(df_all)