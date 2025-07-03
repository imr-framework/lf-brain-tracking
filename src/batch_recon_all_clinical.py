import subprocess
import os

# Parameters (customize as needed)
subjects = ["sub-0012", "sub-0013", "sub-0014", "sub-0015", "sub-0016"]

session = "ses-01"
tp = "tp1"
tp_zssr = "zssr.tp"
base_input_dir = "/Users/niyathigirish/Documents/JHU/Processed"
recon_script = "recon-all-clinical.sh"

def run_recon(subject_id):
    t2_path = os.path.join(base_input_dir, subject_id, session, "anat", f"{subject_id}_{session}_T2w.nii")
    output_dir = os.path.join(base_input_dir, subject_id, session)

    if not os.path.exists(t2_path):
        print(f"[ERROR] T2 file not found for {subject_id} at {t2_path}")
        return

    command = [
        recon_script,
        t2_path,
        f"{subject_id}.{tp}",
        "1",  # Possibly flag or session index
        output_dir
    ]

    try:
        print(f"[INFO] Running recon-all-clinical for {subject_id}")
        subprocess.run(command, check=True)
        print(f"[SUCCESS] Finished processing {subject_id}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed on {subject_id}: {e}")

def run_recon_zssr(subject_id):
    t2_zssr_path = os.path.join(base_input_dir, subject_id, session, "anat", f"{subject_id}_{session}_zssr_noise_True.nii")
    output_dir_zssr = os.path.join(base_input_dir, subject_id, session)

    if not os.path.exists(t2_zssr_path):
        print(f"[ERROR] T2 file not found for {subject_id} at {t2_zssr_path}")
        return

    command = [
        recon_script,
        t2_zssr_path,
        f"{subject_id}.{tp_zssr}"
        "1",  # Possibly flag or session index
        output_dir_zssr
    ]

    try:
        print(f"[INFO] Running recon-all-clinical for {subject_id}")
        subprocess.run(command, check=True)
        print(f"[SUCCESS] Finished processing {subject_id}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed on {subject_id}: {e}")

if __name__ == "__main__":
    for subject in subjects:
        run_recon(subject)
        run_recon_zssr(subject)