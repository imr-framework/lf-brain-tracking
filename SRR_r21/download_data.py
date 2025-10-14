# Install openneuro-py if not already installed
# !pip install openneuro-py

from openneuro import download
import os

# Target folder where dataset will be downloaded
target_dir = "ds006557"

# Make sure target directory exists
os.makedirs(target_dir, exist_ok=True)

print("Starting download of ds006557 version 1.0.2 ...")

# Download dataset
download(
    dataset="ds006557",
    target_dir=target_dir,
    tag="1.0.2"   # version
)

print("✅ Download complete! Dataset stored in:", os.path.abspath(target_dir))
