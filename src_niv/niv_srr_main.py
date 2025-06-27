import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
from src_niv.prep_data import data_ops

nhp_num = '26184'  # Replace with your actual data path
nhp_base_path = 'Data/IRF_3T'  # Replace with your actual base path
nhp_data_path = f'{nhp_base_path}/{nhp_num}'  # Construct the full path to the DICOM folder
nhp_26184 = data_ops(nhp_data_path)
nhp_26184_dates_volumes = nhp_26184.volume
print(f"Loaded data for NHP {nhp_num} with dates and volumes: {nhp_26184_dates_volumes}")

# Read the DICOM data

# Prepare the data

# Display the data

# Perform and call SRR reconstruction framework

# Save the SRR results to NIfTI files

# Display the SRR results

# Compare the SRR results with the original data

# Identify and visualize differences using XAI

# Interpret hallucinations using XAI and improve the SRR framework results 
