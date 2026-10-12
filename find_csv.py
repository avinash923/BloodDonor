from pathlib import Path
import os
for root, dirs, files in os.walk('C:\\BloodDonor'):
    if 'blood_donor_dataset.csv' in files:
        print(f"Found at: {os.path.join(root, 'blood_donor_dataset.csv')}")
