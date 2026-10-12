import pandas as pd
import json
import os
import hashlib

file_path = 'data/raw/blood_donor_dataset_modeling.csv'
expected_cols = {'city', 'blood_group', 'months_since_first_donation', 'number_of_donation', 'pints_donated', 'availability'}

if not os.path.exists(file_path):
    print("File not found.")
    exit(1)

df = pd.read_csv(file_path)
actual_cols = set(df.columns)

if not actual_cols.issubset(expected_cols) or not expected_cols.issubset(actual_cols):
    print(f"ERROR: Column mismatch. Expected: {expected_cols}. Actual: {actual_cols}")
    exit(1)

audit = {
    'rows': len(df),
    'columns': len(df.columns),
    'dtypes': df.dtypes.astype(str).to_dict(),
    'missing_values': df.isnull().sum().to_dict(),
    'duplicates': int(df.duplicated().sum()),
    'availability_dist': df['availability'].value_counts(dropna=False).to_dict(),
    'categorical_dist': {
        'city': df['city'].value_counts(dropna=False).to_dict(),
        'blood_group': df['blood_group'].value_counts(dropna=False).to_dict()
    },
    'numeric_stats': df[['months_since_first_donation', 'number_of_donation', 'pints_donated']].describe().to_dict()
}

with open('data/raw/blood_donor_dataset_modeling.csv', 'rb') as f:
    audit['checksum_sha256'] = hashlib.sha256(f.read()).hexdigest()

os.makedirs('reports', exist_ok=True)
os.makedirs('reports/tables', exist_ok=True)

with open('reports/data_audit.json', 'w') as f:
    json.dump(audit, f, indent=4)

df['availability'].value_counts().reset_index().rename(columns={'index': 'availability', 'availability': 'count', 'count': 'count'}).to_csv('reports/tables/availability_class_distribution.csv', index=False)

print(json.dumps(audit, indent=2))
