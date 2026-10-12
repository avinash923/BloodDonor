import pandas as pd

file_path = 'data/raw/blood_donor_dataset.csv'
try:
    df = pd.read_csv(file_path)
    
    print("--- Dataset Shape ---")
    print(df.shape)
    
    print("\n--- Columns and Types ---")
    print(df.dtypes)
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Duplicates ---")
    print(f"Total duplicates: {df.duplicated().sum()}")
    print(f"Donor ID duplicates: {df['donor_id'].duplicated().sum() if 'donor_id' in df.columns else 'N/A'}")
    
    print("\n--- Class Balance (availability) ---")
    if 'availability' in df.columns:
        print(df['availability'].value_counts(dropna=False))
    
    print("\n--- Summary Stats (Numeric) ---")
    numeric_cols = ['months_since_first_donation', 'number_of_donation', 'pints_donated']
    for col in numeric_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].describe())

    print("\n--- Unique Values for Categoricals ---")
    for col in ['blood_group', 'city']:
        if col in df.columns:
            print(f"\n{col} (top 5):")
            print(df[col].value_counts(dropna=False).head(5))

except Exception as e:
    print(f"Error reading dataset: {e}")
