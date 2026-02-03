import pandas as pd

SOURCE_CSV = r"C:\Users\aadit\Downloads\14825193\dataset.csv"

try:
    df = pd.read_csv(SOURCE_CSV, nrows=5)
    print("Columns:", df.columns.tolist())
    print(df.head())
    
    # Check for useful columns
    if 'fileref' in df.columns or 'filename' in df.columns:
        print("\nHas filename column.")
    if 'region_shape_attributes' in df.columns or 'bbox' in df.columns:
        print("\nHas bbox info.")
        
except Exception as e:
    print(f"Error reading CSV: {e}")
