import pandas as pd
import os

CSV_PATH = r"c:\Bone_Fracture\dataset_raw\dataset.csv"
FOLDER_STRUC = r"c:\Bone_Fracture\dataset_raw\folder_structure"

print(f"--- Checking {CSV_PATH} ---")
try:
    df = pd.read_csv(CSV_PATH, nrows=5)
    print("All Columns:", list(df.columns))
    # Print first row fully
    print("\nFirst Row:", df.iloc[0].to_dict())
    
    # Check if files from csv exist in images_part1
    sample_file = df.iloc[0]['filestem'] + ".png"
    print(f"\nSample file from CSV: {sample_file}")
    
except Exception as e:
    print(f"Error reading CSV: {e}")

print(f"\n--- Checking {FOLDER_STRUC} ---")
try:
    if os.path.exists(FOLDER_STRUC):
        print(os.listdir(FOLDER_STRUC))
    else:
        print("Folder not found.")
except Exception as e:
    print(f"Error listing folder: {e}")
