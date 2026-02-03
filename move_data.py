import os
import shutil
from pathlib import Path

# Config
SOURCE_PATH = r"C:\Users\aadit\Downloads\14825193"
DEST_PATH = r"c:\Bone_Fracture\dataset_raw"

def inspect_and_copy():
    print(f"Inspecting source: {SOURCE_PATH}")
    
    if not os.path.exists(SOURCE_PATH):
        print("Source path not found!")
        return

    # List first level files to understand structure
    print("\n--- Root Contents ---")
    try:
        items = os.listdir(SOURCE_PATH)
        for item in items[:20]: # Show first 20 items
            print(item)
        if len(items) > 20: 
            print("...")
    except Exception as e:
        print(f"Error listing source: {e}")
        return

    # Create destination
    print(f"\nCreating destination: {DEST_PATH}")
    os.makedirs(DEST_PATH, exist_ok=True)
    
    # Check if we should copy (don't formatted copy if it's huge, maybe just list first)
    # But user wants us to "create the dataset", so we need the files.
    # Let's count files first.
    print("\nCounting files...")
    file_count = sum([len(files) for r, d, files in os.walk(SOURCE_PATH)])
    print(f"Found {file_count} files.")
    
    print(f"\nCopying files to {DEST_PATH}...")
    # Using shutil.copytree with dirs_exist_ok=True to merge/update
    try:
        shutil.copytree(SOURCE_PATH, DEST_PATH, dirs_exist_ok=True)
        print("Copy complete.")
    except Exception as e:
        print(f"Error copying: {e}")

if __name__ == "__main__":
    inspect_and_copy()
