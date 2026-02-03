import os

SOURCE_DIR = r"C:\Users\aadit\Downloads\14825193\images_part1"

try:
    files = os.listdir(SOURCE_DIR)[:20]
    print("Files in images_part1:")
    for f in files:
        print(f)
        
    extensions = set([os.path.splitext(f)[1] for f in os.listdir(SOURCE_DIR)])
    print(f"\nExtensions found: {extensions}")

except Exception as e:
    print(f"Error: {e}")
