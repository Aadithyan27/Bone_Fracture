import os

YOLO_DIR = r"c:\Bone_Fracture\dataset_raw\folder_structure\yolov5"
PARTS = [f"images_part{i}" for i in range(1, 5)]
ROOT_DIR = r"c:\Bone_Fracture\dataset_raw"

def check_counts():
    print("Checking counts...")
    
    # Check labels
    label_dir = os.path.join(YOLO_DIR, 'labels')
    if os.path.exists(label_dir):
        # Recursively count
        label_count = sum([len(files) for r, d, files in os.walk(label_dir)])
        print(f"Labels in {label_dir}: {label_count}")
        # Show subfolders
        print(f"Subdirs in labels: {os.listdir(label_dir)}")
    
    # Check images inside yolov5 (if any)
    image_dir = os.path.join(YOLO_DIR, 'images')
    if os.path.exists(image_dir):
        image_count = sum([len(files) for r, d, files in os.walk(image_dir)])
        print(f"Images in {image_dir}: {image_count}")
        print(f"Subdirs in images: {os.listdir(image_dir)}")

    # Check images in parts
    total_images = 0
    for part in PARTS:
        p_path = os.path.join(ROOT_DIR, part)
        if os.path.exists(p_path):
             c = len(os.listdir(p_path))
             print(f"{part}: {c}")
             total_images += c
    print(f"Total separate images: {total_images}")

if __name__ == "__main__":
    check_counts()
