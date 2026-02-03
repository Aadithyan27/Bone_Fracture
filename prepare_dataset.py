import os
import shutil
import random
import yaml
from pathlib import Path
from tqdm import tqdm

# Config
RAW_DIR = Path(r"c:\Bone_Fracture\dataset_raw")
OUTPUT_DIR = Path(r"c:\Bone_Fracture\dataset")
PARTS = [f"images_part{i}" for i in range(1, 5)]
LABEL_SOURCE = RAW_DIR / "folder_structure/yolov5/labels"

# Classes from meta.yaml
CLASSES = [
    'boneanomaly', 'bonelesion', 'foreignbody', 'fracture', 
    'metal', 'periostealreaction', 'pronatorsign', 'softtissue', 'text'
]

def prepare():
    # 1. Gather all image paths
    print("Gathering image paths...")
    all_images = []
    for part in PARTS:
        part_path = RAW_DIR / part
        if part_path.exists():
            images = list(part_path.glob("*.png"))
            all_images.extend(images)
    
    print(f"Found {len(all_images)} images.")
    if len(all_images) == 0:
        print("No images found! Check paths.")
        return

    # 2. Shuffle and Split
    random.seed(42)
    random.shuffle(all_images)
    
    n = len(all_images)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    splits = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }
    
    print(f"Split counts: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    # 3. Create Directories and Copy
    # Structure: dataset/images/train, dataset/labels/train, etc.
    if OUTPUT_DIR.exists():
        print(f"Warning: Output directory {OUTPUT_DIR} exists. Merging/Overwriting...")
    
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {split} split...")
        for img_path in tqdm(splits[split]):
            # Copy Image
            shutil.copy2(img_path, OUTPUT_DIR / "images" / split / img_path.name)
            
            # Find and Copy Label
            # Label name is same as image stem + .txt
            label_name = img_path.stem + ".txt"
            label_src = LABEL_SOURCE / label_name
            
            if label_src.exists():
                shutil.copy2(label_src, OUTPUT_DIR / "labels" / split / label_name)
            else:
                # Create empty label file if missing (means background/healthy in YOLO)
                # But GRAZPEDWRI "healthy" should have empty txt? 
                # Or maybe only fractured ones have txt?
                # If src missing, assume empty.
                (OUTPUT_DIR / "labels" / split / label_name).touch()

    # 4. Create data.yaml
    data_yaml = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print(f"\nDataset preparation complete. Saved to {OUTPUT_DIR}")
    print(f"Data.yaml created at {OUTPUT_DIR / 'data.yaml'}")

if __name__ == "__main__":
    prepare()
