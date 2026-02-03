from ultralytics import YOLO
import torch

# Config
DATA_YAML = r"c:\Bone_Fracture\dataset\data.yaml"
MODEL_NAME = "yolo11l.pt" # Using YOLO11 Large for high accuracy
EPOCHS = 50
IMGSZ = 640
BATCH = 8 # Adjust based on VRAM (RTX 5070 has 12GB? 16GB? Should handle 8-16)

def train():
    # Check GPU
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("WARNING: GPU not found. Training will be slow.")
        device = 'cpu'

    # Load Model
    print(f"Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    # Train
    print("Starting training...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        project="c:/Bone_Fracture/runs/train",
        name="yolo11l_fracture_det",
        exist_ok=True,
        plots=True
    )
    
    print("Training complete.")
    print(f"Results saved to {results.save_dir}")

if __name__ == "__main__":
    train()
