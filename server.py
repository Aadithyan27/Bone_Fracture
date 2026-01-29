import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io

# --- Configuration ---
MODEL_PATH = r"C:\Users\aadit\Downloads\FP_Artefact2191136-1\FP_Artefact2191136\Bone_Fracture_Detection\models\best_bone_fracture_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Healthy', 'Fracture']

app = Flask(__name__)
# Enable CORS for all routes, specifically allowing the frontend origin
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Model Architecture (from Notebook) ---
class BoneFractureClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False): # Pretrained=False for inference since we load weights
        super(BoneFractureClassifier, self).__init__()
        # Load ResNet50 structure
        self.resnet = models.resnet50(weights=None) # Use weights=None for modern torchvision
        
        # Recreate the architecture exactly as trained
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# --- Grad-CAM Implementation (from Notebook) ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        # Backward pass for the target class
        output[0, target_class].backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global Average Pooling of gradients to get weights
        weights = gradients.mean(dim=(1, 2))
        
        # Create CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), target_class, output

# --- Utilities ---
def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = BoneFractureClassifier(num_classes=2, pretrained=False)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Handle different saving methods (state_dict vs full checkpoint)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def apply_gradcam_overlay(original_image_pil, cam):
    # Convert PIL to CV2 (RGB -> BGR usually, but we keep RGB for logic)
    img_np = np.array(original_image_pil)
    height, width = img_np.shape[:2]
    
    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (width, height))
    
    # Create heatmap
    heatmap = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert heatmap to RGB (from BGR)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay: 0.4 heatmap + 0.6 original
    overlay = heatmap_colored * 0.4 + img_np * 0.6
    overlay = np.uint8(overlay)
    
    return Image.fromarray(overlay)

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image, transform(image).unsqueeze(0).to(DEVICE)

# --- Initialization ---
classifier = load_model()
if classifier:
    # Target layer for ResNet50 is usually layer4[-1]
    gradcam = GradCAM(classifier, classifier.resnet.layer4[-1])
else:
    print("WARNING: Classifier could not be initialized.")
    gradcam = None

# --- API Routes ---

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "device": str(DEVICE)}), 200

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read and preprocess
        image_bytes = file.read()
        original_image, input_tensor = transform_image(image_bytes)
        
        # Inference & Grad-CAM
        cam, pred_class_idx, output = gradcam.generate_cam(input_tensor)
        
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0, pred_class_idx].item()
        prediction = CLASS_NAMES[pred_class_idx]
        
        # Generate Overlay Image
        overlay_image = apply_gradcam_overlay(original_image, cam)
        
        # Encode result image to base64
        buffered = io.BytesIO()
        overlay_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return jsonify({
            "prediction": prediction,
            "confidence": f"{confidence * 100:.2f}%",
            "probabilities": {
                CLASS_NAMES[0]: f"{probabilities[0][0].item() * 100:.2f}%",
                CLASS_NAMES[1]: f"{probabilities[0][1].item() * 100:.2f}%"
            },
            "gradcam_image": f"data:image/jpeg;base64,{img_str}"
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask Server...")
    # Run on 0.0.0.0 to be accessible, port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
