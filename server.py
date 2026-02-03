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
import database  # Import the new database module

# --- Configuration ---
MODEL_PATH = r"C:\Users\aadit\Downloads\FP_Artefact2191136-1\FP_Artefact2191136\Bone_Fracture_Detection\models\best_bone_fracture_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Swaapped class names based on alphabetical order assumption (Fracture comes before Healthy)
# and user report of incorrect labeling.
CLASS_NAMES = ['Fracture', 'Healthy'] 

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Model Architecture (from Notebook) ---
class BoneFractureClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(BoneFractureClassifier, self).__init__()
        self.resnet = models.resnet50(weights=None)
        
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

# --- Grad-CAM Implementation ---
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
        output[0, target_class].backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=(1, 2))
        
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
    img_np = np.array(original_image_pil)
    height, width = img_np.shape[:2]
    
    cam_resized = cv2.resize(cam, (width, height))
    
    heatmap = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
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
        image_bytes = file.read()
        original_image, input_tensor = transform_image(image_bytes)
        
        cam, pred_class_idx, output = gradcam.generate_cam(input_tensor)
        
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0, pred_class_idx].item()
        prediction = CLASS_NAMES[pred_class_idx]
        
        overlay_image = apply_gradcam_overlay(original_image, cam)
        
        buffered = io.BytesIO()
        overlay_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        gradcam_image_base64 = f"data:image/jpeg;base64,{img_str}"
        
        # Helper to safely get probability for a class if it exists in CLASS_NAMES
        def get_prob(name):
            try:
                idx = CLASS_NAMES.index(name)
                return probabilities[0][idx].item()
            except ValueError:
                return 0.0

        prob_fracture = get_prob('Fracture')
        prob_healthy = get_prob('Healthy')

        # Check for patient info in form data (optional)
        patient_id = request.form.get('patient_id')
        if patient_id:
            try:
                database.save_analysis(
                    patient_id=int(patient_id),
                    image_filename=file.filename,
                    prediction=prediction,
                    confidence=confidence,
                    probabilities_healthy=prob_healthy,
                    probabilities_fracture=prob_fracture,
                    gradcam_image_path=gradcam_image_base64 # Storing base64 for now for simplicity, ideally should be file path
                )
            except Exception as db_err:
                print(f"Error saving to DB: {db_err}")

        return jsonify({
            "prediction": prediction,
            "confidence": f"{confidence * 100:.2f}%",
            "probabilities": {
                "Healthy": f"{prob_healthy * 100:.2f}%",
                "Fracture": f"{prob_fracture * 100:.2f}%"
            },
            "gradcam_image": gradcam_image_base64
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/patients', methods=['GET', 'POST'])
def manage_patients():
    if request.method == 'POST':
        data = request.json
        if not all(k in data for k in ('name', 'gender', 'age')):
             return jsonify({"error": "Missing required fields"}), 400
        
        try:
            pid = database.create_patient(data['name'], data['gender'], int(data['age']))
            return jsonify({"id": pid, "message": "Patient created successfully"}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        try:
            patients = database.get_all_patients()
            return jsonify(patients), 200
        except Exception as e:
             return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        history = database.get_analysis_history(limit=50)
        return jsonify(history), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        stats = database.get_analysis_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
