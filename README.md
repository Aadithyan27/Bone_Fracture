# Bone Fracture Detection AI (ResNet50 + YOLOv11 + Grad-CAM)

A full-stack, investigational application for **automated bone fracture detection from X-ray images**.  
The system follows a **two-stage decision pipeline**:

1. **Preprocess** the input X-ray (CLAHE + normalization + resizing)
2. **Classify** with **ResNet50** (Healthy vs Fracture)
3. If **Healthy** → return *“No fracture detected”*
4. If **Fracture** → run **YOLOv11** to localize the fracture with **bounding boxes**
5. Optionally generate **Grad-CAM / heatmap overlay** for explainability
6. Return the **final report + annotated image**

The project includes:
- **Flask** backend API (Python) for inference and persistence
- **ResNet50 classifier** (Healthy vs Fracture)
- **YOLOv11 detector** (fracture localization when fracture is predicted)
- Optional **Grad-CAM** explanation overlays
- **SQLite** database for patient metadata and analysis history
- **React + Vite + TypeScript** frontend for a clinical-style workflow

> **Important**: This software is provided for research/educational use only. It is **not** a medical device and must not be used for diagnosis or treatment decisions.

---

## Screenshots

### Pipeline overview (system logic)
![Pipeline overview](./Screenshot%202026-03-08%20135008.png)

### Classification examples (True vs Predicted)
![Classification examples](./Screenshot%202026-03-08%20135104.png)

### Grad-CAM explanation (original → heatmap → overlay)
![Grad-CAM explanation](./Screenshot%202026-03-08%20135121.png)

### YOLO detection examples (bounding boxes + confidence)
![YOLO detection examples](./Screenshot%202026-03-08%20135145.png)


## Contents

- [Architecture](#architecture)
- [Pipeline (How the Model Works)](#pipeline-how-the-model-works)
- [Repository Structure](#repository-structure)
- [Backend API](#backend-api)
  - [Endpoints](#endpoints)
- [Local Setup](#local-setup)
  - [Backend](#backend)
  - [Frontend](#frontend)
  - [Frontend ↔ Backend Proxy (Recommended)](#frontend--backend-proxy-recommended)
- [Verification / Smoke Test](#verification--smoke-test)
- [Training & Experiments](#training--experiments)
- [Data & Privacy](#data--privacy)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Architecture

### Frontend (React / Vite / TypeScript)
- Patient intake (name, gender, age)
- Upload X-ray image for analysis
- View result (prediction + confidence + Grad-CAM image when available)
- Browse analysis history

### Backend (Flask / Python)
- Loads the trained model(s)
- Runs inference and returns:
  - classification (Healthy/Fracture)
  - confidence / probabilities
  - Grad-CAM overlay image (when enabled)
  - history/stats endpoints
- Stores patient records and analysis history in SQLite

### Persistence (SQLite)
- `patients` table
- `analysis_history` table (analysis + optional patient association)

---

## Pipeline (How the Model Works)

The system implements the following workflow (see the pipeline screenshot above):

1. **Input**: X-ray image upload
2. **Preprocessing**:
   - CLAHE (contrast enhancement)
   - normalization
   - resize to model input size
3. **Stage 1 — Classification (ResNet50)**:
   - output: `Healthy` or `Fracture`
4. **Stage 2 — Localization (YOLOv11)** *(only if fracture predicted)*:
   - output: bounding boxes with confidence scores on the image
5. **Explainability (Optional)**:
   - Grad-CAM / heatmap overlay for interpretability
6. **Final Output**:
   - report + annotated image

> Note: In the current repository, the Flask API (`server.py`) clearly implements **ResNet50 classification + Grad-CAM generation**. YOLOv11 training and weights exist in the repository (e.g., `train_yolo.py`, `.pt` weights), but integrating YOLO inference into the same `/api/analyze` endpoint may require an additional backend step if not already wired.

---

## Repository Structure

High-level layout:

- `server.py` — Flask API (classification + Grad-CAM + endpoints)
- `database.py` — SQLite initialization + CRUD (patients + analysis history)
- `bone_fracture.db` — SQLite database (created/used by backend)
- `verify_backend.py` — backend smoke test (calls key endpoints)
- `train_yolo.py` — YOLO training script (Ultralytics)
- `*.ipynb` — notebooks for experiments/training
- `frontend/` — React + Vite + TS frontend

---

## Backend API

### Endpoints

Base URL (default): `http://127.0.0.1:5000`

- `GET /health`  
  Health check.

- `POST /api/analyze`  
  Upload an X-ray image (**multipart form-data**):
  - `file` (required)
  - `patient_id` (optional, form field)

  Response includes:
  - `prediction` (`Fracture` or `Healthy`)
  - `confidence` (percentage string)
  - `probabilities` (per class)
  - `gradcam_image` (base64 JPEG data URL)

- `GET /api/patients`  
  List patients.

- `POST /api/patients`  
  Create a patient. JSON body:
  ```json
  { "name": "Jane Doe", "gender": "Female", "age": 32 }
  ```

- `GET /api/history`  
  Recent analysis history (joined with patient info when present).

- `GET /api/stats`  
  Aggregate counts (analyses + patient totals).

---

## Local Setup

### Backend

#### 1) Create & activate a virtual environment
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate
```

#### 2) Install dependencies
This repo does not currently include a pinned `requirements.txt`.  
Common backend dependencies (based on imports) include:

```bash
pip install flask flask-cors torch torchvision pillow numpy opencv-python requests
```

> If using GPU acceleration, ensure your PyTorch installation matches your CUDA version.

#### 3) Configure the model path
`server.py` currently points to a machine-specific path such as:
```python
MODEL_PATH = r"C:\Users\..."
```

Update it to your `.pth` file location (recommended: use a repo-relative path like `models/best_bone_fracture_model.pth`).

#### 4) Start the backend
```bash
python server.py
```

---

### Frontend

#### 1) Install dependencies
```bash
cd frontend
npm install
```

#### 2) Start dev server
```bash
npm run dev
```

---

### Frontend ↔ Backend Proxy (Recommended)

The frontend uses relative URLs such as:
- `/api/patients`
- `/api/analyze`

Configure a Vite proxy so `/api` routes forward to Flask.

Example `frontend/vite.config.ts`:

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:5000',
      '/health': 'http://127.0.0.1:5000'
    }
  }
})
```

Restart `npm run dev` after changes.

---

## Verification / Smoke Test

`verify_backend.py` tests:
- `/health`
- `/api/stats`
- create patient + list patients
- `/api/history`

Run:
```bash
python verify_backend.py
```

> Note: `verify_backend.py` currently starts the server using a hardcoded path (`c:/Bone_Fracture/server.py`). You may need to update it to match your local repo path.

---

## Training & Experiments

### YOLO training (optional)
`train_yolo.py` uses Ultralytics YOLO and expects a dataset YAML at a hardcoded path:
```python
DATA_YAML = r"c:\Bone_Fracture\dataset\data.yaml"
```

Update it to your dataset location.

Install:
```bash
pip install ultralytics
```

Run:
```bash
python train_yolo.py
```

### Notebooks
- `Fracture_Detection_Pipeline.ipynb`
- `ResNet50cls.ipynb`
- `Train_YOLO11.ipynb`
- `Yolov11.ipynb`

---

## Data & Privacy

- The app stores patient metadata and analysis history in **SQLite** (`bone_fracture.db`).
- For real patient data, implement appropriate privacy and security controls.

---

## Known Limitations

- **Hardcoded paths** exist in `server.py` and training scripts; update for portability.
- **Grad-CAM image storage** is currently stored as base64 (convenient, but can increase DB size). For production, store files and persist paths.
- **YOLO inference integration**: YOLO training assets exist; ensure YOLO inference is wired into the backend if you want the API to return bounding-box annotated outputs.
