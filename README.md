# Bone Fracture Detection AI (with Grad-CAM)

A full-stack, investigational application for **bone fracture classification from X-ray images**.  
The system provides:
- A **Flask** inference API (Python)
- A **ResNet50-based classifier** (Fracture vs Healthy)
- **Grad-CAM** visual explanations returned to the client
- **SQLite** persistence for patients and analysis history
- A **React + Vite + TypeScript** frontend for clinical-style workflow (new analysis + patient history)

> **Important**: This software is provided for research/educational use only. It is **not** a medical device and must not be used for diagnosis or treatment decisions.

---

## Screenshots

### Pipeline overview
![Pipeline overview](image4)

### Classification examples (True vs Predicted)
![Classification examples](image3)

### Grad-CAM explanation (original → heatmap → overlay)
![Grad-CAM explanation](image2)

### YOLO detection examples (bounding boxes + confidence)
![YOLO detection examples](image1)

---

## Contents

- [Architecture](#architecture)
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

**Frontend (React/Vite)**  
- Patient intake → image upload → prediction + Grad-CAM display  
- History page for previous analyses

**Backend (Flask/Python)**  
- Loads a trained classification model
- Runs inference and generates **Grad-CAM**
- Stores patient and analysis metadata in **SQLite**

**Persistence (SQLite)**  
- `patients` table
- `analysis_history` table (linked to patients where applicable)

---

## Repository Structure

High-level layout:

- `server.py` — Flask application (model loading, inference, Grad-CAM, REST endpoints)
- `database.py` — SQLite initialization + data access functions
- `bone_fracture.db` — SQLite database file (used by the backend)
- `verify_backend.py` — backend smoke test (starts server and calls key endpoints)
- `train_yolo.py` — YOLO training script (separate from the classification API)
- `Fracture_Detection_Pipeline.ipynb`, `ResNet50cls.ipynb`, `Train_YOLO11.ipynb`, `Yolov11.ipynb` — notebooks/experiments
- `frontend/` — React (Vite + TypeScript) application

---

## Backend API

### Endpoints

Base URL (default): `http://127.0.0.1:5000`

- `GET /health`  
  Health check.

- `POST /api/analyze`  
  **Multipart form-data** upload:
  - `file` (required): X-ray image
  - `patient_id` (optional): if provided, analysis is stored for that patient

  Response includes:
  - `prediction` (`Fracture` or `Healthy`)
  - `confidence` (percentage string)
  - `probabilities` (per class)
  - `gradcam_image` (base64-encoded JPEG data URL)

- `GET /api/patients`  
  Returns all patients.

- `POST /api/patients`  
  Create a patient. JSON body:
  ```json
  { "name": "Jane Doe", "gender": "Female", "age": 32 }
  ```

- `GET /api/history`  
  Returns recent analysis history (includes patient fields when available).

- `GET /api/stats`  
  Basic counts: total analyses, fracture vs healthy, patient total.

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
This repository currently does **not** include a pinned dependency file (e.g., `requirements.txt`).  
Based on the imports used by the backend:

```bash
pip install flask flask-cors torch torchvision pillow numpy opencv-python requests
```

> If you are using GPU acceleration, ensure your installed PyTorch build matches your CUDA setup.

#### 3) Configure model path
`server.py` currently points to a machine-specific path like:

```python
MODEL_PATH = r"C:\Users\..."
```

Update this to your trained `.pth` model location (recommended: a **repo-relative** path such as `models/best_bone_fracture_model.pth`, or an environment variable).

#### 4) Start the backend
```bash
python server.py
```

---

### Frontend

#### 1) Install Node dependencies
```bash
cd frontend
npm install
```

#### 2) Run dev server
```bash
npm run dev
```

---

### Frontend ↔ Backend Proxy (Recommended)

The frontend uses relative API paths such as:
- `/api/patients`
- `/api/analyze`

For local development, configure a **Vite proxy** so the frontend can call the Flask backend without CORS or hardcoded URLs.

Create or update `frontend/vite.config.ts`:

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

Then restart `npm run dev`.

---

## Verification / Smoke Test

`verify_backend.py` performs a basic check of:
- `/health`
- `/api/stats`
- creates a test patient (`POST /api/patients`)
- lists patients (`GET /api/patients`)
- fetches history (`GET /api/history`)

Run:
```bash
python verify_backend.py
```

**Note:** `verify_backend.py` currently starts the server using a hardcoded path (`c:/Bone_Fracture/server.py`).  
You may need to update it to run the local `server.py` from this repository.

---

## Training & Experiments

This repository contains notebooks and a YOLO training script.

### YOLO training (optional)
`train_yolo.py` uses Ultralytics and expects a dataset YAML at a hardcoded path:
```python
DATA_YAML = r"c:\Bone_Fracture\dataset\data.yaml"
```

Update `DATA_YAML` to your dataset location.

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

- The application stores patient records and analysis history in **SQLite** (`bone_fracture.db`).
- If you plan to use real patient data:
  - remove/avoid storing identifying information,
  - implement access controls,
  - consider encryption and audit logging,
  - comply with applicable privacy regulations and institutional policies.

---

## Known Limitations

- **Hardcoded paths**: model paths and some scripts reference local Windows paths; update for portability.
- **Grad-CAM storage**: the system currently stores Grad-CAM output as a base64 data URL (convenient, but can significantly increase DB size). For production, store image files and persist a file path.
- **Dependency management**: no pinned dependency file is included yet; versions may vary by environment.

---

## License

No license is currently specified. Consider adding one (e.g., MIT/Apache-2.0) if you intend others to reuse or modify this project.
