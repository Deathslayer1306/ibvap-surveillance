# Real-Time AI Surveillance System

## Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd mini_project
```

---

### 2. Create a Virtual Environment
```bash
python -m venv .venv
```

Activate it:

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```powershell
.venv\Scripts\activate
```

---

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Create Required Directories
```bash
mkdir -p data/watchlist data/faces data/databases templates models
```

---

### 5. Download InsightFace Model
```bash
wget https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx -O models/insightface.onnx
```

If `wget` fails:
```bash
curl -L https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx -o models/insightface.onnx
```

---

### 6. Verify Camera
```bash
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

---

### 7. Run the Application
```bash
python -m app.main
```

---

### 8. Open in Browser
```
http://localhost:5000
```

---

### Cleanup Generated Data
```bash
python scripts/cleanup.py
```
