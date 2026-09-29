<div align="center">

# 🛡️ IBVAP — Intelligent Border Video Analytics Platform

### Real-Time Edge AI Multi-Camera Surveillance & Threat Assessment System
**Smart India Hackathon 2026 — Problem Statement ID: SIH26187**  
*Ministry of Home Affairs (MHA) · Sashastra Seema Bal (SSB), Police II Division*

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-GPU%20%2F%20DirectML-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![YOLOv11](https://img.shields.io/badge/YOLOv8%20%2F%20v11-Ultralytics-00FFFF.svg?logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7%2B-orange.svg)](https://github.com/JaidedAI/EasyOCR)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics%20Enabled-E6522C.svg?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Team](https://img.shields.io/badge/Team-FLAME%20KAISER-red.svg)](#team--credits)

[System Architecture](#-system-architecture) • [Features](#-key-capabilities) • [Screenshots](#-tactical-dashboard-showcase) • [Quick Start](#-quick-start--installation) • [API Reference](#-rest-api--websocket-reference) • [Documentation](docs/IBVAP_DOCUMENTATION.md)

</div>

---

## 📌 Executive Summary

Conventional border surveillance at **Border Out Posts (BOPs)**, check posts, and strategic roads depends heavily on human vigilance across dozens of CCTV monitors. This causes operator fatigue, delayed reaction times, and missed anomalous activities.

**IBVAP (Intelligent Border Video Analytics Platform)** upgrades existing standard IP/RTSP camera and CCTV infrastructure into an autonomous, real-time tactical surveillance network. Running on edge/GPU workstations ($\ge 30\text{ FPS}$ per stream), IBVAP combines **7 concurrent AI/ML detection modules**, computes a real-time composite threat score ($0.0 - 10.0$), maintains persistent SQLite forensic logs, captures incident snapshots, and broadcasts telemetry to an interactive web dashboard and remote **Command & Control (C2)** centers.

---

## 🏗️ System Architecture

IBVAP decouples video ingestion from high-throughput AI inference using a dedicated multi-threaded worker model per camera stream.

<div align="center">
  <img src="screenshots/01_system_architecture.jpg" alt="IBVAP System Architecture" width="95%"/>
</div>

### Architectural Highlights
- **Dedicated Worker Pair**: Each camera runs an independent **Capture Thread** (with auto-reconnect & FPS limiter) and an **Inference Thread** connected via a bounded FIFO queue (`maxsize=4`).
- **Zero-Lag Queue Strategy**: Oldest frames are automatically dropped if the inference engine is busy, ensuring zero stream delay.
- **ROI-Gated Execution**: High-cost detectors (e.g. EasyOCR for ANPR) run strictly when vehicle presence is flagged, eliminating wasted GPU cycles on empty scenes.
- **Asynchronous Telemetry Dispatch**: WebSocket broadcasts and Prometheus metric updates run non-blockingly at 1 Hz.
- **WAL-Mode Persistence**: SQLite Write-Ahead Logging allows concurrent dashboard queries without locking inference writes.

---

## ⚡ Threat Assessment & Multi-Signal Scoring

IBVAP aggregates diverse visual and geometrical signals into a normalized continuous threat score from **0.0 to 10.0**, triggering proportional operational responses:

<div align="center">
  <img src="screenshots/02_threat_assessment_data_flow.jpg" alt="IBVAP Threat Assessment Flow" width="95%"/>
</div>

### Multi-Signal Threat Matrix

| Detection Signal | Trigger Condition | Points Added ($w_i$) | Operational Action |
|:---|:---|:---:|:---|
| **🔫 Weapon (Pistol)** | `pistol` detected by YOLOv11 ONNX | **+5.0** | **CRITICAL Alert Escalation** |
| **🔪 Weapon (Knife)** | `knife` detected by YOLOv11 ONNX | **+3.5** | **WARNING Alert** |
| **🚗 ANPR Watchlist Match** | Plate matches hot-reloaded `plates.json` | **+4.0** | **CRITICAL Alert Escalation** |
| **🚧 Geofence Dwell Breach** | Person in restricted zone $>$ threshold seconds | **+2.0** | **CRITICAL Alert Escalation** |
| **⚠️ Geofence Zone Ingress** | Person inside virtual polygon zone | **+1.0** | **WARNING Alert** |
| **🥊 Aggressive Stance Pose** | Body geometry indicates combat/fighting pose | **+2.5** | Multi-signal additive |
| **🙌 Raised Hands Pose** | Wrists elevated above head plane | **+2.0** | Multi-signal additive |
| **🏃 Running / Crouching** | Rapid keypoint displacement / compression | **+1.5** | Multi-signal additive |
| **😡 Hostile Facial Emotion** | Top facial classification is `angry` or `fear` | **+0.5 – +1.0** | Multi-signal additive |
| **🚙 Vehicle Classification** | Car, Truck, Bus, Motorcycle, Bicycle | **+0.0** | Metadata context & ANPR trigger |

### Operational Alert Tiers
- 🟢 **INFO (0.0 – 2.9)**: Normal routine monitoring. Live telemetry streamed to WebSocket dashboard.
- 🟡 **WARNING (3.0 – 5.9)**: Suspicious activity detected. Incident logged in SQLite database and UI indicators lit.
- 🔴 **CRITICAL (6.0 – 10.0)**: Severe breach or threat. High-resolution snapshot stored in `data/snapshots/`, C2 webhook fired, and audio-visual lockdown triggered.

---

## 🌟 Key Capabilities & Detection Modules

| Module | Engine / Model | Resolution & Cadence | Primary Functionality |
|:---|:---|:---|:---|
| **1. Weapon Detection** | `weapon.onnx` (YOLOv11n) | $480\times 480$ (Every 3rd frame) | Real-time detection of handguns (pistols) and edged weapons (knives) with NumPy NMS. |
| **2. Pose Anomaly** | `yolo11n-pose.onnx` | $640\times 640$ (Every 2nd frame) | 17 COCO keypoint tracking; detects raised hands, fighting stances, crouching, and running. |
| **3. Emotion Analysis** | `yolov8n-face.onnx` + `emotion_better.onnx` | Two-Stage (Every 5th frame) | Localizes faces and performs 7-class FER-2013 emotion classification (`angry`, `fear`, `neutral`, etc.). |
| **4. Vehicle Classifier** | `yolov8n.onnx` (COCO-80) | $640\times 640$ (Every 3rd frame) | Real-time categorization of cars, trucks, buses, motorcycles, and bicycles. |
| **5. ANPR & Watchlist** | `anpr_plate_detect.onnx` + EasyOCR | Dynamic Crop (Every 5th frame) | High-accuracy OCR on license plates with instant $O(1)$ JSON watchlist lookup. |
| **6. Geofencing / Virtual Fence** | `cv2.pointPolygonTest` | Pure Math (Every frame) | Multi-zone arbitrary polygon containment testing with per-track dwell-time counters. |
| **7. Night Preprocessor** | CLAHE (LAB Color Space) | OpenCV (Every frame) | Enhances low-light contrast and perimeter visibility without blowing out highlights. |

---

## 🖥️ Tactical Dashboard Showcase

The IBVAP web interface is a responsive Single Page Application (SPA) providing comprehensive surveillance telemetry, live camera streams, and incident management.

### 1. Threat Intelligence Center
Comprehensive visual breakdown of active detections, 5-minute bucket timelines, and per-camera operational health.
![Threat Intelligence Dashboard](screenshots/05_threat_intelligence_dashboard.png)

### 2. ANPR Watchlist Manager & Live Tracker
Real-time license plate detection, OCR extraction, vehicle type correlation, and hot-reloadable watchlist management.
![ANPR Watchlist Manager](screenshots/03_anpr_watchlist_manager.png)

### 3. Geofence & Virtual Zone Polygon Editor
Interactive perimeter zone management with real-time dwell-time counters, breach alarms, and camera zone toggles.
![Geofence Zone Editor](screenshots/06_geofence_zone_editor.png)

### 4. Forensic Evidence Vault & Incident Archive
Filterable incident database with severity grading (`CRITICAL`, `WARNING`, `INFO`), timestamping, GPS/location tags, and captured evidence snapshots.
![Forensic Evidence Vault](screenshots/04_forensic_evidence_vault.png)

### 5. SIH 2026 Problem Statement & Team Submission
Official Smart India Hackathon 2026 submission details for Problem Statement ID **SIH26187** (Team: **FLAME KAISER**).
![SIH Problem Statement](screenshots/08_sih_problem_statement_26187.png)
![FLAME KAISER Submission](screenshots/09_flame_kaiser_sih_submission.png)

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- **Python**: 3.11 or higher
- **GPU Acceleration** (Recommended): NVIDIA GPU with CUDA 11.8+ or 12.x / DirectML
- **Operating System**: Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS

### 2. Clone the Repository
```bash
git clone https://github.com/Deathslayer1306/ibvap-surveillance.git
cd ibvap-surveillance
```

### 3. Setup Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Verify Models & Configuration
Ensure ONNX model files are placed in `models/`:
- `models/weapon.onnx`
- `models/yolo11n-pose.onnx`
- `models/yolov8n-face.onnx`
- `models/emotion_better.onnx`
- `models/yolov8n.onnx`

### 6. Configure Camera Feeds (`cameras.yaml`)
Edit `cameras.yaml` to specify your input video sources:
```yaml
cameras:
  - id: "CAM_01"
    name: "Perimeter Main Gate"
    source: 0                                   # 0 for USB webcam
    # source: "data/test_videos/sample.mp4"     # Local MP4 file
    # source: "rtsp://admin:pass@192.168.1.10"  # RTSP IP Camera stream
    loop: true
    night_mode: false
    fps_limit: 30
    zones:
      - id: "ZONE_A"
        name: "Perimeter North"
        polygon: [[0.05, 0.10], [0.60, 0.10], [0.60, 0.90], [0.05, 0.90]]
        dwell_threshold_sec: 3.0
```

### 7. Launch the Platform
```bash
python run.py
```
Open your browser and navigate to: **`http://localhost:8000`**

---

## 📡 REST API & WebSocket Reference

### HTTP Endpoints
| Method | Route | Description | Query / Payload Example |
|:---|:---|:---|:---|
| `GET` | `/` | Web Tactical Dashboard | — |
| `GET` | `/video_feed` | MJPEG video stream | `?camera=CAM_01` |
| `GET` | `/ws/alerts` | Live WebSocket JSON telemetry | — |
| `GET` | `/api/status` | Current status across all active cameras | — |
| `GET` | `/api/cameras` | List configured cameras | — |
| `GET` | `/api/events` | Query SQLite incident history | `?camera=CAM_01&level=CRITICAL&limit=50` |
| `GET` | `/api/snapshots` | List stored snapshot filenames | — |
| `GET` | `/snapshots/{file}`| Retrieve forensic snapshot JPEG | `/snapshots/CRITICAL_2026-08-30_12-44.jpg` |
| `POST`| `/api/watchlist/plates` | Add license plate to ANPR watchlist | `{"plate": "MH12AB1234"}` |
| `GET` | `/api/watchlist/plates` | List all active watchlist plates | — |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint | — |

### Sample WebSocket Telemetry Frame (`/ws/alerts`)
```json
{
  "camera_id": "CAM_01",
  "camera_name": "Perimeter Main Gate",
  "alert_level": "CRITICAL",
  "threat_score": 8.5,
  "weapon_detected": true,
  "weapon_labels": ["pistol"],
  "emotion": "angry",
  "emotion_scores": { "angry": 0.74, "fear": 0.12, "neutral": 0.08, "happy": 0.00 },
  "pose_anomaly": "aggressive_stance",
  "num_persons": 2,
  "fence_breaches": [
    { "zone_id": "ZONE_A", "dwell_sec": 4.8, "is_critical": true }
  ],
  "anpr_plates": ["MH12AB1234"],
  "anpr_watchlist_hit": true,
  "vehicle_counts": { "car": 1, "truck": 0, "bus": 0, "motorcycle": 0 },
  "inf_fps": 32.4,
  "cam_fps": 30.0,
  "timestamp": "2026-08-30T12:44:51Z"
}
```

---

## 🛠️ Developer Utilities

### 1. Interactive Zone Polygon Editor
Draw and test custom geofence polygons interactively on live camera feeds:
```bash
python tools/draw_zones.py --camera CAM_01
```
*Left-click to place polygon points, right-click to close the polygon, and press `S` to save directly to `cameras.yaml`.*

### 2. Fine-Tune Indian License Plate ANPR Model
```bash
python tools/train_anpr.py --epochs 50 --imgsz 640
```
*Fine-tunes a YOLOv11 detector on Indian LP datasets and exports `models/anpr_plate_detect.onnx`.*

---

## 📊 Prometheus & Grafana Integration

IBVAP exposes **15+ real-time Prometheus metrics** at `GET /metrics`:
- `sad_threat_score`: Real-time composite threat score ($0.0 - 10.0$).
- `sad_alert_level`: Current state ($0 = \text{INFO}, 1 = \text{WARNING}, 2 = \text{CRITICAL}$).
- `sad_weapon_detected`: Binary gauge for active weapon detection.
- `sad_camera_fps` & `sad_inference_fps`: Frame throughput telemetry.
- `sad_inference_loop_latency_seconds`: End-to-end model inference histogram.

Import the pre-configured Grafana dashboard from [`grafana/sad-dashboard.json`](grafana/sad-dashboard.json).

---

## 📁 Repository Structure

```
Suspicious-Activity-Detection/
├── app/
│   ├── config.py                 # Central configurations, cadences, & model paths
│   ├── main.py                   # FastAPI application, CameraWorkers, & REST/WS routes
│   ├── metrics.py                # Prometheus telemetry instrumentation
│   ├── event_logger.py           # SQLite WAL persistence & snapshot manager
│   └── detectors/
│       ├── weapon_detector.py    # YOLOv11 ONNX weapon detection (pistol/knife)
│       ├── pose_detector.py      # YOLOv11-Pose anomaly keypoint classification
│       ├── emotion_detector.py   # Two-stage face detection + FER-2013 classification
│       ├── vehicle_detector.py   # YOLOv8n COCO vehicle categorization
│       ├── anpr_detector.py      # License plate localization + EasyOCR + watchlist
│       ├── fence_detector.py     # Mathematical polygon containment & dwell timers
│       ├── night_preprocessor.py # CLAHE low-light enhancement
│       └── threat_aggregator.py  # Multi-signal weighted scoring engine
├── data/
│   ├── databases/events.db       # SQLite persistent event store
│   ├── snapshots/                # CRITICAL alert annotated JPEG frames
│   ├── test_videos/              # Sample MP4 testing videos
│   └── watchlist/plates.json     # Hot-reloadable ANPR plate registry
├── docs/
│   └── IBVAP_DOCUMENTATION.md    # Full technical specifications & architecture guide
├── grafana/
│   └── sad-dashboard.json        # Grafana observability dashboard definition
├── models/                       # ONNX model directory
├── screenshots/                  # High-resolution tactical dashboard screenshots
├── static/                       # CSS, JavaScript UI assets & iconography
├── templates/                    # Tactical UI pages (Live Matrix, ANPR, Logs, Intel)
├── tools/                        # ANPR training & zone configuration utilities
├── cameras.yaml                  # Unified camera matrix & system configuration
├── requirements.txt              # Production Python package dependencies
└── run.py                        # Main platform entry point
```

---

## 👥 Team & Credits

**Team FLAME KAISER** — *Smart India Hackathon 2026*  
- **Mihir Naik** *(Team Leader)* — [GitHub](https://github.com/Deathslayer1306)  
- **Mayoogh Manoj**
- **Ayush Padalkar**
- **Ronit Sinkar**
- **Mrunmayee Tamse**
- **Mentor:** Dr. Pranali Choudhari

---

## 📄 License & Attribution

This project is developed for the **Smart India Hackathon 2026** under the **Ministry of Home Affairs (MHA)** problem statement.  
Licensed under the [MIT License](LICENSE).
