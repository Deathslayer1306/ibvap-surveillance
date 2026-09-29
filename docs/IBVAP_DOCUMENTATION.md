# IBVAP: Intelligent Border Video Analytics Platform
## Technical Documentation & Architecture Guide

**Problem Statement:** SIH26187 (Smart India Hackathon 2026)  
**Organization:** Ministry of Home Affairs | Sashastra Seema Bal (SSB), Police II Division  
**Category:** Software | Blockchain & Cybersecurity  
**Team:** FLAME KAISER  
**Core Stack:** Python 3.11+ · FastAPI · ONNX Runtime (CUDA / DirectML / CPU) · OpenCV · YOLOv8 / YOLOv11 · EasyOCR  

---

## 1. Executive Summary & Problem Context

Conventional border surveillance relies heavily on continuous human monitoring across dozens or hundreds of CCTV video streams at Border Out Posts (BOPs), checkpoints, and perimeter fencing. This approach suffers from operator fatigue, missed anomalous events, lack of automated threat correlation, and prohibitive costs associated with specialized proprietary hardware.

**IBVAP (Intelligent Border Video Analytics Platform)** transforms existing standard CCTV and IP-based camera infrastructure into an autonomous, real-time threat detection and situational awareness network. Built with low-latency asynchronous inference pipelines, IBVAP executes 7 concurrent AI/ML detection modules on edge/GPU compute devices (achieving $\ge 30\text{ FPS}$ per stream), computes multi-signal composite threat scores, maintains persistent forensic logs, and dispatches real-time alerts via WebSockets, MJPEG streams, Prometheus metrics, and C2 (Command & Control) webhooks.

---

## 2. System Architecture

IBVAP decouples video ingestion from high-compute AI inference using a dedicated multi-threaded worker architecture per camera.

```
                                  ┌─────────────────────────────┐
                                  │        Video Sources        │
                                  │ (Webcam / Looped MP4 / RTSP)│
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
┌───────────────────────────┐     ┌─────────────────────────────┐
│       cameras.yaml        │────▶│      FastAPI Lifespan       │
│  (Config & Zone Matrix)   │     │  (Orchestration & Workers)  │
└───────────────────────────┘     └──────────────┬──────────────┘
                                                 │ spawns
                                                 ▼
                      ┌─────────────────────────────────────────────────────┐
                      │             CameraWorker (per stream)               │
                      │                                                     │
                      │  ┌──────────────────┐        ┌───────────────────┐  │
                      │  │  Capture Thread  │───────▶│  Inference Queue  │  │
                      │  │ (cv2.VideoCapture│ (drop) │  (Bounded FIFO)   │  │
                      │  └──────────────────┘        └─────────┬─────────┘  │
                      │                                        │            │
                      │                                        ▼            │
                      │       ┌──────────────────────────────────────────┐  │
                      │       │             Inference Thread             │  │
                      │       │ ┌──────────────────────────────────────┐ │  │
                      │       │ │ 0. NightPreprocessor (CLAHE)         │ │  │
                      │       │ │ 1. PoseDetector (YOLOv11-Pose)       │ │  │
                      │       │ │ 2. WeaponDetector (YOLOv11 ONNX)     │ │  │
                      │       │ │ 3. EmotionDetector (Face + FER2013)  │ │  │
                      │       │ │ 4. VehicleDetector (YOLOv8n COCO)    │ │  │
                      │       │ │ 5. ANPRDetector (ROI-Gated + EasyOCR)│ │  │
                      │       │ │ 6. FenceDetector (Polygon Math)      │ │  │
                      │       │ │ 7. ThreatAggregator (Score Engine)   │ │  │
                      │       │ └──────────────────────────────────────┘ │  │
                      │       └────────────────────┬─────────────────────┘  │
                      └────────────────────────────┼────────────────────────┘
                                                   │
             ┌─────────────────────────────────────┼──────────────────────────────────┐
             │                                     │                                  │
             ▼                                     ▼                                  ▼
┌───────────────────────────┐         ┌───────────────────────────┐      ┌──────────────────────────┐
│        EventLogger        │         │    Prometheus Metrics     │      │   WebSocket & MJPEG      │
│  (SQLite WAL + Snapshots) │         │ (Gauges, Counters, Lat.)  │      │ (Live Feed & Dashboard)  │
└────────────┬──────────────┘         └────────────┬──────────────┘      └────────────┬─────────────┘
             │                                     │                                  │
             ▼                                     ▼                                  ▼
┌───────────────────────────┐         ┌───────────────────────────┐      ┌──────────────────────────┐
│   C2 Webhook Dispatcher   │         │     Grafana Dashboard     │      │   Tactical Web UI / SPA  │
│  (HTTP POST on CRITICAL)  │         │   (sad-dashboard.json)    │      │  (HTML5 / CSS3 / JS)     │
└───────────────────────────┘         └───────────────────────────┘      └──────────────────────────┘
```

### Architecture Diagram
![IBVAP System Architecture](screenshots/01_system_architecture.jpg)

---

## 3. Threat Assessment & Data Flow

Threat assessment aggregates multi-modal signals into a continuous score normalized from **0.0 to 10.0**, mapped to tri-tier operational alert levels:

```
[Detection Signals]
├── Weapon Detection (Pistol: +5.0 / Knife: +3.5)
├── Pose Anomaly (Aggressive: +2.5 / Hands Raised: +2.0 / Running: +1.5)
├── Emotion Classification (Angry / Fear: +0.5 to +1.0)
├── Vehicle Classification (Car / Truck / Bus / Motorcycle)
├── ANPR Match (Watchlist Hit: +4.0 -> CRITICAL)
└── Geofence Breach (Warning: +1.0 / Dwell Exceeded: +2.0 -> CRITICAL)
                          │
                          ▼
            ┌───────────────────────────┐
            │     ThreatAggregator      │
            │ Composite Score: 0.0 - 10 │
            └─────────────┬─────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
 0.0 - 2.9 (INFO)  3.0 - 5.9 (WARNING)  6.0 - 10.0 (CRITICAL)
  [🟢 Normal]       [🟡 Event Logged]    [🔴 Snapshot + C2 Webhook]
```

### Threat Assessment Diagram
![Threat Assessment & Data Flow](screenshots/02_threat_assessment_data_flow.jpg)

---

## 4. Detection Pipeline & Module Specifications

| Stage | Module | Model / Technology | Frame Cadence | Primary Output | GPU Optimization |
|:---|:---|:---|:---|:---|:---|
| **0** | `NightPreprocessor` | CLAHE on LAB L-Channel | Every frame (if enabled) | Enhanced contrast frame | Zero ONNX overhead (CPU/OpenCV) |
| **1** | `PoseDetector` | `yolo11n-pose.onnx` (17 COCO keypoints) | Every 2nd frame | `PersonPose` list + Anomaly Label | Downscaled 640×360 inference |
| **2** | `WeaponDetector` | `weapon.onnx` (YOLOv11n) | Every 3rd frame | `WeaponResult` (BBoxes, pistol/knife) | Pure-NumPy per-class NMS |
| **3** | `EmotionDetector` | `yolov8n-face.onnx` + `emotion_better.onnx` | Every 5th frame | 7-Class Softmax Probabilities | Two-stage cascade on face crops |
| **4** | `VehicleDetector` | `yolov8n.onnx` (COCO-80 subset) | Every 3rd frame | `VehicleResult` (Class counts & boxes) | Filters bicycle, car, motorcycle, bus, truck |
| **5** | `ANPRDetector` | `anpr_plate_detect.onnx` + `EasyOCR` | Every 5th frame | Extracted text & Watchlist match | **ROI-Gated**: Runs only if vehicle detected |
| **6** | `FenceDetector` | `cv2.pointPolygonTest` | Every frame | `FenceResult` (Breaches & dwell times) | Pure mathematical geometry ($O(V)$) |
| **7** | `ThreatAggregator` | Heuristic weighted scoring | Every frame | `threat_score` (0–10) + `alert_level` | In-memory evaluation |

---

## 5. End-to-End Workflow

### 5.1 Startup Sequence
1. `python run.py` initializes the Uvicorn ASGI server with `app.main:app`.
2. FastAPI `lifespan` context manager loads configuration from `cameras.yaml`.
3. One `CameraWorker` instance is initialized per configured camera stream.
4. Each `CameraWorker` starts two daemon threads:
   - **Capture Thread**: Reads raw frames from source with auto-reconnection and drops stale frames when the queue reaches capacity (`maxsize=4`).
   - **Inference Thread**: Executes the 7-stage detection pipeline, annotates frames, writes logs, and updates shared `CameraState`.
5. Background event loop reference is captured for thread-safe WebSocket broadcasts.
6. The tactical dashboard is immediately accessible at `http://localhost:8000`.

### 5.2 Threading & Concurrency Model
- **Non-blocking Frame Ingestion**: Oldest frames are discarded if the inference worker is busy, preventing video lag and drift.
- **Thread-safe Event Persistence**: SQLite operates with Write-Ahead Logging (`PRAGMA journal_mode=WAL;`), enabling simultaneous non-blocking reads from API clients while the inference thread writes incident logs.
- **Asynchronous WebSocket Broadcasts**: Worker threads use `asyncio.run_coroutine_threadsafe()` to dispatch JSON telemetry frames to all connected dashboard clients every second.
- **Fire-and-Forget C2 Webhooks**: CRITICAL alerts trigger non-blocking HTTP POST requests to remote dispatch nodes using a background worker.

---

## 6. Threat Scoring Formula & Action Policy

Composite threat score $S \in [0.0, 10.0]$ is computed as:

$$S = \min\left(10.0, \sum w_i \cdot \mathbb{I}_i\right)$$

Where detection weights $w_i$ are defined as:

| Detection Signal | Trigger Condition | Score Addition ($w_i$) | Operational Action |
|:---|:---|:---:|:---|
| **Handgun / Pistol** | `pistol` detected in weapon labels | **+5.0** | Immediate CRITICAL alert escalation |
| **Blade / Knife** | `knife` detected in weapon labels | **+3.5** | WARNING level alert |
| **ANPR Watchlist Match** | Plate matches hot-reloaded `plates.json` | **+4.0** | Immediate CRITICAL alert escalation |
| **Geofence Dwell Breach** | Target dwell time in zone $>$ threshold | **+2.0** | Immediate CRITICAL alert escalation |
| **Geofence Zone Ingress** | Target inside perimeter zone polygon | **+1.0** | WARNING level alert |
| **Aggressive Stance Pose** | Stance geometry indicates combat pose | **+2.5** | Weighted addition |
| **Raised Hands Pose** | Wrists elevated above head plane | **+2.0** | Weighted addition |
| **Running / Crouching** | Rapid keypoint displacement or compression | **+1.5** | Weighted addition |
| **Hostile Emotion** | Top softmax class is `angry` or `fear` | **+0.5 – +1.0** | Weighted addition |
| **Vehicle Ingress** | Any vehicle classification | **+0.0** | Metadata context & ANPR trigger |

### Operational Alert Levels
- **INFO (0.0 – 2.9, 🟢)**: Normal operations. Telemetry streamed to WebSocket dashboard.
- **WARNING (3.0 – 5.9, 🟡)**: Suspicious activity detected. Incident entry logged to SQLite database and dashboard highlighted.
- **CRITICAL (6.0 – 10.0, 🔴)**: Severe security incident. High-resolution annotated JPEG saved to forensic storage, alert dispatched to remote C2 webhook, and audible/visual UI lockdown triggered.

---

## 7. REST API & WebSocket Specifications

### 7.1 Core Endpoints

| Method | Endpoint | Description | Query / Payload Parameters |
|:---|:---|:---|:---|
| `GET` | `/` | Web Tactical Dashboard | — |
| `GET` | `/video_feed` | MJPEG video stream | `?camera=CAM_01` |
| `GET` | `/ws/alerts` | Live WebSocket JSON telemetry stream | — |
| `GET` | `/api/status` | Current status across all active cameras | — |
| `GET` | `/api/cameras` | List configured cameras from `cameras.yaml` | — |
| `GET` | `/api/events` | Query SQLite incident history | `?camera=CAM_01&level=CRITICAL&limit=50` |
| `GET` | `/api/snapshots` | List forensic snapshot filenames | — |
| `GET` | `/snapshots/{filename}` | Retrieve specific snapshot image | — |
| `POST` | `/api/watchlist/plates` | Add license plate to ANPR watchlist | `{"plate": "MH12AB1234"}` |
| `GET` | `/api/watchlist/plates` | Retrieve all current watchlist entries | — |
| `GET` | `/metrics` | Prometheus exposition endpoint | — |

### 7.2 WebSocket Telemetry Payload Format
```json
{
  "camera_id": "CAM_01",
  "camera_name": "Perimeter North Gate",
  "alert_level": "CRITICAL",
  "threat_score": 8.5,
  "weapon_detected": true,
  "weapon_labels": ["pistol"],
  "emotion": "angry",
  "emotion_scores": {
    "angry": 0.74,
    "fear": 0.12,
    "neutral": 0.08,
    "disgust": 0.03,
    "sad": 0.02,
    "surprise": 0.01,
    "happy": 0.00
  },
  "pose_anomaly": "aggressive_stance",
  "num_persons": 2,
  "fence_breaches": [
    {
      "zone_id": "ZONE_ALPHA",
      "dwell_sec": 7.4,
      "is_critical": true
    }
  ],
  "anpr_plates": ["MH12AB1234"],
  "anpr_watchlist_hit": true,
  "vehicle_counts": {
    "car": 1,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0,
    "bicycle": 0
  },
  "inf_fps": 31.2,
  "cam_fps": 30.0,
  "timestamp": "2026-08-30T12:44:51Z"
}
```

---

## 8. AI Models & Runtime Specifications

| Model File | Network Architecture | Input Dimensions | Target Output | Execution Provider | Status |
|:---|:---|:---:|:---|:---:|:---:|
| `weapon.onnx` | YOLOv11n Custom | $480 \times 480 \times 3$ | Pistol, Knife BBoxes + Scores | CUDA / DirectML / CPU | ✅ Production Ready |
| `yolo11n-pose.onnx` | YOLOv11n-Pose | $640 \times 640 \times 3$ | 17 COCO Human Keypoints | CUDA / DirectML / CPU | ✅ Production Ready |
| `yolov8n-face.onnx` | YOLOv8n Face | $640 \times 640 \times 3$ | Face Region Bounding Boxes | CUDA / DirectML / CPU | ✅ Production Ready |
| `emotion_better.onnx` | EfficientNet-B0 | $64 \times 64 \times 3$ | 7 FER-2013 Class Logits | CUDA / DirectML / CPU | ✅ Production Ready |
| `yolov8n.onnx` | YOLOv8n COCO-80 | $640 \times 640 \times 3$ | Vehicle Class Filter (1, 2, 3, 5, 7) | CUDA / DirectML / CPU | ✅ Production Ready |
| `anpr_plate_detect.onnx` | YOLOv11n LP Fine-tune | $640 \times 640 \times 3$ | License Plate Bounding Boxes | CUDA / DirectML / CPU | ✅ Fine-tuning Tool Ready |
| `EasyOCR Engine` | CRAFT + CRNN | Dynamic Crop | Alphanumeric Character Sequences | PyTorch / CUDA | ✅ Auto-loaded |

---

## 9. Performance & Edge Optimization Strategies

1. **Interleaved Inference Cadences**: Non-critical detectors run on staggered frame intervals (`WEAPON_EVERY=3`, `POSE_EVERY=2`, `EMOTION_EVERY=5`, `VEHICLE_EVERY=3`, `ANPR_EVERY=5`), saving up to $70\%$ compute compared to full-frame processing.
2. **ROI-Gated ANPR**: The optical character recognition engine and plate detector run strictly when the vehicle detector flags a positive detection in the frame, saving ~40% unnecessary inference cycles.
3. **Scaled Resolution Processing**: Inference runs on downscaled $640 \times 360$ frames and scales bounding coordinates back to native display dimensions, yielding a $2.5\times$ throughput speedup.
4. **Graph-Optimized ONNX Runtime**: Sessions are initialized with `ORT_ENABLE_ALL` graph optimization level and tuned inter/intra-op thread pools.
5. **Zero-Copy Memory Transfers**: NumPy slicing and buffer re-use minimize GC pressure during MJPEG frame streaming.
