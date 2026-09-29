"""
IBVAP — Intelligent Border Video Analytics Platform
FastAPI Backend  (full rewrite of main.py)

Architecture
============
cameras.yaml ──► lifespan()
                   └─ one CameraWorker thread per camera entry
                         ├─ cv2.VideoCapture + loop/fps_limit
                         ├─ NightPreprocessor (if enabled)
                         ├─ Detector pipeline with per-detector frame-skip
                         │     night_preprocessor → pose → weapon → emotion
                         │     → face/FRS → vehicle → anpr (ROI-gated) → fence
                         ├─ ThreatAggregator (extended with new signals)
                         ├─ WS broadcast  (camera_id tagged)
                         └─ EventLogger  (WARNING/CRITICAL → SQLite + snapshot)

Endpoints preserved
-------------------
  GET  /                   → index.html dashboard
  GET  /video_feed         → MJPEG stream (?camera=CAM_01)
  GET  /ws/alerts          → WebSocket broadcast
  GET  /api/status         → per-camera last known state
  GET  /metrics            → Prometheus

New endpoints
-------------
  GET  /api/cameras
  GET  /api/events
  GET  /api/snapshots
  GET  /snapshots/{filename}
  POST /api/watchlist/plates
  GET  /api/watchlist/plates
"""
from __future__ import annotations

# ── Bootstrap: make `from app...` imports work from any CWD ─────────────────
import sys
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import asyncio
import json
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set

# Suppress FFmpeg/OpenCV verbose codec warnings (H.264 header damaged spam)
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # AV_LOG_QUIET
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Existing detectors ───────────────────────────────────────────────────────
from app.detectors.weapon_detector   import WeaponDetector,  WeaponResult
from app.detectors.emotion_detector  import EmotionDetector, EmotionResult
from app.detectors.pose_detector     import PoseDetector,    PersonPose, PoseAnomaly
from app.detectors.threat_aggregator import ThreatAggregator

# ── New detectors ────────────────────────────────────────────────────────────
from app.detectors.anpr_detector    import ANPRDetector,    ANPRResult
from app.detectors.vehicle_detector import VehicleDetector, VehicleResult
from app.detectors.fence_detector   import FenceDetector,   FenceResult
from app.detectors.night_preprocessor import NightPreprocessor

# ── Event logger ─────────────────────────────────────────────────────────────
from app.event_logger import EventLogger

# ── Prometheus metrics (existing) ────────────────────────────────────────────
from app.metrics import (
    REQUESTS_TOTAL,
    REQUEST_LATENCY_SECONDS,
    update_threat_metrics,
    format_metrics,
    set_camera_fps,
    set_inference_fps,
    observe_inference_loop_latency_seconds,
    CONTENT_TYPE_LATEST,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ibvap")


# ═══════════════════════════════════════════════════════════════════════════════
# ── YAML CONFIG LOADER ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _load_cameras_yaml() -> dict:
    """Load cameras.yaml from project root. Returns parsed dict."""
    yaml_path = _PROJECT_ROOT / "cameras.yaml"
    if not yaml_path.exists():
        logger.error(f"cameras.yaml not found at {yaml_path}. Using defaults.")
        return {"cameras": [], "settings": {}}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


_CFG: dict = _load_cameras_yaml()
_CAMERAS_CFG: list[dict] = _CFG.get("cameras", [])
_SETTINGS: dict = _CFG.get("settings", {})

# ── Derived paths from settings ──────────────────────────────────────────────
_LOG_CFG       = _SETTINGS.get("logging", {})
_DB_PATH       = _LOG_CFG.get("db_path", "data/databases/events.db")
_SNAPSHOTS_DIR = _LOG_CFG.get("snapshots_dir", "data/snapshots/")
_MAX_SNAPSHOTS = int(_LOG_CFG.get("max_snapshots", 500))

_ANPR_CFG      = _SETTINGS.get("anpr", {})
_ANPR_ENABLED  = bool(_ANPR_CFG.get("enabled", True))
_ANPR_MODEL    = _ANPR_CFG.get("model", "models/anpr_plate_detect.onnx")
_ANPR_OCR      = _ANPR_CFG.get("ocr_engine", "easyocr")
_ANPR_WATCHLIST= _ANPR_CFG.get("watchlist_file", "data/watchlist/plates.json")
_ANPR_CONF     = float(_ANPR_CFG.get("confidence_threshold", 0.5))

_NIGHT_CFG     = _SETTINGS.get("night_mode", {})
_CLAHE_CLIP    = float(_NIGHT_CFG.get("clip_limit", 2.0))
_CLAHE_TILE    = tuple(_NIGHT_CFG.get("tile_grid", [8, 8]))

_C2_CFG        = _SETTINGS.get("c2", {})
_C2_URL        = _C2_CFG.get("webhook_url", "") or ""

# ── GPU providers (same as existing config.py logic) ─────────────────────────
def _build_providers() -> list:
    try:
        import onnxruntime as ort
        from app.config import GPU_DEVICE_ID, CUDA_MEM_LIMIT
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return [
                ("CUDAExecutionProvider", {
                    "device_id": GPU_DEVICE_ID,
                    "gpu_mem_limit": CUDA_MEM_LIMIT,
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                }),
                "CPUExecutionProvider",
            ]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


_ONNX_PROVIDERS = _build_providers()

# ── Ensure required directories exist ────────────────────────────────────────
for _d in ["data/databases", "data/snapshots", "data/watchlist", "data/faces", "data/test_videos"]:
    os.makedirs(_d, exist_ok=True)

# ── Global event logger (shared across all cameras) ──────────────────────────
_event_logger = EventLogger(
    db_path=_DB_PATH,
    snapshots_dir=_SNAPSHOTS_DIR,
    max_snapshots=_MAX_SNAPSHOTS,
)

# ── Server constants ─────────────────────────────────────────────────────────
try:
    from app.config import HOST, PORT, JPEG_QUALITY
    from app.config import POSE_EVERY, WEAPON_EVERY, EMOTION_EVERY
    from app.config import STREAM_WIDTH, STREAM_HEIGHT, STREAM_QUALITY
except Exception:
    HOST, PORT, JPEG_QUALITY = "0.0.0.0", 8000, 85
    POSE_EVERY, WEAPON_EVERY, EMOTION_EVERY = 4, 6, 10
    STREAM_WIDTH, STREAM_HEIGHT, STREAM_QUALITY = 960, 540, 60

# OPT-05: raised frame-skip intervals for new IBVAP detectors
ANPR_EVERY    = 8   # every 8th frame  (was 5)
VEHICLE_EVERY = 5   # every 5th frame  (was 3)
FACE_EVERY    = 6   # every 6th frame  (was 4)


# ═══════════════════════════════════════════════════════════════════════════════
# ── EXTENDED THREAT AGGREGATOR ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def compute_threat_score(
    weapon_detected: bool,
    weapon_labels: List[str],
    pose_anomaly: str,
    emotion: str,
    fence_result: Optional[FenceResult] = None,
    anpr_result: Optional[ANPRResult] = None,
    vehicle_result: Optional[VehicleResult] = None,
    night_mode_active: bool = False,
    num_persons: int = 0,
) -> tuple[float, str]:
    """
    Compute unified threat score (0-10) from all detector signals.

    Score contributions (additive, capped at 10):
      weapon_detected      : +4.0 per weapon
      pose_aggressive      : +2.5
      pose_raised_hands    : +1.5
      fence_breach         : +3.0
      fence_breach_critical: +1.0 (bonus dwell bonus)
      emotion_angry/fear   : +1.0
      anpr_watchlist_hit   : +3.5
      night_movement       : +1.0 (person detected in night mode)
      vehicle_large        : +0.5 (truck/bus near perimeter)

    Alert levels: 0-3=INFO, 4-6=WARNING, 7-10=CRITICAL
    """
    score = 0.0

    # ── Weapon ────────────────────────────────────────────────────────────────
    if weapon_detected:
        score += 4.0 * max(1, len(weapon_labels))

    # ── Pose ──────────────────────────────────────────────────────────────────
    if pose_anomaly == PoseAnomaly.AGGRESSIVE.value:
        score += 2.5
    elif pose_anomaly == PoseAnomaly.RAISED_HANDS.value:
        score += 1.5

    # ── Emotion ───────────────────────────────────────────────────────────────
    if emotion in ("angry", "fear"):
        score += 1.0

    # ── Fence breaches ────────────────────────────────────────────────────────
    if fence_result and fence_result.has_breach:
        score += 3.0
        if fence_result.has_critical:
            score += 1.0

    # ── ANPR watchlist ────────────────────────────────────────────────────────
    if anpr_result and anpr_result.has_watchlist_hit:
        score += 3.5

    # ── Night movement ────────────────────────────────────────────────────────
    if night_mode_active and num_persons > 0:
        score += 1.0

    # ── Large vehicle ─────────────────────────────────────────────────────────
    if vehicle_result and vehicle_result.has_large_vehicle:
        score += 0.5

    score = min(score, 10.0)

    if score >= 7.0:
        level = "CRITICAL"
    elif score >= 4.0:
        level = "WARNING"
    else:
        level = "INFO"

    return round(score, 2), level


# ═══════════════════════════════════════════════════════════════════════════════
# ── PER-CAMERA STATE ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class CameraState:
    """Thread-safe per-camera state container."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.lock = threading.Lock()
        # Latest raw frame (captured by capture_thread)
        self.raw_frame: Optional[np.ndarray] = None
        # Latest fully-rendered display frame (set by display_thread)
        self.display_frame: Optional[np.ndarray] = None
        # BBox-only annotation data produced by inference thread (OPT-02/03/04)
        # Format: list of {"box": [x1,y1,x2,y2], "color": (B,G,R), "label": str, "thick": int}
        self.bbox_data: List[dict] = []
        # Latest threat state JSON string
        self.last_state_json: str = "{}"
        # Latest full state dict (for /api/status)
        self.last_state: dict = {}
        # Camera metadata
        self.status: str = "starting"
        self.cam_fps: float = 0.0
        self.inf_fps: float = 0.0
        # display_frame identity for MJPEG dedup (OPT-09)
        self._display_frame_id: int = -1

    def update_raw(self, frame: np.ndarray) -> None:
        with self.lock:
            self.raw_frame = frame

    def update_bbox(self, bbox_data: List[dict], state_json: str, state: dict) -> None:
        """Called by inference thread — stores bbox data + state, no full frame copy."""
        with self.lock:
            self.bbox_data = bbox_data
            self.last_state_json = state_json
            self.last_state = state

    def update_display(self, frame: np.ndarray) -> None:
        """Called by display thread — stores rendered frame for MJPEG."""
        with self.lock:
            self.display_frame = frame
            self._display_frame_id = id(frame)

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.raw_frame

    def get_display_frame(self) -> tuple[Optional[np.ndarray], int]:
        """Returns (frame, frame_id) for dedup check."""
        with self.lock:
            return self.display_frame, self._display_frame_id

    # Legacy compat for any code that still calls get_stream_frame()
    def get_stream_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            f = self.display_frame
            if f is None:
                f = self.raw_frame
        return f

    def update_stream(self, frame: np.ndarray, state_json: str, state: dict) -> None:
        """Legacy compat — not used in the new display-split path."""
        with self.lock:
            self.display_frame = frame
            self._display_frame_id = id(frame)
            self.last_state_json = state_json
            self.last_state = state


# ── Global registry of camera states ─────────────────────────────────────────
_camera_states: Dict[str, CameraState] = {}
_stop_event = threading.Event()
_ws_loop: Optional[asyncio.AbstractEventLoop] = None
ws_clients: Set[WebSocket] = set()
_last_panel_bc: list[float] = [0.0]   # mutable container; avoids 'global' inside thread methods
_panel_bc_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# ── WEBSOCKET BROADCAST ───────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _broadcast_sync(msg: str) -> None:
    if _ws_loop is None or not ws_clients:
        return
    asyncio.run_coroutine_threadsafe(_broadcast_async(msg), _ws_loop)


async def _broadcast_async(msg: str) -> None:
    dead: Set[WebSocket] = set()
    for ws in list(ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


# ═══════════════════════════════════════════════════════════════════════════════
# ── C2 WEBHOOK ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _send_c2_webhook(payload: dict) -> None:
    """Fire-and-forget C2 webhook in a daemon thread."""
    if not _C2_URL:
        return
    retry_attempts = int(_C2_CFG.get("retry_attempts", 3))
    timeout_sec = float(_C2_CFG.get("timeout_sec", 5))

    def _post():
        try:
            import httpx
            for attempt in range(retry_attempts):
                try:
                    with httpx.Client(timeout=timeout_sec) as client:
                        client.post(_C2_URL, json=payload)
                    break
                except Exception as exc:
                    logger.warning(f"[C2] Webhook attempt {attempt + 1} failed: {exc}")
                    time.sleep(1)
        except ImportError:
            logger.warning("[C2] httpx not installed. pip install httpx")

    threading.Thread(target=_post, daemon=True, name="C2Webhook").start()



# ═══════════════════════════════════════════════════════════════════════════════
# ── STRUCTURED LOG WORKER (OPT-08) ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

_STRUCTURED_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS pose_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    camera_id   TEXT    NOT NULL,
    anomaly     TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    keypoints   TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS emotion_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    camera_id   TEXT    NOT NULL,
    dominant    TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    scores      TEXT    NOT NULL,
    face_box    TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS weapon_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    camera_id   TEXT    NOT NULL,
    labels      TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    boxes       TEXT    NOT NULL
);
"""


class StructuredLogWorker:
    """
    OPT-08: Non-blocking async log worker.
    Inference thread calls push(); this worker drains the queue in the
    background and writes to SQLite in batches. Never blocks the caller.
    """

    BATCH_INTERVAL = 1.5   # seconds between DB flushes
    MAX_QUEUE      = 500   # drop oldest when full to prevent runaway memory

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=self.MAX_QUEUE)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="StructuredLogWorker"
        )
        self._thread.start()
        self._init_schema()

    def _connect(self):
        import sqlite3
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        try:
            conn = self._connect()
            conn.executescript(_STRUCTURED_LOG_SCHEMA)
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning(f"[StructuredLogWorker] Schema init failed: {exc}")

    def push(self, record: dict) -> None:
        """Non-blocking push. Drops oldest record if queue is full."""
        try:
            self._q.put_nowait(record)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(record)
            except queue.Full:
                pass

    def _run(self) -> None:
        while True:
            batch: list[dict] = []
            deadline = time.perf_counter() + self.BATCH_INTERVAL
            while time.perf_counter() < deadline:
                remaining = max(0.05, deadline - time.perf_counter())
                try:
                    item = self._q.get(timeout=remaining)
                    batch.append(item)
                except queue.Empty:
                    break

            if not batch:
                continue

            poses, emotions, weapons = [], [], []
            for rec in batch:
                t = rec.get("_type")
                if t == "pose":      poses.append(rec)
                elif t == "emotion": emotions.append(rec)
                elif t == "weapon":  weapons.append(rec)

            try:
                conn = self._connect()
                if poses:
                    conn.executemany(
                        "INSERT INTO pose_logs "
                        "(ts,camera_id,anomaly,confidence,keypoints) "
                        "VALUES (:ts,:camera_id,:anomaly,:confidence,:keypoints)",
                        poses,
                    )
                if emotions:
                    conn.executemany(
                        "INSERT INTO emotion_logs "
                        "(ts,camera_id,dominant,confidence,scores,face_box) "
                        "VALUES (:ts,:camera_id,:dominant,:confidence,:scores,:face_box)",
                        emotions,
                    )
                if weapons:
                    conn.executemany(
                        "INSERT INTO weapon_logs "
                        "(ts,camera_id,labels,confidence,boxes) "
                        "VALUES (:ts,:camera_id,:labels,:confidence,:boxes)",
                        weapons,
                    )
                conn.commit()
                conn.close()
            except Exception as exc:
                logger.debug(f"[StructuredLogWorker] Write failed: {exc}")


# Singleton initialised in lifespan startup
_structured_logger: Optional["StructuredLogWorker"] = None


# ═══════════════════════════════════════════════════════════════════════════════
# ── HUD OVERLAY ───────────────────────────────────════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_hud(frame: np.ndarray, state: dict, camera_id: str, fps: float) -> None:
    h, w = frame.shape[:2]
    alert_colors = {
        "INFO":     (0, 200, 80),
        "WARNING":  (0, 165, 255),
        "CRITICAL": (0, 0, 255),
    }
    level = state.get("alert_level", "INFO")
    color = alert_colors.get(level, (200, 200, 200))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (8, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    score = state.get("threat_score", 0)
    cv2.putText(frame, f"[{level}]",           (10, 26),  cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
    cv2.putText(frame, f"THREAT:{score}/10",   (160, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"CAM:{camera_id}",     (340, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 220, 255), 2)

    wpn = "WPN:YES" if state.get("weapon_detected") else "WPN:NO"
    wcol = (0, 0, 255) if state.get("weapon_detected") else (80, 200, 80)
    cv2.putText(frame, wpn, (530, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, wcol, 2)

    anpr_hits = state.get("anpr_hits", [])
    if anpr_hits:
        cv2.putText(frame, f"PLATE:{anpr_hits[0]}", (660, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.putText(frame, f"{fps:.1f}FPS", (w - 90, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 100, 100), 1)


# ── BBox-only annotation helper (OPT-02/03/04) ────────────────────────────────────────────────────────

def _draw_bboxes_only(frame: np.ndarray, bbox_data: List[dict]) -> None:
    """
    OPT-02/03/04: Draw ONLY bounding boxes + a small label.
    No keypoints, no skeleton bones, no confidence text blocks.
    This is the fast path run by the display thread at 30 FPS.
    Each entry in bbox_data:
        {"box": [x1,y1,x2,y2], "color": (B,G,R), "label": str, "thick": int}
    """
    for item in bbox_data:
        x1, y1, x2, y2 = item["box"]
        color = item.get("color", (0, 220, 100))
        thick = item.get("thick", 2)
        label = item.get("label", "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        if label:
            # Compact single-line label above the box
            cv2.putText(
                frame, label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ── CAMERA WORKER THREAD ─────────────────────────════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

class CameraWorker:
    """
    Per-camera worker: captures frames, runs inference pipeline, broadcasts.
    Each instance runs THREE daemon threads:
      1. capture_thread   — reads frames from cv2.VideoCapture into a queue
      2. inference_thread — pulls frames, runs detectors, pushes bbox_data
      3. display_thread   — reads raw_frame + bbox_data, renders, stores display_frame
    The display thread always targets 30 FPS independently of inference speed (OPT-01).
    """

    RECONNECT_DELAY = 2.0
    MAX_RETRIES = 0          # 0 = retry forever

    # Pose anomaly → BGR color for live bbox (OPT-02)
    _POSE_COLORS = {
        "normal":            (0,  200, 80),
        "raised_hands":      (0,  165, 255),
        "aggressive_stance": (0,  0,   255),
        "crouching":         (0,  200, 200),
        "running":           (255, 100, 0),
    }

    def __init__(self, cam_cfg: dict) -> None:
        self.cam_id   = str(cam_cfg["id"])
        self.cam_name = str(cam_cfg.get("name", self.cam_id))
        source_raw    = cam_cfg.get("source", 0)

        # Convert string "0" or "1" to int for webcam index
        if isinstance(source_raw, str):
            try:
                self.source = int(source_raw)
            except ValueError:
                self.source = source_raw
        else:
            self.source = source_raw

        self.loop       = bool(cam_cfg.get("loop", False))
        self.night_mode = bool(cam_cfg.get("night_mode", False))
        self.fps_limit  = float(cam_cfg.get("fps_limit", 30))
        self.zones_cfg  = cam_cfg.get("zones", [])
        self.zones      = FenceDetector.parse_zones_from_config(self.zones_cfg)

        # Per-camera state
        _camera_states[self.cam_id] = CameraState(self.cam_id)
        self.state = _camera_states[self.cam_id]

        # Frame queue between capture and inference (maxsize=2 prevents stale backlog)
        self._frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)

        logger.info(f"[{self.cam_id}] Worker created — source: {self.source}")

    def start(self) -> None:
        threading.Thread(
            target=self._capture_loop, daemon=True, name=f"Cap-{self.cam_id}"
        ).start()
        threading.Thread(
            target=self._inference_loop, daemon=True, name=f"Inf-{self.cam_id}"
        ).start()
        # OPT-01: dedicated display thread
        threading.Thread(
            target=self._display_loop, daemon=True, name=f"Disp-{self.cam_id}"
        ).start()

    # ── Capture thread ────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        attempt = 0
        min_interval = 1.0 / max(self.fps_limit, 1)

        while not _stop_event.is_set():
            attempt += 1
            logger.info(f"[{self.cam_id}] Connecting (attempt {attempt}) …")

            cap = cv2.VideoCapture(self.source)
            # Suppress per-frame FFmpeg codec warnings for local files
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                logger.error(f"[{self.cam_id}] Cannot open source: {self.source}")
                cap.release()
                self.state.status = "error"
                if self.MAX_RETRIES and attempt >= self.MAX_RETRIES:
                    return
                time.sleep(self.RECONNECT_DELAY)
                continue

            self.state.status = "live"
            attempt = 0
            t0, frames, fails = time.perf_counter(), 0, 0
            last_t = time.perf_counter()

            while not _stop_event.is_set():
                ok, frame = cap.read()

                if not ok:
                    fails += 1
                    # For looped file sources: restart capture
                    if self.loop and isinstance(self.source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        fails = 0
                        continue
                    if fails >= 10:
                        logger.warning(f"[{self.cam_id}] Stream lost. Reconnecting…")
                        break
                    time.sleep(0.005)
                    continue

                fails = 0
                self.state.update_raw(frame)

                # FPS throttle: drop frame if we're ahead of fps_limit
                now = time.perf_counter()
                elapsed_since_last = now - last_t
                if elapsed_since_last < min_interval:
                    time.sleep(min_interval - elapsed_since_last)
                last_t = time.perf_counter()

                # Push to inference queue (drop oldest if full)
                try:
                    self._frame_q.put_nowait(frame.copy())
                except queue.Full:
                    try:
                        self._frame_q.get_nowait()    # discard oldest
                    except queue.Empty:
                        pass
                    try:
                        self._frame_q.put_nowait(frame.copy())
                    except queue.Full:
                        pass

                frames += 1
                elapsed = time.perf_counter() - t0
                if elapsed >= 2.0:
                    self.state.cam_fps = frames / elapsed
                    set_camera_fps(self.state.cam_fps)
                    frames, t0 = 0, time.perf_counter()

            cap.release()
            if _stop_event.is_set():
                break
            self.state.status = "reconnecting"
            time.sleep(self.RECONNECT_DELAY)

        self.state.status = "stopped"
        logger.info(f"[{self.cam_id}] Capture stopped.")

    # ── Inference thread ──────────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        logger.info(f"[{self.cam_id}] Loading detectors…")

        # ── Load detectors (each gracefully disabled if model missing) ────────
        pose_det: Optional[PoseDetector] = None
        weapon_det: Optional[WeaponDetector] = None
        emo_det: Optional[EmotionDetector] = None
        anpr_det: Optional[ANPRDetector] = None
        vehicle_det: Optional[VehicleDetector] = None
        fence_det = FenceDetector()
        night_pre = NightPreprocessor(clip_limit=_CLAHE_CLIP, tile_grid=_CLAHE_TILE)
        aggregator = ThreatAggregator()

        try:
            pose_det = PoseDetector()
        except Exception as e:
            logger.warning(f"[{self.cam_id}] PoseDetector failed: {e}")

        try:
            weapon_det = WeaponDetector()
        except Exception as e:
            logger.warning(f"[{self.cam_id}] WeaponDetector failed: {e}")

        try:
            emo_det = EmotionDetector()
        except Exception as e:
            logger.warning(f"[{self.cam_id}] EmotionDetector failed: {e}")

        if _ANPR_ENABLED:
            try:
                anpr_det = ANPRDetector(
                    model_path=_ANPR_MODEL,
                    watchlist_path=_ANPR_WATCHLIST,
                    confidence_threshold=_ANPR_CONF,
                    ocr_engine=_ANPR_OCR,
                    onnx_providers=_ONNX_PROVIDERS,
                )
            except Exception as e:
                logger.warning(f"[{self.cam_id}] ANPRDetector failed: {e}")

        try:
            vehicle_det = VehicleDetector(onnx_providers=_ONNX_PROVIDERS)
        except Exception as e:
            logger.warning(f"[{self.cam_id}] VehicleDetector failed: {e}")

        logger.info(f"[{self.cam_id}] Detectors ready.")

        frame_count = 0
        last_weapon  = WeaponResult()
        last_poses: list[PersonPose] = []
        last_emos: list[EmotionResult] = []
        last_emotion_scores: dict = {}
        last_anpr    = ANPRResult()
        last_vehicle = VehicleResult()
        last_fence   = FenceResult()
        t0, inf_frames = time.perf_counter(), 0

        while not _stop_event.is_set():
            loop_t0 = time.perf_counter()

            # Pull frame from queue (block up to 1s)
            try:
                raw = self._frame_q.get(timeout=1.0)
            except queue.Empty:
                continue

            frame_count += 1
            frame = raw
            H, W = frame.shape[:2]

            # ── 0. Night mode pre-processing ─────────────────────────────────
            if self.night_mode:
                frame = night_pre.process(frame, enabled=True)

            # ── Inference at reduced resolution for speed ─────────────────────
            proc_w, proc_h = 640, 360
            small = cv2.resize(frame, (proc_w, proc_h))
            sx, sy = W / proc_w, H / proc_h

            # ── 1. Weapon detection (every WEAPON_EVERY frames) ───────────────
            if weapon_det and frame_count % WEAPON_EVERY == 0:
                try:
                    from app.config import WEAPON_CONF
                    last_weapon = weapon_det.detect(small, conf=WEAPON_CONF)
                    scaled_boxes = []
                    for box in last_weapon.boxes:
                        scaled_boxes.append([
                            int(box[0] * sx), int(box[1] * sy),
                            int(box[2] * sx), int(box[3] * sy),
                        ])
                    last_weapon.boxes = scaled_boxes
                except Exception as e:
                    logger.debug(f"[{self.cam_id}] Weapon err: {e}")

            # ── 2. Pose detection (every POSE_EVERY frames) ───────────────────
            if pose_det and frame_count % POSE_EVERY == 0:
                try:
                    last_poses = pose_det.detect(small)
                    for p in last_poses:
                        p.keypoints[:, 0] *= sx
                        p.keypoints[:, 1] *= sy
                        p.bbox = [
                            int(p.bbox[0] * sx), int(p.bbox[1] * sy),
                            int(p.bbox[2] * sx), int(p.bbox[3] * sy),
                        ]
                except Exception as e:
                    logger.debug(f"[{self.cam_id}] Pose err: {e}")

            # ── 3. Emotion detection (every EMOTION_EVERY frames) ─────────────
            if emo_det and frame_count % EMOTION_EVERY == 0:
                try:
                    last_emos = emo_det.detect(small)
                    for er in last_emos:
                        if er.face_box:
                            er.face_box = [
                                int(er.face_box[0] * sx), int(er.face_box[1] * sy),
                                int(er.face_box[2] * sx), int(er.face_box[3] * sy),
                            ]
                    if last_emos:
                        last_emotion_scores = last_emos[0].scores
                except Exception as e:
                    logger.debug(f"[{self.cam_id}] Emotion err: {e}")

            # ── 4. Vehicle detection (every VEHICLE_EVERY frames) ─────────────
            if vehicle_det and frame_count % VEHICLE_EVERY == 0:
                try:
                    last_vehicle = vehicle_det.detect(small)
                    # Scale vehicle boxes back to original resolution
                    for v in last_vehicle.vehicles:
                        v.bbox = [
                            int(v.bbox[0] * sx), int(v.bbox[1] * sy),
                            int(v.bbox[2] * sx), int(v.bbox[3] * sy),
                        ]
                except Exception as e:
                    logger.debug(f"[{self.cam_id}] Vehicle err: {e}")

            # ── 5. ANPR (every ANPR_EVERY frames, ROI-gated by vehicle) ───────
            if anpr_det and frame_count % ANPR_EVERY == 0:
                # ROI gate: only run ANPR if at least one vehicle is present
                if last_vehicle.detected:
                    try:
                        last_anpr = anpr_det.detect(frame)
                    except Exception as e:
                        logger.debug(f"[{self.cam_id}] ANPR err: {e}")
                else:
                    last_anpr = ANPRResult()

            # ── 6. Virtual fence detection ────────────────────────────────────
            if self.zones:
                try:
                    person_bboxes = [p.bbox for p in last_poses]
                    last_fence = fence_det.detect(person_bboxes, self.zones, W, H)
                except Exception as e:
                    logger.debug(f"[{self.cam_id}] Fence err: {e}")
            else:
                last_fence = FenceResult()

            # ── Build bbox_data for display thread (OPT-01/02/03/04) ──────────
            # Only lightweight box descriptors — NO cv2 draw calls in this thread.
            bbox_data: List[dict] = []

            # Pose boxes: color-coded by anomaly (OPT-02)
            for p in last_poses:
                color = self._POSE_COLORS.get(p.anomaly.value, (0, 200, 80))
                lbl   = p.anomaly.value.replace("_", " ").upper() if p.anomaly.value != "normal" else ""
                bbox_data.append({"box": p.bbox, "color": color, "label": lbl, "thick": 2})

            # Weapon boxes: thick red/orange, compact label (OPT-04)
            for box, label in zip(last_weapon.boxes, last_weapon.labels):
                tier_color = (0, 0, 255) if label == "pistol" else (0, 120, 255)
                bbox_data.append({"box": box, "color": tier_color, "label": f"WPN:{label.upper()}", "thick": 3})

            # Emotion face boxes: thin border + emotion label (OPT-03)
            _emo_colors = {
                "angry":    (0,   0,   255),
                "fear":     (128, 0,   128),
                "happy":    (0,   255, 128),
                "neutral":  (180, 180, 180),
                "sad":      (255, 128, 0),
                "surprise": (0,   200, 255),
                "disgust":  (0,   128, 128),
            }
            for er in last_emos:
                if er.face_box:
                    ec = _emo_colors.get(er.label, (200, 200, 200))
                    bbox_data.append({"box": er.face_box, "color": ec, "label": er.label.upper(), "thick": 1})

            # Vehicle boxes: thin yellow-grey
            for v in last_vehicle.vehicles:
                bbox_data.append({"box": v.bbox, "color": (180, 180, 60), "label": v.vehicle_type.upper(), "thick": 1})

            # ── Compute threat score ──────────────────────────────────────────
            top_emo    = last_emos[0].label  if last_emos else "neutral"
            top_scores = last_emos[0].scores if last_emos else last_emotion_scores
            pose_labels= [p.anomaly.value for p in last_poses]
            worst_pose = "normal"
            for pa in pose_labels:
                if pa == PoseAnomaly.AGGRESSIVE.value:
                    worst_pose = pa; break
                elif pa == PoseAnomaly.RAISED_HANDS.value:
                    worst_pose = pa

            threat_score, alert_level = compute_threat_score(
                weapon_detected=last_weapon.detected,
                weapon_labels=last_weapon.labels,
                pose_anomaly=worst_pose,
                emotion=top_emo,
                fence_result=last_fence,
                anpr_result=last_anpr,
                vehicle_result=last_vehicle,
                night_mode_active=self.night_mode,
                num_persons=len(last_poses),
            )

            existing_state = aggregator.update(
                weapon_detected=last_weapon.detected,
                weapon_labels=last_weapon.labels,
                weapon_display=last_weapon.display_labels,
                emotion=top_emo,
                emotion_scores=top_scores,
                pose_anomalies=pose_labels,
                num_persons=len(last_poses),
            )

            # ── Build state dict (WS payload) ─────────────────────────────────
            fence_breaches_list = [
                {
                    "zone_id":     b.zone_id,
                    "zone_name":   b.zone_name,
                    "track_id":    b.track_id,
                    "dwell_sec":   b.dwell_sec,
                    "is_critical": b.is_critical,
                }
                for b in last_fence.breaches
            ]
            vehicle_counts = last_vehicle.counts if last_vehicle else {}
            anpr_plates = [p.plate_text for p in last_anpr.plates] if last_anpr else []
            anpr_hits   = last_anpr.watchlist_hits if last_anpr else []

            state_dict: dict[str, Any] = {
                "camera_id":            self.cam_id,
                "camera_name":          self.cam_name,
                "alert_level":          alert_level,
                "threat_score":         threat_score,
                "weapon_detected":      last_weapon.detected,
                "weapon_labels":        last_weapon.labels,
                "weapon_display_labels":last_weapon.display_labels,
                "emotion":              top_emo,
                "emotion_scores":       top_scores,
                "pose_anomaly":         worst_pose,
                "num_persons":          len(last_poses),
                "fence_breaches":       fence_breaches_list,
                "fence_has_breach":     last_fence.has_breach,
                "fence_has_critical":   last_fence.has_critical,
                "anpr_plates":          anpr_plates,
                "anpr_hits":            anpr_hits,
                "anpr_watchlist_hit":   bool(anpr_hits),
                "vehicle_counts":       vehicle_counts,
                "vehicle_detected":     last_vehicle.detected,
                "night_mode":           self.night_mode,
                "timestamp":            existing_state.timestamp,
                "inf_fps":              round(self.state.inf_fps, 1),
                "cam_fps":              round(self.state.cam_fps, 1),
            }
            state_json = json.dumps(state_dict)

            # OPT-01: push bbox_data + state — display_loop renders independently
            self.state.update_bbox(bbox_data, state_json, state_dict)

            # Carry fence state for display_loop zone rendering
            with self.state.lock:
                self.state._last_fence  = last_fence
                self.state._fence_zones = self.zones
                self.state._fence_det   = fence_det

            # ── Prometheus metrics ────────────────────────────────────────────
            update_threat_metrics(
                threat_score=threat_score,
                alert_level=alert_level,
                weapon_detected=last_weapon.detected,
                emotion=top_emo,
                emotion_scores=top_scores,
                pose_anomaly=worst_pose,
                num_persons=len(last_poses),
                alert_event=(alert_level != "INFO"),
            )

            # ── WS broadcast: every 1 s (OPT-10) ─────────────────────────────
            now = time.perf_counter()
            with _panel_bc_lock:
                if (now - _last_panel_bc[0]) >= 1.0:
                    _broadcast_sync(state_json)
                    _last_panel_bc[0] = now

            # ── Event logging (WARNING / CRITICAL) ───────────────────────────
            if alert_level in ("WARNING", "CRITICAL"):
                snap_frame = None
                if alert_level == "CRITICAL":
                    snap_frame = self.state.get_stream_frame()
                _event_logger.log(
                    camera_id=self.cam_id,
                    alert_level=alert_level,
                    threat_score=threat_score,
                    weapon_detected=last_weapon.detected,
                    weapon_labels=last_weapon.labels,
                    emotion=top_emo,
                    pose_anomaly=worst_pose,
                    fence_breaches=fence_breaches_list,
                    anpr_plates=anpr_plates,
                    anpr_hits=anpr_hits,
                    frame=snap_frame,
                    raw_payload=state_dict,
                )
                if alert_level == "CRITICAL":
                    _send_c2_webhook(state_dict)

            # ── FPS counter ───────────────────────────────────────────────────
            inf_frames += 1
            elapsed = time.perf_counter() - t0
            if elapsed >= 2.0:
                self.state.inf_fps = inf_frames / elapsed
                set_inference_fps(self.state.inf_fps)
                inf_frames, t0 = 0, time.perf_counter()

            observe_inference_loop_latency_seconds(time.perf_counter() - loop_t0)

        logger.info(f"[{self.cam_id}] Inference stopped.")

    # ── Display thread (OPT-01) ───────────────────────────────────────────────

    def _display_loop(self) -> None:
        """
        OPT-01: Dedicated display thread targeting 30 FPS.
        Reads raw_frame + bbox_data and renders only lightweight bboxes.
        Completely decoupled from inference speed.
        """
        target_interval = 1.0 / 30.0
        last_t = time.perf_counter()

        while not _stop_event.is_set():
            raw = self.state.get_raw_frame()
            if raw is None:
                time.sleep(0.01)
                continue

            # One copy for this display frame
            vis = raw.copy()

            # Snapshot bbox_data + state without holding lock long
            with self.state.lock:
                bboxes      = list(self.state.bbox_data)
                cur_state   = dict(self.state.last_state)
                last_fence  = getattr(self.state, "_last_fence",  FenceResult())
                fence_zones = getattr(self.state, "_fence_zones", [])
                fence_det   = getattr(self.state, "_fence_det",   None)

            # Fast bbox-only drawing (OPT-02/03/04)
            _draw_bboxes_only(vis, bboxes)

            # Geofence zone lines (lightweight vector ops only)
            if fence_zones and fence_det is not None:
                try:
                    fence_det.annotate(vis, last_fence, fence_zones)
                except Exception:
                    pass

            # HUD strip
            _draw_hud(vis, cur_state, self.cam_id, self.state.inf_fps)

            # Store rendered frame for MJPEG generator
            self.state.update_display(vis)

            # Throttle to ~30 FPS
            now = time.perf_counter()
            sleep_t = target_interval - (now - last_t)
            if sleep_t > 0:
                time.sleep(sleep_t)
            last_t = time.perf_counter()

        logger.info(f"[{self.cam_id}] Display stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
# ── MJPEG GENERATOR ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _mjpeg_gen(camera_id: str):
    """
    OPT-06/09: Downscaled (960×540) MJPEG stream with frame deduplication.
    Only encodes a new JPEG when the display thread has produced a new frame;
    never sends the same frame twice.
    """
    boundary      = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY]   # OPT-06: quality=60
    interval      = 1.0 / 30.0
    last_t        = time.perf_counter()
    last_frame_id = -1   # OPT-09: dedup sentinel

    state = _camera_states.get(camera_id)

    while True:
        if state is None:
            state = _camera_states.get(camera_id)

        if state is None:
            time.sleep(0.02)
            continue

        frame, frame_id = state.get_display_frame()

        # OPT-09: skip if nothing new
        if frame is None or frame_id == last_frame_id:
            time.sleep(0.005)
            continue

        last_frame_id = frame_id

        # OPT-06: downscale to 960×540 before encode
        try:
            small = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
        except Exception:
            small = frame

        ret, buf = cv2.imencode(".jpg", small, encode_params)
        if not ret:
            time.sleep(0.01)
            continue

        yield boundary + buf.tobytes() + b"\r\n"

        now     = time.perf_counter()
        sleep_t = interval - (now - last_t)
        if sleep_t > 0:
            time.sleep(sleep_t)
        last_t = time.perf_counter()


# ═══════════════════════════════════════════════════════════════════════════════
# ── FASTAPI APP ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ws_loop, _structured_logger
    _ws_loop = asyncio.get_event_loop()

    # OPT-08: start async structured log worker (pose/emotion/weapon tables)
    _structured_logger = StructuredLogWorker(_DB_PATH)
    logger.info("[IBVAP] StructuredLogWorker started.")

    # Start one CameraWorker per camera in cameras.yaml
    workers: list[CameraWorker] = []
    if not _CAMERAS_CFG:
        logger.warning("No cameras defined in cameras.yaml! The pipeline will idle.")
    for cam_cfg in _CAMERAS_CFG:
        try:
            worker = CameraWorker(cam_cfg)
            worker.start()
            workers.append(worker)
        except Exception as exc:
            logger.error(f"Failed to start worker for {cam_cfg.get('id', '?')}: {exc}")

    logger.info(f"[IBVAP] {len(workers)} camera worker(s) started.")
    logger.info(f"[IBVAP] Dashboard → http://localhost:{PORT}")

    yield

    _stop_event.set()
    logger.info("[IBVAP] Shutting down…")


app = FastAPI(title="IBVAP — Intelligent Border Video Analytics Platform", lifespan=lifespan)

# ── Static files + Jinja2 templates ─────────────────────────────────────────
_STATIC_DIR   = _PROJECT_ROOT / "static"
_TEMPLATE_DIR = _PROJECT_ROOT / "templates"

os.makedirs(_STATIC_DIR, exist_ok=True)
os.makedirs(_TEMPLATE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
_jinja = Jinja2Templates(directory=str(_TEMPLATE_DIR))


# ── Prometheus middleware ─────────────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request, call_next):
    skip = request.url.path in ("/metrics", "/video_feed")
    start = time.perf_counter()
    response = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = getattr(response, "status_code", 500)
        return response
    finally:
        if not skip:
            latency = time.perf_counter() - start
            REQUESTS_TOTAL.labels(
                method=request.method,
                path=request.url.path,
                http_status=str(status_code),
            ).inc()
            REQUEST_LATENCY_SECONDS.labels(
                method=request.method, path=request.url.path
            ).observe(latency)


# ── Multi-page Dashboard Routes ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def page_live_matrix(request: Request):
    return _jinja.TemplateResponse(request=request, name="live_matrix.html",
                                    context={"active_page": "live"})


@app.get("/threat-feed", response_class=HTMLResponse)
async def page_threat_feed(request: Request):
    return _jinja.TemplateResponse(request=request, name="threat_feed.html",
                                    context={"active_page": "threat"})


@app.get("/threat-intel", response_class=HTMLResponse)
async def page_threat_intel(request: Request):
    return _jinja.TemplateResponse(request=request, name="threat_intel.html",
                                    context={"active_page": "threat_intel"})


@app.get("/anpr-tracker", response_class=HTMLResponse)
async def page_anpr_tracker(request: Request):
    return _jinja.TemplateResponse(request=request, name="anpr_tracker.html",
                                    context={"active_page": "anpr"})


@app.get("/geofence-editor", response_class=HTMLResponse)
async def page_geofence_editor(request: Request):
    return _jinja.TemplateResponse(request=request, name="geofence_editor.html",
                                    context={"active_page": "geofence"})


@app.get("/incident-logs", response_class=HTMLResponse)
async def page_incident_logs(request: Request):
    return _jinja.TemplateResponse(request=request, name="incident_logs.html",
                                    context={"active_page": "incidents"})


# ── MJPEG Video Feed ──────────────────────────────────────────────────────────
@app.get("/video_feed")
async def video_feed(camera: str = ""):
    # Default to first camera if not specified
    if not camera and _CAMERAS_CFG:
        camera = _CAMERAS_CFG[0]["id"]
    if camera not in _camera_states:
        # Fallback: serve first available camera
        if _camera_states:
            camera = next(iter(_camera_states))
        else:
            return Response(content="No cameras available", status_code=503)

    return StreamingResponse(
        _mjpeg_gen(camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket Alerts ─────────────────────────────────────────────────────────
@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        # Send latest state immediately on connect
        if _camera_states:
            first_state = next(iter(_camera_states.values()))
            await ws.send_text(first_state.last_state_json or "{}")
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=25.0)
            except asyncio.TimeoutError:
                await ws.send_text('{"ping":true}')
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ws_clients.discard(ws)


# ── API: Camera list ─────────────────────────────────────────────────────────
@app.get("/api/cameras")
async def api_cameras():
    cams = []
    for cam_cfg in _CAMERAS_CFG:
        cam_id = cam_cfg.get("id", "")
        state  = _camera_states.get(cam_id)
        cams.append({
            "id":         cam_id,
            "name":       cam_cfg.get("name", cam_id),
            "source":     str(cam_cfg.get("source", "")),
            "night_mode": cam_cfg.get("night_mode", False),
            "fps_limit":  cam_cfg.get("fps_limit", 30),
            "status":     state.status if state else "unknown",
            "cam_fps":    round(state.cam_fps, 1) if state else 0.0,
            "inf_fps":    round(state.inf_fps, 1) if state else 0.0,
            "zones":      cam_cfg.get("zones", []),
        })
    return JSONResponse(content={"cameras": cams})


# ── API: Status (per-camera) ─────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    result = {}
    for cam_id, state in _camera_states.items():
        try:
            result[cam_id] = json.loads(state.last_state_json or "{}")
        except Exception:
            result[cam_id] = {}
    return JSONResponse(content=result)


# ── API: Events ──────────────────────────────────────────────────────────────
@app.get("/api/events")
async def api_events(camera: Optional[str] = None, level: Optional[str] = None, limit: int = 50):
    rows = _event_logger.query(camera_id=camera, level=level, limit=limit)
    return JSONResponse(content={"events": rows, "count": len(rows)})


# ── API: Snapshots list ───────────────────────────────────────────────────────
@app.get("/api/snapshots")
async def api_snapshots():
    snaps = _event_logger.list_snapshots()
    return JSONResponse(content={"snapshots": snaps, "count": len(snaps)})


# ── Serve snapshot file ───────────────────────────────────────────────────────
@app.get("/snapshots/{filename}")
async def serve_snapshot(filename: str):
    filepath = os.path.join(_SNAPSHOTS_DIR, filename)
    if not os.path.isfile(filepath):
        return Response(content="Not found", status_code=404)
    # Security: prevent path traversal
    real = os.path.realpath(filepath)
    snap_real = os.path.realpath(_SNAPSHOTS_DIR)
    if not real.startswith(snap_real):
        return Response(content="Forbidden", status_code=403)
    return FileResponse(filepath, media_type="image/jpeg")


# ── API: Watchlist (plates — simple list) ────────────────────────────────────
_ANPR_DETAIL_WATCHLIST = str(_PROJECT_ROOT / "data" / "watchlist" / "plates_detail.json")


@app.get("/api/watchlist/plates")
async def get_watchlist():
    try:
        with open(_ANPR_WATCHLIST, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except FileNotFoundError:
        return JSONResponse(content={"plates": []})
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/watchlist/plates")
async def add_watchlist_plate(payload: dict):
    """Add a plate (simple). Body: {"plate": "MH12AB1234"}"""
    plate = str(payload.get("plate", "")).strip().upper()
    if not plate:
        return JSONResponse(content={"error": "plate field required"}, status_code=400)
    try:
        with open(_ANPR_WATCHLIST, "r") as f:
            data = json.load(f)
    except Exception:
        data = {"plates": []}
    plates_set = set(data.get("plates", []))
    plates_set.add(plate)
    data["plates"] = sorted(plates_set)
    with open(_ANPR_WATCHLIST, "w") as f:
        json.dump(data, f, indent=2)
    return JSONResponse(content={"status": "ok", "plate": plate, "total": len(plates_set)})


# ── API: Watchlist Detail (plates with metadata) ──────────────────────────────

@app.get("/api/watchlist/plates/detail")
async def get_watchlist_detail():
    """Get all plates with full metadata (type, status, notes)."""
    try:
        with open(_ANPR_DETAIL_WATCHLIST, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except FileNotFoundError:
        return JSONResponse(content={"plates": []})
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/watchlist/plates/detail")
async def add_watchlist_detail(payload: dict):
    """
    Add a plate with full metadata.
    Body: {"plate": "MH12", "vehicle_type": "Sedan", "status": "watch", "notes": "..."}
    """
    from datetime import datetime, timezone
    plate = str(payload.get("plate", "")).strip().upper()
    if not plate:
        return JSONResponse(content={"error": "plate field required"}, status_code=400)

    entry = {
        "plate":        plate,
        "vehicle_type": str(payload.get("vehicle_type", "")).strip(),
        "status":       str(payload.get("status", "watch")).strip(),
        "notes":        str(payload.get("notes", "")).strip(),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(_ANPR_DETAIL_WATCHLIST, "r") as f:
            data = json.load(f)
    except Exception:
        data = {"plates": []}

    # Remove existing entry for same plate
    data["plates"] = [p for p in data.get("plates", []) if p.get("plate") != plate]
    data["plates"].insert(0, entry)

    with open(_ANPR_DETAIL_WATCHLIST, "w") as f:
        json.dump(data, f, indent=2)

    # Also sync to the simple watchlist file used by ANPR detector
    try:
        with open(_ANPR_WATCHLIST, "r") as f:
            simple = json.load(f)
    except Exception:
        simple = {"plates": []}
    plates_set = set(simple.get("plates", []))
    plates_set.add(plate)
    simple["plates"] = sorted(plates_set)
    with open(_ANPR_WATCHLIST, "w") as f:
        json.dump(simple, f, indent=2)

    return JSONResponse(content={"status": "ok", "entry": entry})


@app.delete("/api/watchlist/plates/{plate}")
async def delete_watchlist_plate(plate: str):
    """Remove a plate from both watchlist files."""
    plate = plate.strip().upper()

    # Remove from detail list
    try:
        with open(_ANPR_DETAIL_WATCHLIST, "r") as f:
            data = json.load(f)
        original_count = len(data.get("plates", []))
        data["plates"] = [p for p in data.get("plates", []) if p.get("plate") != plate]
        with open(_ANPR_DETAIL_WATCHLIST, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)

    # Remove from simple list
    try:
        with open(_ANPR_WATCHLIST, "r") as f:
            simple = json.load(f)
        plates_set = set(simple.get("plates", []))
        plates_set.discard(plate)
        simple["plates"] = sorted(plates_set)
        with open(_ANPR_WATCHLIST, "w") as f:
            json.dump(simple, f, indent=2)
    except Exception:
        pass

    return JSONResponse(content={"status": "ok", "plate": plate})


# ── API: Threat Breakdown ─────────────────────────────────────────────────────
@app.get("/api/threat-breakdown")
async def api_threat_breakdown(minutes: int = 30):
    """
    Returns:
      live_threats  – per-type count from all live camera states (right now)
      per_camera    – per-camera alert level + active threat labels
      timeline      – 5-min-bucketed detection counts for last `minutes` minutes
    """
    import sqlite3
    from datetime import datetime, timezone, timedelta

    # ── Live threat state from current camera states ──────────────────────────
    live_threats = {
        "weapon":          0,
        "anpr_blacklist":  0,
        "aggressive_pose": 0,
        "raised_hands":    0,
        "fence_breach":    0,
        "hostile_emotion": 0,
        "vehicle":         0,
    }
    per_camera: dict = {}

    for cam_id, st in _camera_states.items():
        try:
            s = json.loads(st.last_state_json or "{}")
        except Exception:
            s = {}
        cam_threats = []

        if s.get("weapon_detected"):
            live_threats["weapon"] += 1
            cam_threats.append("ARMED PERSON")

        if s.get("anpr_watchlist_hit"):
            live_threats["anpr_blacklist"] += 1
            cam_threats.append("BLACKLISTED PLATE")

        pose = s.get("pose_anomaly", "normal")
        if pose == "aggressive_stance":
            live_threats["aggressive_pose"] += 1
            cam_threats.append("AGGRESSIVE STANCE")
        elif pose == "raised_hands":
            live_threats["raised_hands"] += 1
            cam_threats.append("HANDS RAISED")

        if s.get("fence_has_breach"):
            live_threats["fence_breach"] += 1
            cam_threats.append("FENCE BREACH")

        emo = s.get("emotion", "neutral")
        if emo in ("angry", "fear", "disgust"):
            live_threats["hostile_emotion"] += 1
            cam_threats.append(f"HOSTILE EMOTION ({emo.upper()})")

        if s.get("vehicle_detected"):
            live_threats["vehicle"] += 1

        per_camera[cam_id] = {
            "alert_level":    s.get("alert_level", "INFO"),
            "threat_score":   s.get("threat_score", 0),
            "active_threats": cam_threats,
            "num_persons":    s.get("num_persons", 0),
        }

    # ── Historical timeline from incidents DB ─────────────────────────────────
    bucket_min = 5
    now_utc = datetime.now(timezone.utc)
    cutoff  = now_utc - timedelta(minutes=minutes)
    timeline: list[dict] = []

    try:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT timestamp, weapon_detected, pose_anomaly,
                   fence_has_breach, emotion, anpr_watchlist_hit, vehicle_detected
            FROM incidents
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (cutoff.isoformat(),)).fetchall()
        conn.close()

        # Pre-fill empty buckets
        n_buckets = (minutes // bucket_min) + 1
        buckets: dict[str, dict] = {}
        for i in range(n_buckets):
            t   = cutoff + timedelta(minutes=i * bucket_min)
            key = t.strftime("%H:%M")
            buckets[key] = {"time": key, "weapon": 0, "anpr": 0,
                            "pose": 0, "fence": 0, "emotion": 0, "total": 0}

        for row in rows:
            try:
                ts = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
                offset = (ts - cutoff).total_seconds() / 60
                bidx   = int(offset // bucket_min)
                bkey   = (cutoff + timedelta(minutes=bidx * bucket_min)).strftime("%H:%M")
                if bkey not in buckets:
                    continue
                b = buckets[bkey]
                b["total"] += 1
                if row["weapon_detected"]:    b["weapon"] += 1
                if row["anpr_watchlist_hit"]: b["anpr"]   += 1
                if row["fence_has_breach"]:   b["fence"]  += 1
                if (row["pose_anomaly"] or "") in ("aggressive_stance", "raised_hands"):
                    b["pose"] += 1
                if (row["emotion"] or "") in ("angry", "fear", "disgust"):
                    b["emotion"] += 1
            except Exception:
                continue

        timeline = list(buckets.values())
    except Exception as exc:
        logger.debug(f"[ThreatBreakdown] DB error: {exc}")

    return JSONResponse(content={
        "live_threats": live_threats,
        "per_camera":   per_camera,
        "timeline":     timeline,
        "window_min":   minutes,
        "generated_at": now_utc.isoformat(),
    })


# ── Prometheus Metrics ────────────────────────────────────────────────────────
@app.get("/metrics")
async def metrics_endpoint():
    data = format_metrics()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False, workers=1)
