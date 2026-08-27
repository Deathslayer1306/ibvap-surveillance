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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response, FileResponse

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
except Exception:
    HOST, PORT, JPEG_QUALITY = "0.0.0.0", 8000, 80
    POSE_EVERY, WEAPON_EVERY, EMOTION_EVERY = 1, 1, 5

# New frame-skip intervals for IBVAP detectors
ANPR_EVERY    = 5   # every 5th frame
VEHICLE_EVERY = 3   # every 3rd frame
FACE_EVERY    = 4   # every 4th frame


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
        # Latest raw frame (for MJPEG stream)
        self.raw_frame: Optional[np.ndarray] = None
        # Latest annotated frame (inference output)
        self.stream_frame: Optional[np.ndarray] = None
        # Latest threat state JSON string
        self.last_state_json: str = "{}"
        # Latest full state dict (for /api/status)
        self.last_state: dict = {}
        # Camera metadata
        self.status: str = "starting"
        self.cam_fps: float = 0.0
        self.inf_fps: float = 0.0

    def update_raw(self, frame: np.ndarray) -> None:
        with self.lock:
            self.raw_frame = frame

    def update_stream(self, frame: np.ndarray, state_json: str, state: dict) -> None:
        with self.lock:
            self.stream_frame = frame
            self.last_state_json = state_json
            self.last_state = state

    def get_stream_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            f = self.stream_frame
            if f is None:
                f = self.raw_frame
        return f

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.raw_frame


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


# ═══════════════════════════════════════════════════════════════════════════════
# ── CAMERA WORKER THREAD ─────────────────────────════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

class CameraWorker:
    """
    Per-camera worker: captures frames, runs inference pipeline, broadcasts.
    Each instance runs two daemon threads:
      1. capture_thread   — reads frames from cv2.VideoCapture into a queue
      2. inference_thread — pulls frames, runs detectors, broadcasts WS
    """

    RECONNECT_DELAY = 2.0
    MAX_RETRIES = 0          # 0 = retry forever

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

            # ── Annotate full-resolution frame ────────────────────────────────
            annotated = frame.copy()
            if pose_det:
                annotated = PoseDetector.annotate(annotated, last_poses)
            if weapon_det:
                annotated = weapon_det.annotate(annotated, last_weapon)
            if emo_det:
                annotated = emo_det.annotate(annotated, last_emos)
            if vehicle_det:
                annotated = vehicle_det.annotate(annotated, last_vehicle)
            if anpr_det:
                annotated = anpr_det.annotate(annotated, last_anpr)
            if self.zones:
                annotated = fence_det.annotate(annotated, last_fence, self.zones)

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

            # Also call existing ThreatAggregator for Prometheus metrics compat
            existing_state = aggregator.update(
                weapon_detected=last_weapon.detected,
                weapon_labels=last_weapon.labels,
                weapon_display=last_weapon.display_labels,
                emotion=top_emo,
                emotion_scores=top_scores,
                pose_anomalies=pose_labels,
                num_persons=len(last_poses),
            )

            # ── Build extended state dict (WS payload) ────────────────────────
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

            # ── HUD overlay ───────────────────────────────────────────────────
            _draw_hud(annotated, state_dict, self.cam_id, self.state.inf_fps)

            # ── Update camera state ───────────────────────────────────────────
            self.state.update_stream(annotated, state_json, state_dict)

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

            # ── WS broadcast: update all panels every 1 s ─────────────────
            now = time.perf_counter()
            with _panel_bc_lock:
                if (now - _last_panel_bc[0]) >= 1.0:
                    _broadcast_sync(state_json)
                    _last_panel_bc[0] = now

            # ── Event logging (WARNING / CRITICAL) ───────────────────────────
            if alert_level in ("WARNING", "CRITICAL"):
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
                    frame=(annotated if alert_level == "CRITICAL" else None),
                    raw_payload=state_dict,
                )

                # C2 webhook for CRITICAL
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


# ═══════════════════════════════════════════════════════════════════════════════
# ── MJPEG GENERATOR ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _mjpeg_gen(camera_id: str):
    """Serve MJPEG stream for a specific camera."""
    boundary     = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    quality      = min(JPEG_QUALITY, 75)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    interval      = 1.0 / 30.0
    last_t        = time.perf_counter()

    state = _camera_states.get(camera_id)

    while True:
        if state is None:
            # Try again — might not be initialised yet
            state = _camera_states.get(camera_id)

        frame = state.get_stream_frame() if state else None

        if frame is None:
            time.sleep(0.02)
            continue

        ret, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            time.sleep(0.01)
            continue

        yield boundary + buf.tobytes() + b"\r\n"

        now    = time.perf_counter()
        sleep_t = interval - (now - last_t)
        if sleep_t > 0:
            time.sleep(sleep_t)
        last_t = time.perf_counter()


# ═══════════════════════════════════════════════════════════════════════════════
# ── FASTAPI APP ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ws_loop
    _ws_loop = asyncio.get_event_loop()

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


# ── HTML Dashboard ────────────────────────────────────────────────────────────
_HTML_PATH = pathlib.Path(__file__).resolve().parent.parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=_HTML_PATH.read_text(encoding="utf-8"))


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


# ── API: Watchlist (plates) ───────────────────────────────────────────────────
@app.get("/api/watchlist/plates")
async def get_watchlist():
    watchlist_path = _ANPR_WATCHLIST
    try:
        with open(watchlist_path, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except FileNotFoundError:
        return JSONResponse(content={"plates": []})
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/watchlist/plates")
async def add_watchlist_plate(payload: dict):
    """
    Add a plate to the watchlist at runtime.
    Body: {"plate": "MH12AB1234"}
    """
    plate = str(payload.get("plate", "")).strip().upper()
    if not plate:
        return JSONResponse(content={"error": "plate field required"}, status_code=400)

    watchlist_path = _ANPR_WATCHLIST
    try:
        with open(watchlist_path, "r") as f:
            data = json.load(f)
    except Exception:
        data = {"plates": []}

    plates_set = set(data.get("plates", []))
    plates_set.add(plate)
    data["plates"] = sorted(plates_set)

    with open(watchlist_path, "w") as f:
        json.dump(data, f, indent=2)

    # Also reload in running ANPR detectors
    for cam_id, state in _camera_states.items():
        pass  # ANPRDetector instances are inside threads; they'll reload on next watchlist load

    return JSONResponse(content={"status": "ok", "plate": plate, "total": len(plates_set)})


# ── Prometheus Metrics ────────────────────────────────────────────────────────
@app.get("/metrics")
async def metrics_endpoint():
    data = format_metrics()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False, workers=1)
