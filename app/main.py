"""
Suspicious Activity Detection — FastAPI Backend (Optimized)
GPU-ready, low-latency, drone-ready.

Architecture:
  CameraThread  →  latest_frame (global)  →  StreamThread (MJPEG)
                         ↓ (sampled)
                   InferenceThread  →  latest_state (global)
                         ↓
                   ThreatAggregator  →  WS broadcast (throttled)
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Set

import cv2
import numpy as np
import uvicorn
import pathlib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

from app.config import (
    CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    HOST, PORT, JPEG_QUALITY,
    POSE_EVERY, WEAPON_EVERY, EMOTION_EVERY,
    WATCHLIST_DIR, FACES_DIR, DB_DIR,
    WEAPON_CONF,
)
from app.detectors.weapon_detector   import WeaponDetector,  WeaponResult
from app.detectors.emotion_detector  import EmotionDetector, EmotionResult
from app.detectors.pose_detector     import PoseDetector,    PersonPose, PoseAnomaly
from app.detectors.threat_aggregator import ThreatAggregator, ThreatState

# ── Directory setup ───────────────────────────────────────────────────────────
for d in [WATCHLIST_DIR, FACES_DIR, DB_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Shared globals (lock-protected or lock-free reads are fine for numpy) ─────
latest_raw_frame:       np.ndarray | None  = None
latest_annotated_frame: np.ndarray | None  = None
latest_threat:          ThreatState        = ThreatState()
frame_lock   = threading.Lock()
annotated_lock = threading.Lock()
threat_lock  = threading.Lock()

_stop_event  = threading.Event()
_ws_loop: asyncio.AbstractEventLoop | None = None

ws_clients: Set[WebSocket] = set()

# ── Perf counters ─────────────────────────────────────────────────────────────
_inf_fps  = 0.0
_cam_fps  = 0.0


# ── WebSocket broadcast ───────────────────────────────────────────────────────
def _broadcast_sync(msg: str) -> None:
    if _ws_loop is None or not ws_clients:
        return
    asyncio.run_coroutine_threadsafe(_broadcast_async(msg), _ws_loop)


async def _broadcast_async(msg: str) -> None:
    dead = set()
    for ws in list(ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


# ── Camera Thread ─────────────────────────────────────────────────────────────
def camera_thread() -> None:
    global latest_raw_frame, _cam_fps

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print("[CameraThread] ERROR: Cannot open camera.")
        return

    print(f"[CameraThread] Camera opened @ {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

    t0, frames = time.perf_counter(), 0
    while not _stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue
        with frame_lock:
            latest_raw_frame = frame
        frames += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= 2.0:
            _cam_fps = frames / elapsed
            frames, t0 = 0, time.perf_counter()

    cap.release()
    print("[CameraThread] Stopped.")


# ── Inference Thread ──────────────────────────────────────────────────────────
def inference_thread() -> None:
    global latest_annotated_frame, latest_threat, _inf_fps

    print("[InferenceThread] Loading models ...")
    pose_det   = PoseDetector()
    weapon_det = WeaponDetector()
    emo_det    = EmotionDetector()
    aggregator = ThreatAggregator()
    print("[InferenceThread] Models loaded.")

    frame_count = 0
    last_weapon: WeaponResult  = WeaponResult()
    last_emos: list            = []
    last_poses: list[PersonPose] = []

    # Rate-limit WS broadcasts — only suspicious events, 5s cooldown
    last_broadcast = 0.0
    WS_COOLDOWN = 5.0          # seconds between alerts
    last_alert_level = "INFO"  # track for deduplication

    t0, inf_frames = time.perf_counter(), 0

    while not _stop_event.is_set():
        # Grab latest raw frame
        with frame_lock:
            raw = latest_raw_frame
        if raw is None:
            time.sleep(0.005)
            continue

        frame_count += 1
        # Work on a copy so camera thread can keep writing
        frame = raw.copy()

        # ── Inference at reduced resolution for speed ──────────────────────
        H, W = frame.shape[:2]
        proc_w, proc_h = 640, 360
        small = cv2.resize(frame, (proc_w, proc_h))
        sx, sy = W / proc_w, H / proc_h

        # ── Weapon detection ───────────────────────────────────────────────
        if frame_count % WEAPON_EVERY == 0:
            try:
                last_weapon = weapon_det.detect(small, conf=WEAPON_CONF)
                # Scale boxes back to original resolution
                scaled_boxes = []
                for box in last_weapon.boxes:
                    scaled_boxes.append([
                        int(box[0] * sx), int(box[1] * sy),
                        int(box[2] * sx), int(box[3] * sy),
                    ])
                last_weapon.boxes = scaled_boxes
            except Exception as e:
                print(f"[Weapon] {e}")

        # ── Pose detection ─────────────────────────────────────────────────
        if frame_count % POSE_EVERY == 0:
            try:
                last_poses = pose_det.detect(small)
                # Scale poses back to full resolution
                for p in last_poses:
                    p.keypoints[:, 0] *= sx
                    p.keypoints[:, 1] *= sy
                    p.bbox = [
                        int(p.bbox[0] * sx), int(p.bbox[1] * sy),
                        int(p.bbox[2] * sx), int(p.bbox[3] * sy),
                    ]
            except Exception as e:
                print(f"[Pose] {e}")

        # ── Emotion detection ──────────────────────────────────────────────
        if frame_count % EMOTION_EVERY == 0:
            try:
                last_emos = emo_det.detect(small)
                # Scale face boxes back
                for er in last_emos:
                    if er.face_box:
                        er.face_box = [
                            int(er.face_box[0] * sx), int(er.face_box[1] * sy),
                            int(er.face_box[2] * sx), int(er.face_box[3] * sy),
                        ]
            except Exception as e:
                print(f"[Emotion] {e}")

        # ── Annotate full-resolution frame ─────────────────────────────────
        annotated = frame.copy()

        # Draw pose skeleton
        annotated = PoseDetector.annotate(annotated, last_poses)
        # Draw weapon boxes
        annotated = weapon_det.annotate(annotated, last_weapon)
        # Draw emotion labels
        annotated = emo_det.annotate(annotated, last_emos)

        # ── Threat state ───────────────────────────────────────────────────
        top_emo    = last_emos[0].label  if last_emos else "neutral"
        top_scores = last_emos[0].scores if last_emos else {}
        pose_labels = [p.anomaly.value for p in last_poses]

        state = aggregator.update(
            weapon_detected  = last_weapon.detected,
            weapon_labels    = last_weapon.labels,
            weapon_display   = last_weapon.display_labels,
            emotion          = top_emo,
            emotion_scores   = top_scores,
            pose_anomalies   = pose_labels,
            num_persons      = len(last_poses),
        )

        with threat_lock:
            latest_threat = state

        # ── HUD overlay ────────────────────────────────────────────────────
        _draw_hud(annotated, state, _inf_fps)

        # ── Push annotated frame ───────────────────────────────────────────
        with annotated_lock:
            latest_annotated_frame = annotated

        # ── WS broadcast — only WARNING/CRITICAL, 5s cooldown ────────────────
        now = time.perf_counter()
        is_suspicious = state.alert_level in ("WARNING", "CRITICAL")
        cooldown_passed = (now - last_broadcast) >= WS_COOLDOWN

        if is_suspicious and cooldown_passed:
            _broadcast_sync(state.to_json())
            last_broadcast = now
            last_alert_level = state.alert_level
        elif not is_suspicious and last_alert_level != "INFO":
            # Send one final "cleared" update when threat drops back to INFO
            _broadcast_sync(state.to_json())
            last_alert_level = "INFO"

        # ── Save on CRITICAL ───────────────────────────────────────────────
        if state.alert_level == "CRITICAL":
            _maybe_save(frame, state)

        # ── FPS counter ────────────────────────────────────────────────────
        inf_frames += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= 2.0:
            _inf_fps = inf_frames / elapsed
            inf_frames, t0 = 0, time.perf_counter()

    print("[InferenceThread] Stopped.")


# ── HUD overlay ───────────────────────────────────────────────────────────────
def _draw_hud(frame: np.ndarray, state: ThreatState, fps: float) -> None:
    h, w = frame.shape[:2]
    alert_colors = {
        "INFO":     (0, 200, 80),
        "WARNING":  (0, 165, 255),
        "CRITICAL": (0, 0, 255),
    }
    color = alert_colors.get(state.alert_level, (200, 200, 200))

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (8, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"[{state.alert_level}]",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"THREAT:{state.threat_score}/10",
                (160, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"EMO:{state.emotion.upper()}",
                (340, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 255), 2)

    w_txt = "WPN:YES" if state.weapon_detected else "WPN:NO"
    w_col = (0, 0, 255) if state.weapon_detected else (80, 200, 80)
    cv2.putText(frame, w_txt, (530, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, w_col, 2)

    pose_short = state.pose_anomaly[:10].upper()
    cv2.putText(frame, f"POSE:{pose_short}",
                (670, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 50), 2)

    fps_str = f"{fps:.1f}FPS"
    cv2.putText(frame, fps_str, (w - 90, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)


_last_saved: dict = {}


def _maybe_save(frame: np.ndarray, state: ThreatState) -> None:
    ts_key = state.timestamp[:16]
    if _last_saved.get("ts") == ts_key:
        return
    _last_saved["ts"] = ts_key
    ts = ts_key.replace(":", "-").replace("T", "_")
    folder = os.path.join(WATCHLIST_DIR, f"CRITICAL_{ts}")
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, "scene.jpg"), frame)


# ── MJPEG Stream ──────────────────────────────────────────────────────────────
def _mjpeg_gen():
    """
    Reads the latest annotated frame and yields MJPEG chunks.
    Does NOT block on inference — always serves the most recent frame.
    """
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        with annotated_lock:
            frame = latest_annotated_frame

        if frame is None:
            # No frame yet — yield a tiny black placeholder
            time.sleep(0.03)
            continue

        ret, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            time.sleep(0.01)
            continue

        yield boundary + buf.tobytes() + b"\r\n"
        # Target ~30 FPS display (33ms between frames)
        time.sleep(0.033)


# ── FastAPI ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ws_loop
    _ws_loop = asyncio.get_event_loop()

    cam_t = threading.Thread(target=camera_thread,   daemon=True, name="CameraThread")
    inf_t = threading.Thread(target=inference_thread, daemon=True, name="InferenceThread")
    cam_t.start()
    inf_t.start()
    print("[FastAPI] Started. Dashboard -> http://localhost:8000")
    yield
    _stop_event.set()
    cam_t.join(timeout=3)
    inf_t.join(timeout=3)
    print("[FastAPI] Shut down.")


app = FastAPI(title="Suspicious Activity Detection", lifespan=lifespan)

_HTML_PATH = pathlib.Path("templates/index.html")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
async def api_status():
    import json
    with threat_lock:
        state = latest_threat
    return json.loads(state.to_json())


@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        with threat_lock:
            state = latest_threat
        await ws.send_text(state.to_json())
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


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False, workers=1)
