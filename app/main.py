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
    CAMERA_MODE, CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    CAMERA_RECONNECT_DELAY, CAMERA_MAX_RETRIES,
    RPI_HOST, RPI_PORT,
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
latest_annotated_frame: np.ndarray | None  = None  # kept for compat; not used by stream
latest_threat:          ThreatState        = ThreatState()
frame_lock   = threading.Lock()
annotated_lock = threading.Lock()
threat_lock  = threading.Lock()

# ── Stream overlay: inference writes a small JPEG overlay; stream thread merges it ──
# Instead of blocking the stream on inference, we keep the latest annotated
# frame separately so the MJPEG generator can serve raw+overlay at full cam FPS.
latest_stream_frame: np.ndarray | None = None
stream_lock = threading.Lock()

_stop_event  = threading.Event()
_ws_loop: asyncio.AbstractEventLoop | None = None

ws_clients: Set[WebSocket] = set()

# ── Perf counters ─────────────────────────────────────────────────────────────
_inf_fps  = 0.0
_cam_fps  = 0.0
_last_panel_bc = 0.0   # last time we pushed a WS panel-update


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
    """
    Captures frames from the configured source (RPi network stream or local webcam).

    For network sources (HTTP MJPEG / RTSP) the thread will automatically
    reconnect whenever the stream is lost — e.g. when the RPi reboots or
    WiFi drops.  CAMERA_MAX_RETRIES=0 means retry forever.
    """
    global latest_raw_frame, _cam_fps

    is_network = isinstance(CAMERA_SOURCE, str)   # URL → network; int → local webcam
    source_desc = CAMERA_SOURCE if is_network else f"local webcam (index {CAMERA_SOURCE})"
    print(f"[CameraThread] Source: {source_desc}")

    attempt = 0

    while not _stop_event.is_set():
        attempt += 1
        print(f"[CameraThread] Connecting (attempt {attempt}) …")

        cap = cv2.VideoCapture(CAMERA_SOURCE)

        # Tuning hints for network streams
        if is_network:
            # Keep the decode buffer as small as possible so we always get the
            # latest frame rather than a buffered (stale) one.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Resolution is set by the RPi streamer; don't override for MJPEG.
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not cap.isOpened():
            print(f"[CameraThread] ERROR: Cannot open stream at {source_desc}")
            cap.release()
            if CAMERA_MAX_RETRIES and attempt >= CAMERA_MAX_RETRIES:
                print("[CameraThread] Max retries reached. Stopping.")
                return
            print(f"[CameraThread] Retrying in {CAMERA_RECONNECT_DELAY}s …")
            time.sleep(CAMERA_RECONNECT_DELAY)
            continue

        h_actual = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or CAMERA_HEIGHT
        w_actual = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or CAMERA_WIDTH
        print(f"[CameraThread] Stream opened — {w_actual}x{h_actual} from {source_desc}")
        attempt = 0   # reset retry counter on successful open

        t0, frames, consecutive_fails = time.perf_counter(), 0, 0

        while not _stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                consecutive_fails += 1
                if consecutive_fails >= 10:
                    # Stream appears genuinely lost — break inner loop to reconnect
                    print(f"[CameraThread] Stream lost ({consecutive_fails} consecutive failures). Reconnecting …")
                    break
                time.sleep(0.005)
                continue

            consecutive_fails = 0   # reset on good frame
            with frame_lock:
                latest_raw_frame = frame
            frames += 1
            elapsed = time.perf_counter() - t0
            if elapsed >= 2.0:
                _cam_fps = frames / elapsed
                frames, t0 = 0, time.perf_counter()

        cap.release()
        if _stop_event.is_set():
            break
        print(f"[CameraThread] Waiting {CAMERA_RECONNECT_DELAY}s before reconnect …")
        time.sleep(CAMERA_RECONNECT_DELAY)

    print("[CameraThread] Stopped.")


# ── RPi Socket Camera Thread ─────────────────────────────────────────────────────────────
def rpi_socket_camera_thread() -> None:
    """
    Receives frames streamed from the Raspberry Pi over a TCP socket.

    The RPi side runs rpi_sender.py which connects to this laptop's
    IP and pushes frames encoded with pickle + struct length-prefix framing.

    Protocol (matches rpi_sender.py):
        [4-byte unsigned long: payload size] [pickle.dumps(frame)]

    Reconnect logic mirrors camera_thread() so a dropped WiFi/SSH connection
    is recovered automatically (CAMERA_RECONNECT_DELAY seconds between tries).

    To use this mode, set in config.py:
        CAMERA_MODE = "RPI_SOCKET"
        RPI_HOST    = "0.0.0.0"   # listen on all interfaces (server)

    SSH tunnel alternative (no direct LAN access):
        On the RPi: ssh -R 8080:localhost:8080 <LAPTOP_USER>@<LAPTOP_IP>
        Then on the RPi run rpi_sender.py targeting 127.0.0.1:8080.
    """
    global latest_raw_frame, _cam_fps

    import socket as _socket
    import struct as _struct
    import pickle as _pickle

    PAYLOAD_FMT  = "!I"                         # network byte order, always 4 bytes (matches rpi_sender.py)
    PAYLOAD_SIZE = _struct.calcsize(PAYLOAD_FMT)
    RECV_CHUNK   = 4096

    # ─────────────────────────────────────────────────────────────────────────────
    # The laptop acts as SERVER — the RPi connects outward (NAT-friendly).
    # If you use an SSH *local* tunnel instead, set RPI_HOST = "127.0.0.1"
    # and leave this code unchanged; the tunnel makes it look local.
    # ─────────────────────────────────────────────────────────────────────────────
    listen_host = "0.0.0.0"   # accept from any interface
    listen_port = RPI_PORT
    attempt = 0

    while not _stop_event.is_set():
        attempt += 1
        srv_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        srv_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        try:
            srv_sock.bind((listen_host, listen_port))
        except OSError as e:
            print(f"[RPiSocket] Cannot bind {listen_host}:{listen_port} — {e}")
            srv_sock.close()
            time.sleep(CAMERA_RECONNECT_DELAY)
            continue

        srv_sock.listen(1)
        srv_sock.settimeout(5.0)   # so _stop_event is checked periodically
        print(f"[RPiSocket] Waiting for RPi to connect on port {listen_port} … (attempt {attempt})")

        try:
            conn, addr = srv_sock.accept()
        except _socket.timeout:
            srv_sock.close()
            continue   # re-check _stop_event and retry
        except Exception as e:
            print(f"[RPiSocket] Accept error: {e}")
            srv_sock.close()
            time.sleep(CAMERA_RECONNECT_DELAY)
            continue

        print(f"[RPiSocket] RPi connected from {addr}")
        attempt = 0   # reset retry counter
        data = b""
        t0, frames = time.perf_counter(), 0
        conn.settimeout(10.0)   # detect a silently dropped connection

        try:
            while not _stop_event.is_set():
                # ── 1. Read the 4-byte length prefix ────────────────────────────────
                while len(data) < PAYLOAD_SIZE:
                    chunk = conn.recv(RECV_CHUNK)
                    if not chunk:        # connection closed by RPi
                        raise ConnectionResetError("RPi disconnected (EOF)")
                    data += chunk

                packed_size = data[:PAYLOAD_SIZE]
                data        = data[PAYLOAD_SIZE:]
                msg_size    = _struct.unpack(PAYLOAD_FMT, packed_size)[0]

                # ── 2. Read the full pickled frame ────────────────────────────────
                while len(data) < msg_size:
                    chunk = conn.recv(RECV_CHUNK)
                    if not chunk:
                        raise ConnectionResetError("RPi disconnected mid-frame")
                    data += chunk

                frame_data = data[:msg_size]
                data       = data[msg_size:]

                # ── 3. Deserialize & fix colour order ────────────────────────
                # Picamera2 captures RGB888; OpenCV pipeline expects BGR.
                # Without this conversion the frame encodes wrong and looks
                # blank / heavily distorted on the dashboard.
                frame = _pickle.loads(frame_data)
                if frame is not None and frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                with frame_lock:
                    latest_raw_frame = frame

                frames += 1
                if frames % 30 == 0:   # debug: confirm frames are arriving
                    print(f"[RPiSocket] Receiving OK — {frames} frames so far")
                elapsed = time.perf_counter() - t0
                if elapsed >= 2.0:
                    _cam_fps = frames / elapsed
                    frames, t0 = 0, time.perf_counter()

        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            print(f"[RPiSocket] Connection lost: {e}")
        finally:
            conn.close()
            srv_sock.close()

        if _stop_event.is_set():
            break
        print(f"[RPiSocket] Waiting {CAMERA_RECONNECT_DELAY}s before listening again …")
        time.sleep(CAMERA_RECONNECT_DELAY)

    print("[RPiSocket] Stopped.")


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

        # ── Push annotated frame to stream ─────────────────────────────────
        # Stream thread reads this directly — no inference blocking
        with stream_lock:
            latest_stream_frame = annotated

        # ── WS broadcast ───────────────────────────────────────────────────
        # Always broadcast current state so dashboard panels (gauge, emotion,
        # weapon, pose) stay live every N seconds regardless of threat level.
        # Alert-timeline dedup is handled purely on the JS side.
        now = time.perf_counter()
        is_suspicious = state.alert_level in ("WARNING", "CRITICAL")
        cooldown_passed = (now - last_broadcast) >= WS_COOLDOWN

        # Send state update every second so all panels refresh continuously
        global _last_panel_bc
        if (now - _last_panel_bc) >= 1.0:
            _broadcast_sync(state.to_json())
            _last_panel_bc = now

        # For the alert timeline, also fire with 5-s cooldown on WARNING/CRITICAL
        if is_suspicious and cooldown_passed:
            last_broadcast = now
            last_alert_level = state.alert_level
        elif not is_suspicious and last_alert_level != "INFO":
            # One final "cleared" update so JS knows threat dropped
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
    Serves the latest stream frame (annotated by inference) at full camera FPS.
    Decoupled from inference: if inference is slow, the stream still flows at
    camera speed using the most recently annotated frame available.
    """
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    # Lower JPEG quality slightly on CPU for speed (still looks great)
    quality = min(JPEG_QUALITY, 75)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    frame_interval = 1.0 / 30.0   # target 30 FPS; actual cam FPS is the cap
    last_t = time.perf_counter()

    while True:
        # Prefer annotated stream frame; fall back to raw camera frame
        with stream_lock:
            frame = latest_stream_frame
        if frame is None:
            with frame_lock:
                frame = latest_raw_frame

        if frame is None:
            time.sleep(0.02)
            continue

        ret, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            time.sleep(0.01)
            continue

        yield boundary + buf.tobytes() + b"\r\n"

        # Pace output to target FPS without busy-spinning
        now = time.perf_counter()
        sleep_t = frame_interval - (now - last_t)
        if sleep_t > 0:
            time.sleep(sleep_t)
        last_t = time.perf_counter()


# ── FastAPI ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ws_loop
    _ws_loop = asyncio.get_event_loop()

    # ───────────────────────────────────────────────────────────────────────────
    # Camera source is selected by CAMERA_MODE in config.py:
    #   "LOCAL"      → camera_thread()            (local webcam via OpenCV)
    #   "RPI_SOCKET" → rpi_socket_camera_thread() (RPi TCP pickle stream)
    #   "RPI_HTTP"   → camera_thread()            (MJPEG/RTSP URL via OpenCV)
    # ───────────────────────────────────────────────────────────────────────────
    if CAMERA_MODE == "RPI_SOCKET":
        cam_target = rpi_socket_camera_thread
    else:   # "LOCAL" or "RPI_HTTP"
        cam_target = camera_thread

    cam_t = threading.Thread(target=cam_target,   daemon=True, name="CameraThread")
    inf_t = threading.Thread(target=inference_thread, daemon=True, name="InferenceThread")
    cam_t.start()
    inf_t.start()
    print(f"[FastAPI] Started (camera mode: {CAMERA_MODE}). Dashboard -> http://localhost:{PORT}")
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
