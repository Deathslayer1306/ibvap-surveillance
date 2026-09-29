"""
Central configuration for the Suspicious Activity Detection system.
Designed for drone deployment — all tunable params in one place.
"""
import os

# ─── Model Paths ──────────────────────────────────────────────────────────────
MODELS_DIR = "models"

MODEL_FACE     = os.path.join(MODELS_DIR, "yolov8n-face.onnx")
MODEL_POSE     = os.path.join(MODELS_DIR, "yolo11n-pose.onnx")
MODEL_WEAPON   = os.path.join(MODELS_DIR, "weapon.onnx")
MODEL_EMOTION  = os.path.join(MODELS_DIR, "emotion_better.onnx")

# ─── GPU / ONNX ───────────────────────────────────────────────────────────────
GPU_DEVICE_ID      = 0
GPU_MEM_LIMIT_GB   = 4          # RTX 4050 6GB — reserve headroom
CUDA_MEM_LIMIT     = GPU_MEM_LIMIT_GB * 1024 ** 3


def _build_providers():
    """
    Returns GPU providers if CUDA + cuDNN are available,
    otherwise falls back to CPU-only gracefully.
    """
    import onnxruntime as ort
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": GPU_DEVICE_ID,
                    "gpu_mem_limit": CUDA_MEM_LIMIT,
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
    print("[Config] CUDA not available — using CPU.")
    return ["CPUExecutionProvider"]


ONNX_PROVIDERS = _build_providers()

# ─── Camera Mode ──────────────────────────────────────────────────────────────
# Switch between input sources by changing CAMERA_MODE to one of:
#
#   "LOCAL"       — Laptop/desktop webcam (cv2.VideoCapture(0))
#                   Use this for offline testing when RPi is unavailable.
#
#   "RPI_SOCKET"  — RPi streams frames over a raw TCP socket using
#                   pickle + struct framing (see rpi_sender.py on the RPi side).
#                   Set RPI_HOST to the RPi's IP address.
#                   Requires: SSH tunnel OR both devices on the same LAN.
#                   SSH tunnel example (run on laptop before starting):
#                     ssh -L 8080:localhost:8080 pi@<RPI_IP>
#                   Then set RPI_HOST = "127.0.0.1" below.
#
#   "RPI_HTTP"    — RPi serves an MJPEG/RTSP stream over HTTP.
#                   Set RPI_CAMERA_URL to the full stream URL.
#                   Example (libcamera-vid / motion / mjpg-streamer):
#                     http://192.168.1.42:8080/?action=stream
#
# ──────────────────────────────────────────────────────────────────────────────
# ↓↓↓  UNCOMMENT ONE LINE BELOW TO SELECT YOUR CAMERA SOURCE  ↓↓↓

#CAMERA_MODE = "RPI_SOCKET"   # ← RPi Pi Camera over TCP socket (rpi_sender.py)
CAMERA_MODE = "LOCAL"      # ← Laptop/desktop webcam (no RPi needed)
# CAMERA_MODE = "RPI_HTTP"   # ← RPi MJPEG/RTSP HTTP stream

# ↑↑↑  UNCOMMENT ONE LINE ABOVE TO SELECT YOUR CAMERA SOURCE  ↑↑↑
# ──────────────────────────────────────────────────────────────────────────────

# ─── Camera — LOCAL mode ──────────────────────────────────────────────────────
# Index of the local webcam. 0 = default, 1 = second camera, etc.
# Only used when CAMERA_MODE = "LOCAL".
LOCAL_CAMERA_INDEX = 0

# ─── Camera — RPI_SOCKET mode ─────────────────────────────────────────────────
# IP address of the Raspberry Pi (or 127.0.0.1 if using an SSH tunnel).
# Only used when CAMERA_MODE = "RPI_SOCKET".
RPI_HOST = "10.156.86.179"        # ← set to RPi's LAN IP (or 127.0.0.1 for SSH tunnel)
RPI_PORT = 8080                 # must match PORT in rpi_sender.py

# ─── Camera — RPI_HTTP mode ───────────────────────────────────────────────────
# Full URL of the MJPEG / RTSP stream served by the RPi.
# Only used when CAMERA_MODE = "RPI_HTTP".
RPI_CAMERA_URL = f"http://{RPI_HOST}:8080/?action=stream"   # mjpg-streamer default
# RPI_CAMERA_URL = f"rtsp://{RPI_HOST}:8554/stream"         # libcamera-vid RTSP

# ─── Derived CAMERA_SOURCE (used by camera_thread) ────────────────────────────
# Resolved automatically from CAMERA_MODE — do not edit this block.
if CAMERA_MODE == "LOCAL":
    _raw_source = os.environ.get("CAMERA_SOURCE", str(LOCAL_CAMERA_INDEX))
    try:
        CAMERA_SOURCE = int(_raw_source)
    except ValueError:
        CAMERA_SOURCE = _raw_source
elif CAMERA_MODE == "RPI_HTTP":
    CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", RPI_CAMERA_URL)
else:
    # RPI_SOCKET — camera_thread handles the socket directly; CAMERA_SOURCE unused
    CAMERA_SOURCE = None

# Network stream reconnect settings (RPI_HTTP / RPI_SOCKET)
CAMERA_RECONNECT_DELAY = 2.0   # seconds to wait before retrying a dropped stream
CAMERA_MAX_RETRIES     = 0     # 0 = retry forever

CAMERA_WIDTH   = 1280
CAMERA_HEIGHT  = 720
CAMERA_FPS     = 30

# ─── Inference Intervals (run every N inference-loop iterations) ───────────────
# Inference loop runs as fast as it can; these control sub-sampling
# OPT-05: raised intervals — last-known result is reused between skipped frames,
# so the display stays smooth while inference load drops significantly.
POSE_EVERY    = 4    # pose every 4th frame    (was 2)
WEAPON_EVERY  = 6    # weapon every 6th frame  (was 4)
EMOTION_EVERY = 10   # emotion every 10th      (was 5, 2-stage pipeline is heaviest)

# ─── Detection Thresholds ─────────────────────────────────────────────────────
PERSON_CONF        = 0.35
FACE_CONF          = 0.40
WEAPON_CONF        = 0.45
POSE_CONF          = 0.40   # minimum keypoint confidence for inclusion
EMOTION_INPUT_SIZE = (224, 224)

# ─── Pose anomaly min-confidence (only flag if skeleton keypoints are reliable) ─
POSE_MIN_KP_VISIBILITY = 0.50  # raised hands: both wrists must be ≥ this confident

# ─── Threat Scoring ───────────────────────────────────────────────────────────
SCORE_WEAPON             = 5
SCORE_EMOTION_ANGRY      = 2
SCORE_EMOTION_FEAR       = 2
SCORE_POSE_AGGRESSIVE    = 4
SCORE_POSE_RAISED_HANDS  = 3
SCORE_POSE_CROUCHING     = 1

ALERT_WARNING_THRESHOLD  = 4
ALERT_CRITICAL_THRESHOLD = 7

# ─── Data Paths ───────────────────────────────────────────────────────────────
WATCHLIST_DIR = "data/watchlist"
FACES_DIR     = "data/faces"
DB_DIR        = "data/databases"

# ─── Server ───────────────────────────────────────────────────────────────────
HOST          = "0.0.0.0"
PORT          = 8000
JPEG_QUALITY  = 85    # snapshot/event quality (kept high for evidence)

# ─── Live Stream (OPT-06) ─────────────────────────────────────────────────────
# Stream is downscaled before JPEG encode — quarter pixel count vs 1080p.
# Lower quality is fine for live surveillance; high quality is only for snapshots.
STREAM_WIDTH   = 960
STREAM_HEIGHT  = 540
STREAM_QUALITY = 60   # live MJPEG encode quality (60 = fast, visually sufficient)
