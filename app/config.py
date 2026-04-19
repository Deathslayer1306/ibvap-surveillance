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

# ─── Camera ───────────────────────────────────────────────────────────────────
# For drone: set CAMERA_SOURCE env var to RTSP URL, e.g. rtsp://192.168.1.1/stream
CAMERA_SOURCE  = os.environ.get("CAMERA_SOURCE", "0")
try:
    CAMERA_SOURCE = int(CAMERA_SOURCE)
except ValueError:
    pass   # keep as string (RTSP URL)

CAMERA_WIDTH   = 1280
CAMERA_HEIGHT  = 720
CAMERA_FPS     = 30

# ─── Inference Intervals (run every N inference-loop iterations) ───────────────
# Inference loop runs as fast as it can; these control sub-sampling
POSE_EVERY    = 2    # pose every 2nd inference loop iteration
WEAPON_EVERY  = 4    # weapon every 4th
EMOTION_EVERY = 5    # emotion every 5th (heaviest: 2-stage)

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
JPEG_QUALITY  = 80    # 80 = good quality + fast encode (vs 95 = slow)
