"""
Emotion Detector — GPU-accelerated via ONNX Runtime CUDAExecutionProvider.
Classifies emotion from a face crop using emotion_better.onnx.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import cv2
from dataclasses import dataclass, field
from typing import Optional, Dict

from app.config import MODEL_EMOTION, ONNX_PROVIDERS, EMOTION_INPUT_SIZE


# FER-2013 / AffectNet standard 7-class emotion labels
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Emotion → color (BGR) for annotation
EMOTION_COLORS: Dict[str, tuple] = {
    "angry":    (0,   0,   255),
    "disgust":  (0,   128, 128),
    "fear":     (128, 0,   128),
    "happy":    (0,   255, 128),
    "neutral":  (200, 200, 200),
    "sad":      (255, 128, 0),
    "surprise": (0,   200, 255),
}


@dataclass
class EmotionResult:
    label: str = "neutral"
    confidence: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    face_box: Optional[list] = None   # [x1,y1,x2,y2] in frame coords


class EmotionDetector:
    """
    Two-stage pipeline:
      1. Face detection with yolov8n-face.onnx  (GPU)
      2. Emotion classification with emotion_better.onnx (GPU)
    """

    FACE_INPUT_SIZE = (640, 640)

    def __init__(self) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.log_severity_level = 3   # suppress opset warnings

        # Face detector session
        from app.config import MODEL_FACE
        try:
            self.face_session = ort.InferenceSession(
                MODEL_FACE,
                sess_options=opts,
                providers=ONNX_PROVIDERS,
            )
        except Exception:
            print("[EmotionDetector] Face GPU load failed, falling back to CPU")
            self.face_session = ort.InferenceSession(
                MODEL_FACE,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        self.face_input  = self.face_session.get_inputs()[0].name

        # Emotion classifier session
        try:
            self.emotion_session = ort.InferenceSession(
                MODEL_EMOTION,
                sess_options=opts,
                providers=ONNX_PROVIDERS,
            )
        except Exception:
            print("[EmotionDetector] Emotion GPU load failed, falling back to CPU")
            self.emotion_session = ort.InferenceSession(
                MODEL_EMOTION,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        self.emotion_input  = self.emotion_session.get_inputs()[0].name
        self.emotion_output = self.emotion_session.get_outputs()[0].name

        active = self.emotion_session.get_providers()[0]
        print(f"[EmotionDetector] Provider: {active}")

    # ── face detection ────────────────────────────────────────────────────────
    def _detect_faces(self, frame: np.ndarray, conf_thresh: float = 0.30):
        h, w = frame.shape[:2]
        img = cv2.resize(frame, self.FACE_INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.expand_dims(img.transpose(2, 0, 1), 0)

        raw = self.face_session.run(None, {self.face_input: blob})[0][0]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T

        sx, sy = w / 640.0, h / 640.0
        faces = []
        for row in raw:
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            conf = float(row[4]) if row.shape[0] > 4 else 1.0
            if conf < conf_thresh:
                continue
            x1 = max(0, int((cx - bw / 2) * sx))
            y1 = max(0, int((cy - bh / 2) * sy))
            x2 = min(w, int((cx + bw / 2) * sx))
            y2 = min(h, int((cy + bh / 2) * sy))
            if x2 > x1 and y2 > y1:
                faces.append([x1, y1, x2, y2, conf])
        return faces

    # ── emotion classification ────────────────────────────────────────────────
    def _classify_emotion(self, face_crop: np.ndarray) -> EmotionResult:
        img = cv2.resize(face_crop, EMOTION_INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.expand_dims(img, 0)   # NHWC or NCHW handled below

        # Try NHWC first; if model expects NCHW this fails gracefully
        try:
            raw = self.emotion_session.run(
                [self.emotion_output], {self.emotion_input: blob}
            )[0][0]
        except Exception:
            blob_chw = np.expand_dims(img.transpose(2, 0, 1), 0)
            raw = self.emotion_session.run(
                [self.emotion_output], {self.emotion_input: blob_chw}
            )[0][0]

        # Softmax
        raw = raw - raw.max()
        exp  = np.exp(raw)
        probs = exp / exp.sum()

        idx   = int(np.argmax(probs))
        label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else f"class_{idx}"
        scores = {
            EMOTION_LABELS[i]: round(float(probs[i]), 3)
            for i in range(min(len(EMOTION_LABELS), len(probs)))
        }
        return EmotionResult(label=label, confidence=round(float(probs[idx]), 3), scores=scores)

    # ── blur check ────────────────────────────────────────────────────────────
    @staticmethod
    def _is_blurry(img: np.ndarray, threshold: float = 80.0) -> bool:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

    # ── public API ────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[EmotionResult]:
        """Return list of EmotionResult, one per detected face."""
        results = []
        face_boxes = self._detect_faces(frame)

        for x1, y1, x2, y2, _ in face_boxes:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or self._is_blurry(crop):
                continue
            res = self._classify_emotion(crop)
            res.face_box = [x1, y1, x2, y2]
            results.append(res)

        return results

    def annotate(self, frame: np.ndarray, results: list[EmotionResult]) -> np.ndarray:
        for res in results:
            if res.face_box is None:
                continue
            x1, y1, x2, y2 = res.face_box
            color = EMOTION_COLORS.get(res.label, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{res.label} {res.confidence:.0%}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
        return frame
