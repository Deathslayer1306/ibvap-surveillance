"""
Vehicle Detector — GPU-accelerated via ONNX Runtime
====================================================
Uses a YOLOv8n COCO model (80 classes) to detect vehicles.
Falls back to weapon.onnx if no dedicated vehicle model is found
(weapon.onnx is a YOLOv11 model — COCO vehicle classes will NOT be in it;
 in that case vehicle detection is simply disabled, never crashed).

COCO class IDs for vehicles:
  1  → bicycle
  2  → car
  3  → motorcycle
  5  → bus
  7  → truck

Returns
-------
VehicleResult(vehicles=[VehicleDetection(...)])
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    logger.warning("[VehicleDetector] onnxruntime not found — vehicle detection disabled.")


# ── COCO vehicle class mapping ────────────────────────────────────────────────
# Only these class IDs are treated as vehicles.
VEHICLE_CLASS_IDS: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# For threat scoring: large vehicles near the perimeter
LARGE_VEHICLE_TYPES = {"bus", "truck"}

# Display emoji per type
VEHICLE_ICONS: dict[str, str] = {
    "bicycle":    "🚲",
    "car":        "🚗",
    "motorcycle": "🏍",
    "bus":        "🚌",
    "truck":      "🚛",
}

# Colours (BGR) per type
VEHICLE_COLORS: dict[str, tuple[int, int, int]] = {
    "bicycle":    (255, 165, 0),
    "car":        (0, 200, 255),
    "motorcycle": (0, 255, 150),
    "bus":        (0, 140, 255),
    "truck":      (0, 60, 255),
}

# Well-known COCO-80 model that works with YOLOv8n
_DEFAULT_COCO_MODEL = "models/yolov8n.onnx"
_INPUT_SIZE = (640, 640)
_DEFAULT_CONF = 0.45
_IOU_THRESH = 0.45


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class VehicleDetection:
    bbox: List[int]         # [x1, y1, x2, y2] in original frame coords
    confidence: float
    vehicle_type: str       # "car" | "truck" | "bus" | "motorcycle" | "bicycle"
    is_large: bool          # True for bus/truck


@dataclass
class VehicleResult:
    vehicles: List[VehicleDetection] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return len(self.vehicles) > 0

    @property
    def counts(self) -> dict[str, int]:
        """Return vehicle type counts."""
        c: dict[str, int] = {}
        for v in self.vehicles:
            c[v.vehicle_type] = c.get(v.vehicle_type, 0) + 1
        return c

    @property
    def has_large_vehicle(self) -> bool:
        return any(v.is_large for v in self.vehicles)

    @property
    def bboxes(self) -> list[list[int]]:
        return [v.bbox for v in self.vehicles]


# ── NMS ───────────────────────────────────────────────────────────────────────

def _nms(boxes: list, scores: list, iou_threshold: float = _IOU_THRESH) -> list[int]:
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou < iou_threshold]
    return keep


# ── Detector class ────────────────────────────────────────────────────────────

class VehicleDetector:
    """
    Detects vehicles from COCO model output.

    Prefers models/yolov8n.onnx (COCO 80-class).
    Falls back gracefully if the model file is missing.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_COCO_MODEL,
        confidence_threshold: float = _DEFAULT_CONF,
        onnx_providers: Optional[list] = None,
    ) -> None:
        self.enabled = False
        self.conf_threshold = confidence_threshold
        self._session: Optional[object] = None
        self._input_name: str = ""
        self._num_classes: int = 80    # COCO default

        if not _ORT_AVAILABLE:
            logger.warning("[VehicleDetector] onnxruntime unavailable — disabled.")
            return

        if not os.path.isfile(model_path):
            logger.warning(
                f"[VehicleDetector] Model not found at '{model_path}'. "
                "Vehicle detection disabled. Download yolov8n.onnx from Ultralytics."
            )
            return

        providers = onnx_providers or ["CPUExecutionProvider"]
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            opts.log_severity_level = 3

            self._session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            out_shape = self._session.get_outputs()[0].shape
            # Output shape heuristic for num_classes: (1, 4+N_cls, anchors)
            if len(out_shape) >= 2 and isinstance(out_shape[1], int):
                self._num_classes = max(1, out_shape[1] - 4)

            active = self._session.get_providers()[0]
            logger.info(f"[VehicleDetector] Model     : {model_path}")
            logger.info(f"[VehicleDetector] Provider  : {active}")
            logger.info(f"[VehicleDetector] Classes   : {self._num_classes}")
            self.enabled = True
        except Exception as exc:
            logger.warning(f"[VehicleDetector] Failed to load model: {exc}")

    # ── Pre-processing ────────────────────────────────────────────────────────
    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = frame.shape[:2]
        iw, ih = _INPUT_SIZE
        img = cv2.resize(frame, (iw, ih))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.expand_dims(img.transpose(2, 0, 1), 0)
        return blob, w / iw, h / ih

    # ── Post-processing ───────────────────────────────────────────────────────
    def _postprocess(
        self, raw: np.ndarray, orig_w: int, orig_h: int, sx: float, sy: float
    ) -> VehicleResult:
        out = raw[0]
        if out.shape[0] < out.shape[1]:
            out = out.T    # (4+cls, anchors) → (anchors, 4+cls)

        boxes_by_cls: dict[int, list] = {cid: [] for cid in VEHICLE_CLASS_IDS}
        scores_by_cls: dict[int, list] = {cid: [] for cid in VEHICLE_CLASS_IDS}

        for row in out:
            if row.shape[0] < 5:
                continue
            cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            class_scores = row[4:]
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])

            if conf < self.conf_threshold:
                continue
            if cls_id not in VEHICLE_CLASS_IDS:
                continue

            x1 = max(0,      int((cx - bw / 2) * sx))
            y1 = max(0,      int((cy - bh / 2) * sy))
            x2 = min(orig_w, int((cx + bw / 2) * sx))
            y2 = min(orig_h, int((cy + bh / 2) * sy))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes_by_cls[cls_id].append([x1, y1, x2, y2])
            scores_by_cls[cls_id].append(conf)

        result = VehicleResult()
        for cid, vtype in VEHICLE_CLASS_IDS.items():
            if not boxes_by_cls[cid]:
                continue
            kept = _nms(boxes_by_cls[cid], scores_by_cls[cid])
            for k in kept:
                result.vehicles.append(VehicleDetection(
                    bbox=boxes_by_cls[cid][k],
                    confidence=round(scores_by_cls[cid][k], 3),
                    vehicle_type=vtype,
                    is_large=(vtype in LARGE_VEHICLE_TYPES),
                ))
        return result

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> VehicleResult:
        """
        Run vehicle detection on a single frame.

        Args:
            frame: BGR numpy array (any resolution).

        Returns:
            VehicleResult. Empty result if disabled or on inference error.
        """
        if not self.enabled or self._session is None:
            return VehicleResult()

        h, w = frame.shape[:2]
        try:
            blob, sx, sy = self._preprocess(frame)
            raw = self._session.run(None, {self._input_name: blob})[0]
            return self._postprocess(raw, w, h, sx, sy)
        except Exception as exc:
            logger.warning(f"[VehicleDetector] Inference error: {exc}")
            return VehicleResult()

    # ── Annotate ─────────────────────────────────────────────────────────────
    def annotate(self, frame: np.ndarray, result: VehicleResult) -> np.ndarray:
        """Draw vehicle bounding boxes with type label on the frame."""
        for v in result.vehicles:
            x1, y1, x2, y2 = v.bbox
            color = VEHICLE_COLORS.get(v.vehicle_type, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            icon = VEHICLE_ICONS.get(v.vehicle_type, "🚗")
            label = f"{v.vehicle_type.upper()} {v.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return frame
