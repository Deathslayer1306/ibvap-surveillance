"""
Weapon Detector — GPU-accelerated via ONNX Runtime.
Handles YOLOv11 output format (batch=1, 6, anchors) with proper NMS.

Model: weapon.onnx
  - Input : images  [1, 3, 480, 480]
  - Output: output0 [1, 6, anchors]  (4 box coords + 2 class scores)
  - Classes: {0: 'pistol', 1: 'knife'}
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import cv2
from dataclasses import dataclass, field
from typing import List

from app.config import MODEL_WEAPON, ONNX_PROVIDERS, WEAPON_CONF


# ── Class definitions ─────────────────────────────────────────────────────────
# Actual classes from model metadata: {0: 'pistol', 1: 'knife'}
# Extended display names and icons for the dashboard UI
CLASS_NAMES: List[str] = ["pistol", "knife"]

# Richer display labels shown in bounding-box annotations and dashboard
WEAPON_DISPLAY: dict[str, str] = {
    "pistol": "Handgun / Pistol",
    "knife":  "Blade / Knife",
}

# Emoji icons per weapon class (used in the JS dashboard)
WEAPON_ICONS: dict[str, str] = {
    "pistol": "🔫",
    "knife":  "🗡️",
}

# Threat tier per weapon class (used for colour-coding)
WEAPON_THREAT_TIER: dict[str, str] = {
    "pistol": "CRITICAL",   # firearm = highest
    "knife":  "WARNING",    # blade   = medium
}


@dataclass
class WeaponResult:
    detected: bool = False
    boxes: List[List[int]] = field(default_factory=list)   # [x1, y1, x2, y2]
    confidences: List[float] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    # display_labels carries the human-readable names for the dashboard
    display_labels: List[str] = field(default_factory=list)


class WeaponDetector:
    """
    Runs weapon.onnx (YOLOv11 format) on GPU with proper NMS.

    Output layout: (1, 4+num_classes, num_anchors)
      Each column: [cx, cy, w, h, cls0_score, cls1_score, ...]
      YOLOv8/v11 style — NO separate objectness score.
    """

    INPUT_SIZE = (480, 480)   # from model metadata: imgsz=[480, 480]

    def __init__(self) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.log_severity_level = 3

        try:
            self.session = ort.InferenceSession(
                MODEL_WEAPON, sess_options=opts, providers=ONNX_PROVIDERS,
            )
        except Exception as e:
            print(f"[WeaponDetector] GPU load failed: {e}, using CPU with no optimization")
            opts2 = ort.SessionOptions()
            opts2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            opts2.log_severity_level = 4
            self.session = ort.InferenceSession(
                MODEL_WEAPON, sess_options=opts2, providers=["CPUExecutionProvider"],
            )

        inp = self.session.get_inputs()[0]
        self.input_name  = inp.name
        self.input_shape = inp.shape        # e.g. [batch, 3, 480, 480]

        out = self.session.get_outputs()[0]
        self.output_name = out.name

        print(f"[WeaponDetector] Provider : {self.session.get_providers()[0]}")
        print(f"[WeaponDetector] Input    : {self.input_name} {self.input_shape}")
        print(f"[WeaponDetector] Output   : {self.output_name} {out.shape}")
        print(f"[WeaponDetector] Classes  : {CLASS_NAMES}")

    # ── pre-processing ─────────────────────────────────────────────────────────
    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Resize to 480×480, normalise to [0,1], return NCHW blob + scale factors.
        Scale factors map from model-input coords back to original frame coords.
        """
        h, w = frame.shape[:2]
        iw, ih = self.INPUT_SIZE           # 480, 480
        img = cv2.resize(frame, (iw, ih))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.expand_dims(img.transpose(2, 0, 1), 0)   # NCHW
        sx = w / iw   # scale x: model_pixel → original_pixel
        sy = h / ih   # scale y
        return blob, sx, sy

    # ── NMS ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _nms(boxes: list, scores: list, iou_threshold: float = 0.45) -> list:
        """Pure-numpy NMS — keeps highest-scoring non-overlapping boxes."""
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
            iou   = inter / (areas[i] + areas[rest] - inter + 1e-6)
            order = rest[iou < iou_threshold]
        return keep

    # ── post-processing ────────────────────────────────────────────────────────
    def _postprocess(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
        sx: float,
        sy: float,
        conf_threshold: float,
    ) -> WeaponResult:
        """
        YOLOv8/v11 output layout:
          raw shape: (1, 4+num_classes, num_anchors)  — batch dim already removed upstream

        Steps:
          1. Transpose to (num_anchors, 4+num_classes)
          2. For each anchor: max class score IS the confidence (no separate obj score)
          3. Filter by conf_threshold
          4. Convert cx/cy/w/h → x1/y1/x2/y2 in original-frame pixels
          5. NMS
        """
        out = raw[0]   # remove batch dim → shape (4+num_classes, num_anchors)

        # Ensure shape is (num_anchors, 4+num_classes)
        if out.shape[0] < out.shape[1]:
            out = out.T   # transpose: (4+cls, anchors) → (anchors, 4+cls)

        num_classes = out.shape[1] - 4

        raw_boxes:   list[list[int]] = []
        raw_scores:  list[float]     = []
        raw_cls_ids: list[int]       = []

        for row in out:
            cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            class_scores = row[4:]

            # YOLOv8/v11: max class score IS the confidence
            cls_id = int(np.argmax(class_scores))
            conf   = float(class_scores[cls_id])

            if conf < conf_threshold:
                continue

            # Convert model-space box → original-frame pixels
            x1 = max(0,      int((cx - bw / 2) * sx))
            y1 = max(0,      int((cy - bh / 2) * sy))
            x2 = min(orig_w, int((cx + bw / 2) * sx))
            y2 = min(orig_h, int((cy + bh / 2) * sy))

            if x2 <= x1 or y2 <= y1:
                continue

            raw_boxes.append([x1, y1, x2, y2])
            raw_scores.append(conf)
            raw_cls_ids.append(cls_id)

        # Apply per-class NMS (group by class first, then merge)
        keep_final: list[int] = []
        for cid in range(num_classes):
            idxs = [i for i, c in enumerate(raw_cls_ids) if c == cid]
            if not idxs:
                continue
            cls_boxes  = [raw_boxes[i]  for i in idxs]
            cls_scores = [raw_scores[i] for i in idxs]
            kept = self._nms(cls_boxes, cls_scores, iou_threshold=0.45)
            keep_final.extend([idxs[k] for k in kept])

        result = WeaponResult()
        for i in keep_final:
            result.boxes.append(raw_boxes[i])
            result.confidences.append(round(raw_scores[i], 3))
            cls_id = raw_cls_ids[i]
            label  = (CLASS_NAMES[cls_id]
                      if cls_id < len(CLASS_NAMES)
                      else f"weapon_{cls_id}")
            result.labels.append(label)
            result.display_labels.append(
                WEAPON_DISPLAY.get(label, label.replace("_", " ").title())
            )

        result.detected = len(result.boxes) > 0
        return result

    # ── public API ─────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray, conf: float = WEAPON_CONF) -> WeaponResult:
        """
        Run weapon detection on a single frame.

        Args:
            frame: BGR numpy array (any resolution — internally resized to 480×480).
            conf:  Minimum class confidence score (default: WEAPON_CONF from config).

        Returns:
            WeaponResult with boxes, labels, confidences in original-frame coordinates.
        """
        h, w = frame.shape[:2]
        blob, sx, sy = self._preprocess(frame)
        raw = self.session.run(None, {self.input_name: blob})[0]
        return self._postprocess(raw, w, h, sx, sy, conf)

    def annotate(self, frame: np.ndarray, result: WeaponResult) -> np.ndarray:
        """Draw weapon bounding boxes with label and confidence on the frame."""
        for (x1, y1, x2, y2), label, disp, conf in zip(
            result.boxes, result.labels, result.display_labels, result.confidences
        ):
            # Colour per threat tier
            tier = WEAPON_THREAT_TIER.get(label, "WARNING")
            color = (0, 0, 255) if tier == "CRITICAL" else (0, 120, 255)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Label text
            icon = WEAPON_ICONS.get(label, "⚠")
            txt  = f"WEAPON: {disp.upper()} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, txt, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return frame
