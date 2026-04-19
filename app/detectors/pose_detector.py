"""
Pose Detector — GPU-accelerated, drone-ready.
Detects human keypoints and classifies pose anomalies.

Keypoint index (COCO 17-point skeleton):
  0:nose  1:L_eye  2:R_eye  3:L_ear  4:R_ear
  5:L_shoulder  6:R_shoulder  7:L_elbow  8:R_elbow
  9:L_wrist  10:R_wrist  11:L_hip  12:R_hip
  13:L_knee  14:R_knee  15:L_ankle  16:R_ankle
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

from app.config import MODEL_POSE, ONNX_PROVIDERS


class PoseAnomaly(str, Enum):
    NORMAL           = "normal"
    RAISED_HANDS     = "raised_hands"
    AGGRESSIVE       = "aggressive_stance"
    CROUCHING        = "crouching"
    RUNNING          = "running"


SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

KP_COLOR   = (0, 255, 180)
BONE_COLOR = (0, 180, 255)

# Minimum keypoint confidence to treat as visible
_KP_CONF_THRESH = 0.50


def _nms_poses(persons: List["PersonPose"], iou_thresh: float = 0.50) -> List["PersonPose"]:
    """Remove duplicate pose detections via bounding-box IoU NMS."""
    if len(persons) <= 1:
        return persons
    persons = sorted(persons, key=lambda p: p.confidence, reverse=True)
    keep = []
    suppressed = [False] * len(persons)
    for i, p in enumerate(persons):
        if suppressed[i]:
            continue
        keep.append(p)
        ax1, ay1, ax2, ay2 = p.bbox
        aarea = max((ax2 - ax1) * (ay2 - ay1), 1)
        for j in range(i + 1, len(persons)):
            if suppressed[j]:
                continue
            bx1, by1, bx2, by2 = persons[j].bbox
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            barea = max((bx2 - bx1) * (by2 - by1), 1)
            iou   = inter / (aarea + barea - inter + 1e-6)
            if iou > iou_thresh:
                suppressed[j] = True
    return keep


@dataclass
class PersonPose:
    keypoints: np.ndarray       # (17, 3) → x, y, conf  — in INFERENCE frame coords
    bbox: List[int]             # [x1, y1, x2, y2]
    anomaly: PoseAnomaly = PoseAnomaly.NORMAL
    confidence: float = 0.0


class PoseDetector:
    """Runs yolo11n-pose.onnx; returns list[PersonPose]."""

    INPUT_SIZE = (640, 640)

    def __init__(self) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.log_severity_level = 3

        try:
            self.session = ort.InferenceSession(
                MODEL_POSE, sess_options=opts, providers=ONNX_PROVIDERS,
            )
        except Exception:
            print("[PoseDetector] GPU load failed, using CPU")
            self.session = ort.InferenceSession(
                MODEL_POSE, sess_options=opts, providers=["CPUExecutionProvider"],
            )

        self.input_name = self.session.get_inputs()[0].name
        print(f"[PoseDetector] Provider: {self.session.get_providers()[0]}")

    # ── pre / post ────────────────────────────────────────────────────────────
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        h, w = frame.shape[:2]
        img = cv2.resize(frame, self.INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.expand_dims(img.transpose(2, 0, 1), 0)
        return blob, w / 640.0, h / 640.0

    @staticmethod
    def _postprocess(
        raw: np.ndarray,
        sx: float, sy: float,
        obj_thresh: float = 0.55,   # raised: eliminates ghost person detections
        kp_thresh:  float = _KP_CONF_THRESH,
    ) -> List[PersonPose]:
        out = raw[0]
        if out.shape[0] < out.shape[1]:
            out = out.T   # (N, 5+51)

        persons: List[PersonPose] = []
        for row in out:
            if row.shape[0] < 56:
                continue
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            obj_conf = float(row[4])
            if obj_conf < obj_thresh:
                continue

            x1 = max(0, int((cx - bw / 2) * sx))
            y1 = max(0, int((cy - bh / 2) * sy))
            x2 = int((cx + bw / 2) * sx)
            y2 = int((cy + bh / 2) * sy)

            # Skip degenerate / tiny boxes
            if x2 - x1 < 30 or y2 - y1 < 30:
                continue

            kp_raw = row[5:56].reshape(17, 3)
            kp = kp_raw.copy()
            kp[:, 0] *= sx
            kp[:, 1] *= sy
            kp[kp[:, 2] < kp_thresh, :2] = -1   # mark low-confidence as missing

            persons.append(PersonPose(
                keypoints=kp, bbox=[x1, y1, x2, y2],
                confidence=round(obj_conf, 3),
            ))

        # NMS — removes overlapping duplicate ghost detections
        persons = _nms_poses(persons, iou_thresh=0.45)
        return persons

    # ── anomaly logic (conservative, high-confidence only) ────────────────────
    @staticmethod
    def _classify(pose: PersonPose) -> PoseAnomaly:
        kp = pose.keypoints

        def vis(i: int) -> bool:
            """Keypoint is reliably visible."""
            return kp[i, 0] > 0 and kp[i, 1] > 0 and kp[i, 2] >= _KP_CONF_THRESH

        def pt(i: int) -> Tuple[float, float]:
            return float(kp[i, 0]), float(kp[i, 1])

        # Need a reference head point
        if not vis(0):
            return PoseAnomaly.NORMAL

        nose_y = pt(0)[1]
        h_box  = pose.bbox[3] - pose.bbox[1]     # person bounding-box height
        if h_box < 20:
            return PoseAnomaly.NORMAL

        # ── RAISED HANDS: BOTH wrists clearly above the nose ─────────────────
        # Require both wrists visible AND above nose by at least 10% of body height
        if vis(9) and vis(10):
            lw_y, rw_y = pt(9)[1], pt(10)[1]
            margin = h_box * 0.10
            if lw_y < nose_y - margin and rw_y < nose_y - margin:
                return PoseAnomaly.RAISED_HANDS

        # ── AGGRESSIVE: wrists near hip center (punching/fighting range) ──────
        # Only flag if hips AND wrists all visible
        if vis(9) and vis(10) and vis(11) and vis(12):
            hip_cx = (pt(11)[0] + pt(12)[0]) / 2
            hip_cy = (pt(11)[1] + pt(12)[1]) / 2
            # Approximate torso height using hips vs nose
            torso_h = max(abs(hip_cy - nose_y), 1)

            clenched = 0
            for wi in [9, 10]:
                if vis(wi):
                    wx, wy = pt(wi)
                    dist = np.hypot(wx - hip_cx, wy - hip_cy)
                    # Wrist within 40% of torso height from hip center
                    if dist < torso_h * 0.40:
                        clenched += 1
            if clenched >= 2:
                return PoseAnomaly.AGGRESSIVE

        # ── CROUCHING: both hips close to both ankles ─────────────────────────
        if vis(11) and vis(12) and vis(15) and vis(16):
            hip_y   = (pt(11)[1] + pt(12)[1]) / 2
            ankle_y = (pt(15)[1] + pt(16)[1]) / 2
            # distance < 25% of body height → crouching
            if abs(hip_y - ankle_y) < h_box * 0.25:
                return PoseAnomaly.CROUCHING

        return PoseAnomaly.NORMAL

    # ── annotation ────────────────────────────────────────────────────────────
    @staticmethod
    def annotate(frame: np.ndarray, persons: List[PersonPose]) -> np.ndarray:
        anomaly_colors = {
            PoseAnomaly.NORMAL:    (0, 220, 100),
            PoseAnomaly.RAISED_HANDS: (0, 220, 255),
            PoseAnomaly.AGGRESSIVE:   (0, 60, 255),
            PoseAnomaly.CROUCHING:    (0, 165, 255),
            PoseAnomaly.RUNNING:      (255, 165, 0),
        }
        for person in persons:
            kp    = person.keypoints
            color = anomaly_colors.get(person.anomaly, (0, 220, 100))

            # Bones
            for i, j in SKELETON_PAIRS:
                if kp[i, 0] > 0 and kp[j, 0] > 0:
                    cv2.line(frame,
                             (int(kp[i, 0]), int(kp[i, 1])),
                             (int(kp[j, 0]), int(kp[j, 1])),
                             BONE_COLOR, 2)

            # Keypoints
            for idx in range(17):
                if kp[idx, 0] > 0:
                    cv2.circle(frame, (int(kp[idx, 0]), int(kp[idx, 1])), 4, KP_COLOR, -1)

            # Bounding box + label
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if person.anomaly != PoseAnomaly.NORMAL:
                label = person.anomaly.value.replace("_", " ").upper()
                cv2.putText(frame, f"POSE:{label}",
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    # ── public API ────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[PersonPose]:
        blob, sx, sy = self._preprocess(frame)
        raw = self.session.run(None, {self.input_name: blob})[0]
        persons = self._postprocess(raw, sx, sy)
        for p in persons:
            p.anomaly = self._classify(p)
        return persons
