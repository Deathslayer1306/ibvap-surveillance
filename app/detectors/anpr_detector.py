"""
ANPR Detector — Automatic Number Plate Recognition
===================================================
Stage 1 : YOLOv11 ONNX plate localisation (model path from cameras.yaml)
Stage 2 : EasyOCR on the cropped plate region
Post    : Normalise plate string (uppercase, strip spaces/hyphens)
          Watchlist O(1) lookup via Python set

If the ONNX model file is missing or EasyOCR fails to import, the detector
disables itself silently — it will never crash the main pipeline.

Returns
-------
ANPRResult(plates=[PlateDetection(...)], watchlist_hits=["MH12AB1234", ...])
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Try importing heavy optional deps ────────────────────────────────────────
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    logger.warning("[ANPRDetector] onnxruntime not found — ANPR disabled.")

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False
    logger.warning("[ANPRDetector] easyocr not found — OCR stage disabled.")


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class PlateDetection:
    """Single plate detection result."""
    bbox: List[int]               # [x1, y1, x2, y2] in original frame coords
    confidence: float             # plate localisation confidence
    raw_text: str                 # raw OCR string (un-normalised)
    plate_text: str               # normalised plate string (uppercase, stripped)
    is_watchlist_hit: bool = False


@dataclass
class ANPRResult:
    """Aggregated ANPR result for one frame."""
    plates: List[PlateDetection] = field(default_factory=list)
    watchlist_hits: List[str] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return len(self.plates) > 0

    @property
    def has_watchlist_hit(self) -> bool:
        return len(self.watchlist_hits) > 0


# ── Plate text normalisation ──────────────────────────────────────────────────

def _normalise_plate(raw: str) -> str:
    """Uppercase, strip spaces, hyphens, and common OCR artefacts."""
    text = raw.upper()
    text = re.sub(r"[\s\-_.]", "", text)       # remove whitespace, separators
    text = re.sub(r"[^A-Z0-9]", "", text)      # keep only alphanumeric
    return text


# ── Watchlist loader ─────────────────────────────────────────────────────────

def _load_watchlist(path: str) -> set[str]:
    """
    Load plate watchlist JSON from path.
    Format: {"plates": ["MH12AB1234", "DL01AB0001", ...]}
    Returns a set of normalised plate strings for O(1) lookup.
    """
    if not os.path.isfile(path):
        logger.warning(f"[ANPRDetector] Watchlist file not found: {path}. Creating empty watchlist.")
        # Create default empty watchlist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"plates": []}, f, indent=2)
        return set()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        plates = {_normalise_plate(p) for p in data.get("plates", [])}
        logger.info(f"[ANPRDetector] Loaded {len(plates)} plates from watchlist.")
        return plates
    except Exception as exc:
        logger.warning(f"[ANPRDetector] Failed to load watchlist: {exc}")
        return set()


# ── ANPR NMS ─────────────────────────────────────────────────────────────────

def _nms(boxes: list, scores: list, iou_threshold: float = 0.45) -> list:
    """Pure-numpy NMS — returns indices of kept boxes."""
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


# ── Main Detector Class ───────────────────────────────────────────────────────

class ANPRDetector:
    """
    Two-stage ANPR detector:
      1. YOLOv11 ONNX → plate bounding boxes
      2. EasyOCR     → plate text from cropped region

    Designed to match the existing detector pattern in this repo:
      __init__(...)  → load model (graceful failure)
      detect(frame)  → ANPRResult
    """

    INPUT_SIZE = (640, 640)      # YOLOv11 default input size
    _MIN_PLATE_AREA = 400        # pixels² — ignore tiny detections

    def __init__(
        self,
        model_path: str,
        watchlist_path: str,
        confidence_threshold: float = 0.5,
        ocr_engine: str = "easyocr",
        onnx_providers: Optional[list] = None,
    ) -> None:
        self.enabled = False
        self.conf_threshold = confidence_threshold
        self.ocr_engine_name = ocr_engine
        self._session: Optional[object] = None
        self._ocr_reader: Optional[object] = None
        self._input_name: str = ""

        # ── Load watchlist (always attempt, even without ONNX model) ─────────
        self._watchlist: set[str] = _load_watchlist(watchlist_path)
        self._watchlist_path = watchlist_path

        # ── Load ONNX plate localisation model ───────────────────────────────
        if not _ORT_AVAILABLE:
            logger.warning("[ANPRDetector] onnxruntime unavailable — ANPR disabled.")
            return

        if not os.path.isfile(model_path):
            logger.warning(
                f"[ANPRDetector] Model not found at '{model_path}'. "
                "ANPR localisation disabled. Train with: python tools/train_anpr.py"
            )
            # Fall through — OCR-only mode is also useless without localisation
            return

        providers = onnx_providers or ["CPUExecutionProvider"]
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            opts.log_severity_level = 3
            self._session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            active_provider = self._session.get_providers()[0]
            logger.info(f"[ANPRDetector] Model loaded  : {model_path}")
            logger.info(f"[ANPRDetector] Provider      : {active_provider}")
        except Exception as exc:
            logger.warning(f"[ANPRDetector] Failed to load ONNX model: {exc}")
            return

        # ── Load OCR engine ──────────────────────────────────────────────────
        if ocr_engine == "easyocr":
            if not _EASYOCR_AVAILABLE:
                logger.warning(
                    "[ANPRDetector] easyocr not installed — running in LOCALISATION-ONLY mode. "
                    "Plates will be detected and boxed but not read (no OCR). "
                    "Install: pip install easyocr"
                )
                # Still enable the detector — just without the OCR reader
                self._ocr_reader = None
            else:
                try:
                    self._ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
                    logger.info("[ANPRDetector] EasyOCR reader initialised.")
                except Exception as exc:
                    logger.warning(f"[ANPRDetector] EasyOCR init failed: {exc} — localisation-only mode.")
                    self._ocr_reader = None
        else:
            logger.warning(f"[ANPRDetector] OCR engine '{ocr_engine}' not yet implemented. "
                           "Running in localisation-only mode.")
            self._ocr_reader = None

        # Detect if model is COCO-80 (placeholder) vs real plate model (1–2 classes)
        out_shape = self._session.get_outputs()[0].shape
        # YOLOv8 output: (1, 4+num_classes, anchors) — shape[1] tells us class count
        if len(out_shape) >= 2 and isinstance(out_shape[1], int):
            num_cls = max(1, out_shape[1] - 4)
            if num_cls > 4:
                # This is a multi-class COCO model used as placeholder
                self._coco_proxy_mode = True
                logger.info(
                    f"[ANPRDetector] COCO proxy mode ({num_cls} classes) — "
                    "using vehicle detections as plate candidates. "
                    "Run tools/train_anpr.py for real plate detection."
                )
            else:
                self._coco_proxy_mode = False
        else:
            self._coco_proxy_mode = False

        self.enabled = True
        logger.info(
            f"[ANPRDetector] Ready. OCR={'enabled' if self._ocr_reader else 'localisation-only'}  "
            f"Watchlist={len(self._watchlist)}"
        )

    # ── Public: reload watchlist at runtime ──────────────────────────────────
    def reload_watchlist(self) -> int:
        """Reload plates.json from disk. Returns new watchlist size."""
        self._watchlist = _load_watchlist(self._watchlist_path)
        return len(self._watchlist)

    def add_plate(self, plate: str) -> None:
        """Add a plate to the in-memory watchlist and persist to disk."""
        normalised = _normalise_plate(plate)
        self._watchlist.add(normalised)
        self._persist_watchlist()

    def _persist_watchlist(self) -> None:
        """Write current in-memory watchlist back to plates.json."""
        try:
            with open(self._watchlist_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
        data["plates"] = sorted(self._watchlist)
        with open(self._watchlist_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Pre-processing ────────────────────────────────────────────────────────
    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = frame.shape[:2]
        iw, ih = self.INPUT_SIZE
        img = cv2.resize(frame, (iw, ih))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.expand_dims(img.transpose(2, 0, 1), 0)      # NCHW
        return blob, w / iw, h / ih

    # ── Post-processing ───────────────────────────────────────────────────────
    # COCO vehicle class IDs treated as plate candidates when in proxy mode
    _COCO_VEHICLE_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

    def _postprocess(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
        sx: float,
        sy: float,
    ) -> list[tuple[list[int], float]]:
        """
        Parse YOLOv8/v11 output → list of (bbox, confidence).
        In COCO proxy mode: only vehicle class IDs are kept as plate candidates.
        In real plate model mode: all detections above threshold are kept.
        """
        out = raw[0]
        if out.shape[0] < out.shape[1]:
            out = out.T      # (4+classes, anchors) → (anchors, 4+classes)

        detections: list[tuple[list[int], float]] = []
        for row in out:
            cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            class_scores = row[4:]
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])

            if conf < self.conf_threshold:
                continue

            # In COCO proxy mode only keep vehicle classes
            if getattr(self, '_coco_proxy_mode', False):
                if cls_id not in self._COCO_VEHICLE_IDS:
                    continue

            x1 = max(0,      int((cx - bw / 2) * sx))
            y1 = max(0,      int((cy - bh / 2) * sy))
            x2 = min(orig_w, int((cx + bw / 2) * sx))
            y2 = min(orig_h, int((cy + bh / 2) * sy))
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) * (y2 - y1) < self._MIN_PLATE_AREA:
                continue
            detections.append(([x1, y1, x2, y2], conf))

        if not detections:
            return []

        boxes  = [d[0] for d in detections]
        scores = [d[1] for d in detections]
        kept   = _nms(boxes, scores)
        return [detections[i] for i in kept]


    # ── OCR on crop ──────────────────────────────────────────────────────────
    def _ocr_crop(self, frame: np.ndarray, bbox: list[int]) -> str:
        """Run EasyOCR on a plate crop. Returns raw OCR text."""
        x1, y1, x2, y2 = bbox
        # Add small padding for better OCR accuracy
        pad = 4
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(frame.shape[1], x2 + pad)
        y2p = min(frame.shape[0], y2 + pad)
        crop = frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            return ""
        try:
            results = self._ocr_reader.readtext(crop, detail=0, paragraph=False)
            return " ".join(results) if results else ""
        except Exception as exc:
            logger.debug(f"[ANPRDetector] OCR error: {exc}")
            return ""

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> ANPRResult:
        """
        Run ANPR on one frame.

        Args:
            frame: BGR numpy array (any resolution).

        Returns:
            ANPRResult with plates and watchlist_hits.
            Returns empty result immediately if detector is disabled.
        """
        if not self.enabled or self._session is None:
            return ANPRResult()

        h, w = frame.shape[:2]
        blob, sx, sy = self._preprocess(frame)

        try:
            raw = self._session.run(None, {self._input_name: blob})[0]
        except Exception as exc:
            logger.warning(f"[ANPRDetector] Inference error: {exc}")
            return ANPRResult()

        detections = self._postprocess(raw, w, h, sx, sy)

        result = ANPRResult()
        for bbox, conf in detections:
            raw_text = ""
            if self._ocr_reader is not None:
                raw_text = self._ocr_crop(frame, bbox)

            normalised = _normalise_plate(raw_text) if raw_text else ""
            hit = normalised in self._watchlist and bool(normalised)

            plate = PlateDetection(
                bbox=bbox,
                confidence=round(conf, 3),
                raw_text=raw_text,
                # In localisation-only or proxy mode, show a meaningful placeholder
                plate_text=normalised if normalised else (
                    "[LP]" if getattr(self, '_coco_proxy_mode', False) else "PLATE?"
                ),
                is_watchlist_hit=hit,
            )
            result.plates.append(plate)
            if hit:
                result.watchlist_hits.append(normalised)

        return result

    # ── Annotate ─────────────────────────────────────────────────────────────
    def annotate(self, frame: np.ndarray, result: ANPRResult) -> np.ndarray:
        """Draw plate bounding boxes and OCR text on the frame."""
        for plate in result.plates:
            x1, y1, x2, y2 = plate.bbox
            color = (0, 0, 255) if plate.is_watchlist_hit else (0, 200, 255)
            thickness = 3 if plate.is_watchlist_hit else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = plate.plate_text if plate.plate_text else "PLATE"
            if plate.is_watchlist_hit:
                label = f"⚠ WATCHLIST: {label}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame
