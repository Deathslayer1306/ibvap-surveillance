"""Prometheus metrics for Suspicious Activity Detection.

Thread-safe usage:
- Gauges/counters/histograms from prometheus_client are safe to update from
  multiple threads.
- This module only defines metrics + helper functions.
"""

from __future__ import annotations

from typing import Dict, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
)


# ── All 7 emotion labels in FER-2013 / AffectNet order ────────────────────────
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ── Process/endpoint metrics ───────────────────────────────────────────────────

REQUESTS_TOTAL = Counter(
    "sad_requests_total",
    "Total FastAPI HTTP requests",
    ["method", "path", "http_status"],
)

REQUEST_LATENCY_SECONDS = Histogram(
    "sad_request_latency_seconds",
    "FastAPI request latency in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


# ── Camera/inference metrics ───────────────────────────────────────────────────

CAMERA_FPS = Gauge(
    "sad_camera_fps",
    "Camera FPS (measured by capture thread)",
)

INFERENCE_FPS = Gauge(
    "sad_inference_fps",
    "Inference FPS (measured by inference thread)",
)

INFERENCE_LOOP_LATENCY_SECONDS = Histogram(
    "sad_inference_loop_latency_seconds",
    "Time spent per inference loop iteration (end-to-end in the thread)",
    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0),
)


# ── Threat/detection metrics ───────────────────────────────────────────────────

THREAT_SCORE = Gauge(
    "sad_threat_score",
    "Current threat score (0-10) as computed by ThreatAggregator",
)

ALERT_LEVEL = Gauge(
    "sad_alert_level",
    "Current alert level as numeric value: 0=INFO, 1=WARNING, 2=CRITICAL",
)

ALERT_LEVEL_EVENTS_TOTAL = Counter(
    "sad_alert_level_events_total",
    "Count of times alert level transitions to WARNING/CRITICAL in the inference loop",
    ["alert_level"],
)

WEAPON_DETECTED = Gauge(
    "sad_weapon_detected",
    "Whether weapon detector currently reports weapon (0=no, 1=yes)",
)

PERSONS_IN_FRAME = Gauge(
    "sad_persons_in_frame",
    "Number of detected persons in current inference cycle",
)


# ── Per-emotion confidence gauges (all 7 emotions) ────────────────────────────
# Each gauge holds the softmax probability (0.0–1.0) for that emotion class.
# When no face is detected, these are set to 0.0.
# This allows Grafana to plot each emotion's confidence over time separately.

EMOTION_SCORE_GAUGES: Dict[str, Gauge] = {
    emotion: Gauge(
        f"sad_emotion_{emotion}",
        f"Softmax confidence for emotion '{emotion}' (0.0–1.0). 0 when no face detected.",
    )
    for emotion in EMOTION_LABELS
}

# Combined dominant emotion as an index (kept for backward compat)
LAST_EMOTION_LABEL = Gauge(
    "sad_last_emotion_label",
    "Dominant emotion label as index: 0=angry,1=disgust,2=fear,3=happy,4=neutral,5=sad,6=surprise. -1=no face.",
)

# ── Pose anomaly metric ────────────────────────────────────────────────────────
LAST_POSE_ANOMALY_LABEL = Gauge(
    "sad_last_pose_anomaly_label",
    "Pose anomaly as index: 0=normal,1=raised_hands,2=aggressive_stance,3=crouching,4=running.",
)


# ── Index maps ────────────────────────────────────────────────────────────────

_EMOTION_INDEX: Dict[str, int] = {e: i for i, e in enumerate(EMOTION_LABELS)}

_POSE_INDEX = {
    "normal":           0,
    "raised_hands":     1,
    "aggressive_stance": 2,
    "crouching":        3,
    "running":          4,
}


# ── Helper functions ───────────────────────────────────────────────────────────


def set_camera_fps(v: float) -> None:
    CAMERA_FPS.set(v)


def set_inference_fps(v: float) -> None:
    INFERENCE_FPS.set(v)


def observe_inference_loop_latency_seconds(v: float) -> None:
    INFERENCE_LOOP_LATENCY_SECONDS.observe(v)


def update_threat_metrics(
    *,
    threat_score: int,
    alert_level: str,
    weapon_detected: bool,
    emotion: str,
    emotion_scores: Optional[Dict[str, float]] = None,
    pose_anomaly: str,
    num_persons: int,
    alert_event: bool = False,
) -> None:
    """Update all Prometheus metrics from the latest inference result.

    Call this once per inference loop iteration from InferenceThread.
    All prometheus_client objects are thread-safe.
    """
    # ── Threat / alert ──────────────────────────────────────────────────────
    THREAT_SCORE.set(threat_score)

    level_val = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}.get(alert_level, 0)
    ALERT_LEVEL.set(level_val)

    if alert_event:
        ALERT_LEVEL_EVENTS_TOTAL.labels(alert_level=alert_level).inc()

    # ── Detection ───────────────────────────────────────────────────────────
    WEAPON_DETECTED.set(1 if weapon_detected else 0)
    PERSONS_IN_FRAME.set(num_persons)

    # ── Emotion — individual confidence gauges ──────────────────────────────
    if emotion_scores:
        # Face was detected; set each emotion's softmax probability
        for emo_label, gauge in EMOTION_SCORE_GAUGES.items():
            gauge.set(float(emotion_scores.get(emo_label, 0.0)))
        LAST_EMOTION_LABEL.set(_EMOTION_INDEX.get((emotion or "").lower(), -1))
    else:
        # No face detected — zero all emotion gauges, set index to -1
        for gauge in EMOTION_SCORE_GAUGES.values():
            gauge.set(0.0)
        LAST_EMOTION_LABEL.set(-1)

    # ── Pose ────────────────────────────────────────────────────────────────
    LAST_POSE_ANOMALY_LABEL.set(_POSE_INDEX.get((pose_anomaly or "").lower(), 0))


def format_metrics() -> bytes:
    """Return current Prometheus text exposition payload for /metrics."""
    return generate_latest()
