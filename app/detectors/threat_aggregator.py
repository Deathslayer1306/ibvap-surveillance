"""
Threat Aggregator — combines signals from all detectors into a unified threat score.
Designed for drone use: threshold-driven alert levels, JSON-serialisable output.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional
import json

from app.config import (
    SCORE_WEAPON,
    SCORE_EMOTION_ANGRY, SCORE_EMOTION_FEAR,
    SCORE_POSE_AGGRESSIVE, SCORE_POSE_RAISED_HANDS, SCORE_POSE_CROUCHING,
    ALERT_WARNING_THRESHOLD, ALERT_CRITICAL_THRESHOLD,
)
from app.detectors.pose_detector import PoseAnomaly


@dataclass
class ThreatState:
    # Raw signals
    weapon_detected:       bool       = False
    weapon_labels:         List[str]  = field(default_factory=list)
    weapon_display_labels: List[str]  = field(default_factory=list)
    emotion:               str        = "neutral"
    emotion_scores:        dict       = field(default_factory=dict)
    pose_anomaly:          str        = PoseAnomaly.NORMAL.value
    num_persons:           int        = 0

    # Derived
    threat_score: int = 0
    alert_level:  str = "INFO"        # INFO | WARNING | CRITICAL
    timestamp:    str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @property
    def is_suspicious(self) -> bool:
        return self.threat_score >= ALERT_WARNING_THRESHOLD


class ThreatAggregator:
    """
    Stateless scoring engine.
    Call update(...) every inference cycle to get a fresh ThreatState.
    """

    @staticmethod
    def update(
        weapon_detected:  bool              = False,
        weapon_labels:    Optional[List[str]] = None,
        weapon_display:   Optional[List[str]] = None,
        emotion:          str               = "neutral",
        emotion_scores:   Optional[dict]    = None,
        pose_anomalies:   Optional[List[str]] = None,
        num_persons:      int               = 0,
    ) -> ThreatState:

        score = 0

        # ── Weapon ────────────────────────────────────────────────────────────
        if weapon_detected:
            score += SCORE_WEAPON

        # ── Emotion ───────────────────────────────────────────────────────────
        if emotion == "angry":
            score += SCORE_EMOTION_ANGRY
        elif emotion == "fear":
            score += SCORE_EMOTION_FEAR

        # ── Pose ──────────────────────────────────────────────────────────────
        worst_pose = PoseAnomaly.NORMAL.value
        if pose_anomalies:
            for anomaly in pose_anomalies:
                if anomaly == PoseAnomaly.AGGRESSIVE.value:
                    score += SCORE_POSE_AGGRESSIVE
                    worst_pose = anomaly
                elif anomaly == PoseAnomaly.RAISED_HANDS.value:
                    score += SCORE_POSE_RAISED_HANDS
                    if worst_pose not in [PoseAnomaly.AGGRESSIVE.value]:
                        worst_pose = anomaly
                elif anomaly == PoseAnomaly.CROUCHING.value:
                    score += SCORE_POSE_CROUCHING
                    if worst_pose == PoseAnomaly.NORMAL.value:
                        worst_pose = anomaly

        # ── Alert Level ───────────────────────────────────────────────────────
        if score >= ALERT_CRITICAL_THRESHOLD:
            alert = "CRITICAL"
        elif score >= ALERT_WARNING_THRESHOLD:
            alert = "WARNING"
        else:
            alert = "INFO"

        return ThreatState(
            weapon_detected       = weapon_detected,
            weapon_labels         = weapon_labels  or [],
            weapon_display_labels = weapon_display or [],
            emotion               = emotion,
            emotion_scores        = emotion_scores or {},
            pose_anomaly          = worst_pose,
            num_persons           = num_persons,
            threat_score          = min(score, 10),
            alert_level           = alert,
            timestamp             = datetime.now(timezone.utc).isoformat(),
        )
