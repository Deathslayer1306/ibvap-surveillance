"""
Virtual Fence / Zone Breach Detector
=====================================
Input  : list of person centroids (derived from pose bboxes),
         list of Zone objects (from cameras.yaml)
Output : FenceResult(breaches=[FenceBreach(...)])

Algorithm
---------
1. For each person centroid, test containment against each enabled zone
   using cv2.pointPolygonTest (fast C extension).
2. A per-track dwell timer is incremented each frame the centroid is inside.
   Timers are decremented (not reset) when the person leaves — this avoids
   false-clear on a momentary missed detection.
3. A breach becomes CRITICAL when dwell_sec >= zone.dwell_threshold_sec.

Track IDs
---------
This detector uses centroid-based pseudo-tracking: the nearest centroid
from the previous frame is matched by Euclidean distance within a radius.
This is simple and fast; replace with ByteTrack if you add tracking later.

Zone polygon points are normalised 0-1 coords from cameras.yaml.
They are scaled to actual pixel coords (frame_w, frame_h) per-call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Zone:
    """Parsed zone configuration from cameras.yaml."""
    id: str
    name: str
    points: List[List[float]]           # normalised [x, y] pairs (0-1)
    dwell_threshold_sec: float
    enabled: bool = True


@dataclass
class FenceBreach:
    """A single person–zone breach event."""
    zone_id: str
    zone_name: str
    track_id: int
    dwell_sec: float
    is_critical: bool                   # dwell_sec >= zone.dwell_threshold_sec
    centroid: Tuple[int, int]           # pixel centroid at time of breach


@dataclass
class FenceResult:
    """Aggregated fence breach result for one frame."""
    breaches: List[FenceBreach] = field(default_factory=list)

    @property
    def has_breach(self) -> bool:
        return len(self.breaches) > 0

    @property
    def has_critical(self) -> bool:
        return any(b.is_critical for b in self.breaches)


# ── Centroid tracker helpers ──────────────────────────────────────────────────

def _bbox_centroid(bbox: List[int]) -> Tuple[int, int]:
    """Return (cx, cy) pixel centroid from [x1, y1, x2, y2] bbox."""
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


def _match_centroids(
    prev_ids: list[int],
    prev_cents: list[Tuple[int, int]],
    new_cents: list[Tuple[int, int]],
    max_dist: float = 120.0,
) -> dict[int, int]:
    """
    Match new centroids to previous track IDs by nearest-neighbour.
    Returns mapping: new_index → track_id
    Unmatched new centroids get fresh IDs.
    """
    if not prev_ids or not new_cents:
        return {}

    mapping: dict[int, int] = {}
    used_prev: set[int] = set()

    for ni, nc in enumerate(new_cents):
        best_dist = max_dist
        best_pi = -1
        for pi, pc in enumerate(prev_cents):
            if pi in used_prev:
                continue
            d = float(np.hypot(nc[0] - pc[0], nc[1] - pc[1]))
            if d < best_dist:
                best_dist = d
                best_pi = pi
        if best_pi >= 0:
            mapping[ni] = prev_ids[best_pi]
            used_prev.add(best_pi)

    return mapping


# ── Main Detector Class ───────────────────────────────────────────────────────

class FenceDetector:
    """
    Virtual fence / zone breach detector.

    Stateful: holds dwell timers per track_id per zone.
    Call detect() once per inference frame.
    """

    _DWELL_DECAY_RATE = 0.5        # seconds per frame to decay dwell when outside zone
    _DWELL_DECAY_INTERVAL = 1.0   # minimum interval between dwell decrements (seconds)

    def __init__(self) -> None:
        # dwell_timers[zone_id][track_id] = elapsed_seconds_inside
        self._dwell_timers: Dict[str, Dict[int, float]] = {}
        # Track state for simple centroid matching
        self._prev_track_ids: list[int]             = []
        self._prev_centroids: list[Tuple[int, int]] = []
        self._next_track_id: int                    = 0
        self._last_call_time: float                 = time.perf_counter()

    def _get_pixel_polygon(
        self, zone: Zone, frame_w: int, frame_h: int
    ) -> Optional[np.ndarray]:
        """Convert normalised 0-1 points to pixel-space polygon."""
        if len(zone.points) < 3:
            return None
        pts = []
        for p in zone.points:
            if len(p) < 2:
                continue
            px = int(p[0] * frame_w)
            py = int(p[1] * frame_h)
            pts.append([px, py])
        if len(pts) < 3:
            return None
        return np.array(pts, dtype=np.int32)

    def _point_in_polygon(self, cx: int, cy: int, polygon: np.ndarray) -> bool:
        """
        Returns True if point (cx, cy) is inside the polygon.
        Uses cv2.pointPolygonTest: > 0 = inside, 0 = on edge, < 0 = outside.
        """
        result = cv2.pointPolygonTest(polygon, (float(cx), float(cy)), measureDist=False)
        return result >= 0

    def detect(
        self,
        person_bboxes: List[List[int]],
        zones: List[Zone],
        frame_w: int,
        frame_h: int,
    ) -> FenceResult:
        """
        Detect zone breaches for the current frame.

        Args:
            person_bboxes : list of [x1, y1, x2, y2] from PoseDetector.
            zones         : list of Zone objects (from cameras.yaml).
            frame_w       : frame width  in pixels.
            frame_h       : frame height in pixels.

        Returns:
            FenceResult with breach events.
        """
        now = time.perf_counter()
        dt  = min(now - self._last_call_time, 5.0)   # cap delta at 5s
        self._last_call_time = now

        # ── Compute centroids and match to track IDs ──────────────────────
        new_centroids: list[Tuple[int, int]] = [
            _bbox_centroid(b) for b in person_bboxes
        ]

        # Match new detections to existing track IDs
        id_map = _match_centroids(
            self._prev_track_ids, self._prev_centroids, new_centroids
        )

        # Assign track IDs for this frame
        current_track_ids: list[int] = []
        for ni in range(len(new_centroids)):
            if ni in id_map:
                current_track_ids.append(id_map[ni])
            else:
                current_track_ids.append(self._next_track_id)
                self._next_track_id += 1

        # ── Process each zone ─────────────────────────────────────────────
        result = FenceResult()
        enabled_zones = [z for z in zones if z.enabled]

        for zone in enabled_zones:
            if zone.id not in self._dwell_timers:
                self._dwell_timers[zone.id] = {}

            polygon = self._get_pixel_polygon(zone, frame_w, frame_h)
            if polygon is None:
                continue

            zone_timers = self._dwell_timers[zone.id]
            active_tracks: set[int] = set()

            # ── Check each person against this zone ───────────────────────
            for ni, (tid, centroid) in enumerate(zip(current_track_ids, new_centroids)):
                inside = self._point_in_polygon(centroid[0], centroid[1], polygon)

                if inside:
                    active_tracks.add(tid)
                    # Increment dwell timer
                    zone_timers[tid] = zone_timers.get(tid, 0.0) + dt

                    dwell = zone_timers[tid]
                    is_critical = dwell >= zone.dwell_threshold_sec

                    result.breaches.append(FenceBreach(
                        zone_id=zone.id,
                        zone_name=zone.name,
                        track_id=tid,
                        dwell_sec=round(dwell, 2),
                        is_critical=is_critical,
                        centroid=centroid,
                    ))
                else:
                    # Decay timer for tracks that left the zone
                    if tid in zone_timers:
                        zone_timers[tid] = max(0.0, zone_timers[tid] - dt * self._DWELL_DECAY_RATE)
                        if zone_timers[tid] <= 0:
                            del zone_timers[tid]

            # Prune stale track IDs no longer seen at all
            stale = [tid for tid in zone_timers if tid not in set(current_track_ids)]
            for tid in stale:
                del zone_timers[tid]

        # ── Update track state for next frame ─────────────────────────────
        self._prev_track_ids = current_track_ids
        self._prev_centroids  = new_centroids

        return result

    # ── Annotate ─────────────────────────────────────────────────────────────
    def annotate(
        self,
        frame: np.ndarray,
        result: FenceResult,
        zones: List[Zone],
    ) -> np.ndarray:
        """Draw zone polygons and breach indicators on the frame."""
        h, w = frame.shape[:2]

        for zone in zones:
            if not zone.enabled:
                continue
            polygon = self._get_pixel_polygon(zone, w, h)
            if polygon is None:
                continue

            # Check if any breach in this zone
            zone_breached = any(b.zone_id == zone.id for b in result.breaches)
            zone_critical = any(b.zone_id == zone.id and b.is_critical for b in result.breaches)

            color = (0, 0, 255) if zone_critical else (0, 100, 255) if zone_breached else (0, 200, 100)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], (*color[::-1], 30))  # semi-transparent fill
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)

            # Zone label
            label_pos = tuple(polygon[0])
            cv2.putText(frame, zone.name, (label_pos[0] + 4, label_pos[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw dwell timers on breach centroids
        for breach in result.breaches:
            cx, cy = breach.centroid
            label = f"ZONE:{breach.zone_id} {breach.dwell_sec:.1f}s"
            if breach.is_critical:
                label = f"⚠ CRITICAL {label}"
            color = (0, 0, 255) if breach.is_critical else (0, 165, 255)
            cv2.putText(frame, label, (cx - 60, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        return frame

    @staticmethod
    def parse_zones_from_config(zones_cfg: list[dict]) -> List[Zone]:
        """Parse a list of zone dicts from cameras.yaml into Zone objects."""
        zones: List[Zone] = []
        for z in zones_cfg:
            try:
                zones.append(Zone(
                    id=str(z.get("id", "ZONE_?")),
                    name=str(z.get("name", "Zone")),
                    points=z.get("points", []),
                    dwell_threshold_sec=float(z.get("dwell_threshold_sec", 5.0)),
                    enabled=bool(z.get("enabled", True)),
                ))
            except Exception:
                pass
        return zones
