"""
tools/draw_zones.py — Interactive Virtual Fence Zone Editor
============================================================
Opens an OpenCV GUI window showing the camera feed from cameras.yaml.
Click to draw polygon zones; coordinates are saved (normalised 0-1)
back into cameras.yaml.

Controls
--------
  Left-click      : Add polygon vertex
  Right-click     : Close current polygon and save zone
  'u'             : Undo last vertex
  'c'             : Cancel / clear current polygon
  'q' / ESC       : Quit (saves all completed polygons)

Usage
-----
  python tools/draw_zones.py --camera CAM_01
  python tools/draw_zones.py --camera CAM_02 --zone-id ZONE_C --zone-name "South Wall"

The updated cameras.yaml is written atomically (backup is kept as cameras.yaml.bak).
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import shutil
import sys
from typing import Optional

import cv2
import numpy as np

# ── Ensure project root on PYTHONPATH ────────────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import yaml
except ImportError:
    print("[draw_zones] ERROR: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(1)


# ── YAML helpers ─────────────────────────────────────────────────────────────

def _load_yaml() -> dict:
    yaml_path = _PROJECT_ROOT / "cameras.yaml"
    if not yaml_path.exists():
        print(f"[draw_zones] cameras.yaml not found at {yaml_path}")
        sys.exit(1)
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(data: dict) -> None:
    yaml_path = _PROJECT_ROOT / "cameras.yaml"
    backup    = _PROJECT_ROOT / "cameras.yaml.bak"
    # Backup first
    shutil.copy2(yaml_path, backup)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"[draw_zones] cameras.yaml updated. Backup at: {backup}")


def _find_camera_cfg(data: dict, cam_id: str) -> Optional[dict]:
    for cam in data.get("cameras", []):
        if str(cam.get("id", "")) == cam_id:
            return cam
    return None


# ── Drawing state ─────────────────────────────────────────────────────────────

class ZoneDrawer:
    POINT_RADIUS = 6
    LINE_COLOR   = (0, 255, 120)
    FILL_COLOR   = (0, 255, 120)
    SAVED_COLOR  = (0, 200, 255)
    PENDING_COLOR= (255, 170, 0)
    TEXT_COLOR   = (255, 255, 255)

    def __init__(self, cam_id: str, zone_id: str, zone_name: str) -> None:
        self.cam_id    = cam_id
        self.zone_id   = zone_id
        self.zone_name = zone_name

        self.current_pts: list[tuple[int, int]] = []    # pixel coords being drawn
        self.saved_zones: list[dict] = []               # completed zones (normalised)

        self.frame_w = 1
        self.frame_h = 1
        self.frame: Optional[np.ndarray] = None

    def set_frame(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.frame_h, self.frame_w = frame.shape[:2]

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._close_polygon()

    def _close_polygon(self) -> None:
        if len(self.current_pts) < 3:
            print(f"[draw_zones] Need at least 3 points to form a polygon.")
            return
        # Normalise
        norm = [
            [round(pt[0] / self.frame_w, 4), round(pt[1] / self.frame_h, 4)]
            for pt in self.current_pts
        ]
        zone = {
            "id":                  self.zone_id,
            "name":                self.zone_name,
            "points":              norm,
            "dwell_threshold_sec": 5.0,
            "enabled":             True,
        }
        self.saved_zones.append(zone)
        print(f"[draw_zones] Zone '{self.zone_name}' saved with {len(norm)} points.")
        self.current_pts = []

        # Auto-increment zone_id for next zone
        base = self.zone_id.rstrip("0123456789")
        num_str = self.zone_id[len(base):]
        if num_str.isdigit():
            self.zone_id = f"{base}{int(num_str)+1}"
            self.zone_name = f"Zone {int(num_str)+1}"

    def render(self, base_frame: np.ndarray) -> np.ndarray:
        frame = base_frame.copy()
        h, w = frame.shape[:2]

        # Draw all saved zones (filled + outline)
        for zone in self.saved_zones:
            pts = np.array(
                [[int(p[0] * w), int(p[1] * h)] for p in zone["points"]], dtype=np.int32
            )
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], self.SAVED_COLOR)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [pts], isClosed=True, color=self.SAVED_COLOR, thickness=2)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(frame, zone["name"], (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)

        # Draw current in-progress polygon
        if self.current_pts:
            for i, pt in enumerate(self.current_pts):
                cv2.circle(frame, pt, self.POINT_RADIUS, self.PENDING_COLOR, -1)
                if i > 0:
                    cv2.line(frame, self.current_pts[i-1], pt, self.PENDING_COLOR, 2)
            # Close preview line
            if len(self.current_pts) >= 2:
                cv2.line(frame, self.current_pts[-1], self.current_pts[0],
                         self.PENDING_COLOR, 1)

        # HUD instructions
        instructions = [
            "Left-click: add vertex",
            "Right-click: close polygon & save",
            "U: undo last vertex",
            "C: clear current polygon",
            "Q / ESC: save & quit",
        ]
        for i, txt in enumerate(instructions):
            cv2.putText(frame, txt, (10, 22 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.putText(frame,
                    f"Drawing: {self.zone_id} — {self.zone_name}  |  "
                    f"Points: {len(self.current_pts)}  |  Zones saved: {len(self.saved_zones)}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        return frame


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive zone editor for IBVAP cameras.yaml")
    parser.add_argument("--camera",    required=True,          help="Camera ID (e.g. CAM_01)")
    parser.add_argument("--zone-id",   default="ZONE_NEW",     help="Starting zone ID")
    parser.add_argument("--zone-name", default="New Zone",     help="Starting zone name")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace all existing zones for this camera (default: append)")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    data       = _load_yaml()
    cam_cfg    = _find_camera_cfg(data, args.camera)
    if cam_cfg is None:
        print(f"[draw_zones] Camera '{args.camera}' not found in cameras.yaml.")
        print("Available:", [c.get("id") for c in data.get("cameras", [])])
        sys.exit(1)

    source = cam_cfg.get("source", 0)
    if isinstance(source, str):
        try:
            source = int(source)
        except ValueError:
            pass

    # ── Open camera ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[draw_zones] Cannot open source: {source}")
        sys.exit(1)

    # ── Set up drawer ─────────────────────────────────────────────────────────
    drawer = ZoneDrawer(
        cam_id=args.camera,
        zone_id=args.zone_id,
        zone_name=args.zone_name,
    )

    # Pre-load existing zones so user can see them
    existing_zones = cam_cfg.get("zones", []) if not args.overwrite else []
    drawer.saved_zones = copy.deepcopy(existing_zones)

    win = f"IBVAP Zone Editor — {args.camera}"
    cv2.namedWindow(win, cv2.WINDOW_RESIZABLE)
    cv2.setMouseCallback(win, drawer.on_mouse)

    print(f"[draw_zones] Editing zones for camera: {args.camera}")
    print(f"[draw_zones] Source: {source}")
    print(f"[draw_zones] Existing zones pre-loaded: {[z['name'] for z in existing_zones]}")

    ok, base_frame = cap.read()
    if not ok:
        # Use a blank canvas if no frame available
        base_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    drawer.set_frame(base_frame)

    while True:
        ok, live = cap.read()
        if ok:
            drawer.set_frame(live)
            display = drawer.render(live)
        else:
            # EOF / file source — rewind or hold last frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            display = drawer.render(base_frame)

        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('u'):      # undo
            if drawer.current_pts:
                drawer.current_pts.pop()
        elif key == ord('c'):      # clear
            drawer.current_pts = []

    cap.release()
    cv2.destroyAllWindows()

    if not drawer.saved_zones:
        print("[draw_zones] No zones saved. cameras.yaml unchanged.")
        return

    # ── Write updated zones back to cameras.yaml ──────────────────────────────
    # Find the camera entry and update its zones
    for cam in data.get("cameras", []):
        if str(cam.get("id", "")) == args.camera:
            cam["zones"] = drawer.saved_zones
            break

    _save_yaml(data)
    print(f"[draw_zones] Saved {len(drawer.saved_zones)} zone(s) for camera {args.camera}:")
    for z in drawer.saved_zones:
        print(f"  • {z['id']} — {z['name']} ({len(z['points'])} pts)")


if __name__ == "__main__":
    main()
