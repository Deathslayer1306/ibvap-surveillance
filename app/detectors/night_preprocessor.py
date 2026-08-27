"""
Night Pre-Processor — CLAHE-based low-light enhancement.
=========================================================
If enabled (per-camera night_mode flag in cameras.yaml), converts each frame
from BGR → LAB colour space, applies CLAHE to the Luminance (L) channel only,
then converts back to BGR.  This enhances contrast in dark scenes without
over-saturating colours.

Global CLAHE parameters (clip_limit, tile_grid) come from cameras.yaml:
  settings.night_mode.clip_limit  (default 2.0)
  settings.night_mode.tile_grid   (default [8, 8])

Usage
-----
    pre = NightPreprocessor(clip_limit=2.0, tile_grid=(8, 8))
    frame = pre.process(frame, enabled=camera_cfg["night_mode"])
"""
from __future__ import annotations

import cv2
import numpy as np


class NightPreprocessor:
    """
    Applies CLAHE to the L-channel of a BGR frame.

    The CLAHE object is created once and reused (it is thread-safe for
    sequential single-frame calls as used in the pipeline).
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)) -> None:
        self._clip_limit = clip_limit
        self._tile_grid  = tuple(tile_grid)   # ensure tuple even if list from YAML
        # cv2.createCLAHE is cheap and thread-safe for per-frame application
        self._clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=tuple(tile_grid),
        )

    def process(self, frame: np.ndarray, enabled: bool = True) -> np.ndarray:
        """
        Apply CLAHE enhancement if *enabled* is True; return frame unchanged otherwise.

        Args:
            frame:   BGR numpy array (H × W × 3, uint8).
            enabled: Per-camera flag from cameras.yaml → night_mode.

        Returns:
            Enhanced BGR frame (same shape, same dtype).
        """
        if not enabled or frame is None or frame.ndim != 3:
            return frame

        try:
            lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq  = self._clahe.apply(l)
            lab_eq = cv2.merge([l_eq, a, b])
            return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        except Exception:
            # Never let a pre-processing failure crash the pipeline
            return frame
