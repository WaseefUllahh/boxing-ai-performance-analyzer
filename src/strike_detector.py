"""
src/strike_detector.py — Rule-based punch classifier.

Responsibilities
----------------
- Classify punches as: JAB, CROSS, HOOK, UPPERCUT, or NONE.
- Use PoseFeatures + a small velocity history buffer per fighter.
- No neural network, no training data needed.

Classification logic (heuristic)
---------------------------------
A "punch event" is triggered when:
  1. Arm extension ratio exceeds PUNCH_EXTENSION_THRESHOLD.
  2. Wrist velocity exceeds PUNCH_VELOCITY_THRESHOLD (pixels / frame).
  3. Extension is sustained for at least PUNCH_MIN_FRAMES frames.

Punch type is then determined by:
  - Which arm is extended (left vs right).
  - Horizontal vs vertical wrist trajectory.
  - Height of the wrist relative to the shoulder (uppercut heuristic).

Output
------
A StrikeResult dataclass per fighter per frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from config import CFG
from src.pose_features import PoseFeatures


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

STRIKE_LABELS = ("NONE", "JAB", "CROSS", "HOOK", "UPPERCUT")


@dataclass
class StrikeResult:
    track_id: int
    label: str = "NONE"          # one of STRIKE_LABELS
    arm: str = ""                 # "left" | "right" | ""
    confidence: float = 0.0      # heuristic score 0-1


# ---------------------------------------------------------------------------
# Per-fighter state
# ---------------------------------------------------------------------------

class _FighterStrikeState:
    """Tracks velocity history and extension state for one fighter."""

    def __init__(self) -> None:
        self.left_wrist_history:  deque[tuple[float, float]] = deque(maxlen=CFG.SMOOTHING_WINDOW)
        self.right_wrist_history: deque[tuple[float, float]] = deque(maxlen=CFG.SMOOTHING_WINDOW)
        self.left_ext_frames:  int = 0   # consecutive extended frames (left arm)
        self.right_ext_frames: int = 0


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class StrikeDetector:
    """
    Frame-by-frame, stateful punch classifier.

    Call ``detect(features)`` every frame for each fighter.
    """

    def __init__(
        self,
        extension_threshold: float = CFG.PUNCH_EXTENSION_THRESHOLD,
        velocity_threshold:  float = CFG.PUNCH_VELOCITY_THRESHOLD,
        min_frames:          int   = CFG.PUNCH_MIN_FRAMES,
    ) -> None:
        self.ext_thresh = extension_threshold
        self.vel_thresh = velocity_threshold
        self.min_frames = min_frames
        self._states: dict[int, _FighterStrikeState] = {}

    # ------------------------------------------------------------------
    def detect(self, features: PoseFeatures) -> StrikeResult:
        """
        Classify the current frame's pose into a punch label.

        Parameters
        ----------
        features : PoseFeatures
            Output of PoseFeatureExtractor.extract() for this fighter.

        Returns
        -------
        StrikeResult
        """
        tid = features.track_id
        if tid not in self._states:
            self._states[tid] = _FighterStrikeState()

        state = self._states[tid]

        if not features.valid:
            return StrikeResult(track_id=tid)

        # ── Update wrist history ──────────────────────────────────────────
        state.left_wrist_history.append(features.left_wrist_xy)
        state.right_wrist_history.append(features.right_wrist_xy)

        # ── Compute velocities ────────────────────────────────────────────
        l_vel = self._velocity(state.left_wrist_history)
        r_vel = self._velocity(state.right_wrist_history)

        # ── Check extension + velocity thresholds ────────────────────────
        l_extended = (features.left_arm_extension  >= self.ext_thresh and l_vel >= self.vel_thresh)
        r_extended = (features.right_arm_extension >= self.ext_thresh and r_vel >= self.vel_thresh)

        # Count consecutive extension frames
        state.left_ext_frames  = (state.left_ext_frames  + 1) if l_extended  else 0
        state.right_ext_frames = (state.right_ext_frames + 1) if r_extended  else 0

        # ── Trigger punch event ───────────────────────────────────────────
        if state.left_ext_frames >= self.min_frames:
            label, hconf = self._classify(
                "left", features, state.left_wrist_history, l_vel
            )
            return StrikeResult(track_id=tid, label=label, arm="left", confidence=hconf)

        if state.right_ext_frames >= self.min_frames:
            label, hconf = self._classify(
                "right", features, state.right_wrist_history, r_vel
            )
            return StrikeResult(track_id=tid, label=label, arm="right", confidence=hconf)

        return StrikeResult(track_id=tid)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _velocity(history: deque[tuple[float, float] | None]) -> float:
        """Mean pixel velocity over the last two valid positions."""
        valid = [p for p in history if p is not None]
        if len(valid) < 2:
            return 0.0
        p1, p2 = valid[-2], valid[-1]
        
        # We can import pose_features if we want, but np.hypot is fine
        from src.pose_features import distance
        dist = distance(p1, p2)
        return dist if dist is not None else 0.0

    @staticmethod
    def _classify(
        arm: str,
        features: PoseFeatures,
        history: deque[tuple[float, float] | None],
        velocity: float,
    ) -> tuple[str, float]:
        """
        Heuristic punch-type classifier.

        Returns (label, confidence_score).
        """
        valid = [p for p in history if p is not None]
        if len(valid) < 2:
            return ("JAB" if arm == "left" else "CROSS", 0.5)

        p_prev, p_curr = valid[-2], valid[-1]
        dx = p_curr[0] - p_prev[0]
        dy = p_curr[1] - p_prev[1]

        # Wrist height relative to shoulder
        if arm == "left":
            wrist_y = features.left_wrist[1] if features.left_wrist else 0.0
            ext = features.left_arm_extension
        else:
            wrist_y = features.right_wrist[1] if features.right_wrist else 0.0
            ext = features.right_arm_extension

        shoulder_y = features.head_center[1] + (features.shoulder_width or 0.0) if features.head_center else 0.0

        # Uppercut: wrist moving upward significantly
        sw = features.shoulder_width or 1.0
        if dy < -0.5 * sw and wrist_y < shoulder_y:
            return ("UPPERCUT", min(1.0, ext))

        # Hook: large horizontal component, wrist not fully extended forward
        if abs(dx) > abs(dy) * 1.5 and ext < 0.85:
            return ("HOOK", min(1.0, ext))

        # Jab = lead hand (left for orthodox), Cross = rear hand
        if arm == "left":
            return ("JAB", min(1.0, ext))
        return ("CROSS", min(1.0, ext))

