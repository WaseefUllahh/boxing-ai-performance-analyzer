"""
src/defense_detector.py — Defensive action classifier.

Responsibilities
----------------
- Classify defensive postures as: GUARD, SLIP, DUCK, CLINCH, or NONE.
- Use PoseFeatures + a small history buffer per fighter.
- Pure heuristic (no ML model needed).

Classification logic
--------------------
GUARD   : Both wrists within ``guard_wrist_head_ratio`` × shoulder_width
          of the head position.
DUCK    : Hip centre drops significantly relative to its smoothed baseline
          (hip_drop_ratio × frame_height).
SLIP    : Torso leans sideways by more than ``slip_lateral_deg`` degrees.
CLINCH  : Two fighters' bounding box centres are very close together
          (requires both fighters' features — handled in fight_analyzer).

Output
------
A DefenseResult dataclass per fighter per frame.
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

DEFENSE_LABELS = ("NONE", "GUARD", "SLIP", "DUCK", "CLINCH")


@dataclass
class DefenseResult:
    track_id: int
    label: str = "NONE"       # one of DEFENSE_LABELS
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Per-fighter state
# ---------------------------------------------------------------------------

class _FighterDefenseState:
    """Rolling hip-height baseline for DUCK detection."""

    def __init__(self, window: int = CFG.SMOOTHING_WINDOW * 3) -> None:
        self.hip_history: deque[float] = deque(maxlen=window)

    def baseline_hip(self) -> float:
        """Mean hip_height_norm over the recent window (standing baseline)."""
        if not self.hip_history:
            return 0.5
        return float(np.mean(self.hip_history))


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DefenseDetector:
    """
    Frame-by-frame, stateful defensive action classifier.

    Call ``detect(features)`` every frame for each tracked fighter.
    """

    def __init__(
        self,
        guard_wrist_head_ratio: float = CFG.GUARD_WRIST_HEAD_RATIO,
        duck_hip_drop_ratio:    float = CFG.DUCK_HIP_DROP_RATIO,
        slip_lateral_deg:       float = 15.0,   # degrees of torso lean for a slip
    ) -> None:
        self.guard_ratio     = guard_wrist_head_ratio
        self.duck_ratio      = duck_hip_drop_ratio
        self.slip_deg        = slip_lateral_deg
        self._states: dict[int, _FighterDefenseState] = {}

    # ------------------------------------------------------------------
    def detect(self, features: PoseFeatures) -> DefenseResult:
        """
        Classify one fighter's defensive posture for the current frame.

        Parameters
        ----------
        features : PoseFeatures

        Returns
        -------
        DefenseResult
        """
        tid = features.track_id
        if tid not in self._states:
            self._states[tid] = _FighterDefenseState()

        state = self._states[tid]

        if not features.valid:
            return DefenseResult(track_id=tid)

        # Update hip baseline (only when fighter is likely standing still)
        state.hip_history.append(features.hip_height_norm)

        # ── DUCK: hips drop below baseline ───────────────────────────────
        baseline = state.baseline_hip()
        hip_drop = features.hip_height_norm - baseline   # positive = lower on screen
        if hip_drop > self.duck_ratio:
            return DefenseResult(track_id=tid, label="DUCK",
                                 confidence=min(1.0, hip_drop / self.duck_ratio))

        # ── GUARD: both wrists near head ──────────────────────────────────
        if features.left_guard and features.right_guard:
            return DefenseResult(track_id=tid, label="GUARD", confidence=0.8)

        # ── SLIP: significant lateral torso lean ─────────────────────────
        if abs(features.torso_lean_deg) > self.slip_deg:
            conf = min(1.0, abs(features.torso_lean_deg) / (self.slip_deg * 2))
            return DefenseResult(track_id=tid, label="SLIP", confidence=conf)

        return DefenseResult(track_id=tid)
