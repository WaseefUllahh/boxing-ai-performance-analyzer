"""
src/pose_features.py — Keypoint → numerical feature extraction.

Responsibilities
----------------
- Convert raw (17, 3) COCO keypoint arrays into named, interpretable
  features that the strike and defense detectors can reason about.
- All computation is pure NumPy; no I/O, no model calls.

Outputs  (PoseFeatures dataclass)
----------------------------------
- wrist velocities (left / right)
- arm extension ratios (left / right)
- shoulder width (normalisation reference)
- head position
- hip centre position
- torso lean angle
- guard flags (wrists near head)
- stance width (ankle distance)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Import keypoint index map from config (no circular deps — config is pure data)
from config import CFG

KP = CFG.KP
MIN_CONF = CFG.KP_CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _kp(keypoints: np.ndarray, name: str) -> tuple[float, float, float]:
    """
    Return (x, y, confidence) for a named keypoint.

    Returns (0, 0, 0) when confidence is below threshold or the index is out
    of range — so callers should always check the confidence field.
    """
    idx = KP.get(name)
    if idx is None or idx >= len(keypoints):
        return (0.0, 0.0, 0.0)
    x, y, c = float(keypoints[idx, 0]), float(keypoints[idx, 1]), float(keypoints[idx, 2])
    if c < MIN_CONF:
        return (0.0, 0.0, 0.0)
    return (x, y, c)


def _dist(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    """Euclidean distance between two (x, y, _) tuples.  Returns 0 if either is invalid."""
    if p1[2] == 0.0 or p2[2] == 0.0:
        return 0.0
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PoseFeatures:
    """All derived features for a single fighter in a single frame."""

    track_id: int = -1

    # ── Arm extension ratios (wrist-to-shoulder / forearm length) ─────────
    # 0 = fully bent, 1 = fully extended.  >1 is possible (arm fully out).
    left_arm_extension: float = 0.0
    right_arm_extension: float = 0.0

    # ── Wrist positions (absolute pixels) ────────────────────────────────
    left_wrist_xy: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    right_wrist_xy: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    # ── Head position ─────────────────────────────────────────────────────
    head_xy: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    # ── Hip centre ───────────────────────────────────────────────────────
    hip_centre_xy: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    # ── Shoulder width (pixels) — used as a normalisation scale ──────────
    shoulder_width: float = 1.0          # never 0 (guarded below)

    # ── Torso lean (degrees; positive = leaning right in image coords) ────
    torso_lean_deg: float = 0.0

    # ── Guard detection: is wrist close to head? ──────────────────────────
    left_guard: bool = False
    right_guard: bool = False

    # ── Stance width (ankle separation, normalised by shoulder width) ─────
    stance_width_norm: float = 0.0

    # ── Hip height normalised (fraction of frame height; set externally) ──
    hip_height_norm: float = 0.0

    # ── Raw validity — False means too many keypoints were missing ────────
    valid: bool = True


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class PoseFeatureExtractor:
    """
    Converts a raw (17, 3) YOLO keypoint array into a PoseFeatures instance.

    Frame height is needed to normalise vertical positions.
    """

    def extract(
        self,
        keypoints: np.ndarray,
        track_id: int,
        frame_height: int,
        guard_wrist_head_ratio: float = 0.60,
    ) -> PoseFeatures:
        """
        Parameters
        ----------
        keypoints : np.ndarray, shape (17, 3)
            COCO keypoints [x, y, confidence].
        track_id : int
            Fighter identity.
        frame_height : int
            Height of the video frame in pixels (for normalisation).
        guard_wrist_head_ratio : float
            Fraction of shoulder width below which a wrist is "near" the head.

        Returns
        -------
        PoseFeatures
        """
        feats = PoseFeatures(track_id=track_id)

        # ── Named keypoints ───────────────────────────────────────────────
        nose         = _kp(keypoints, "nose")
        l_shoulder   = _kp(keypoints, "left_shoulder")
        r_shoulder   = _kp(keypoints, "right_shoulder")
        l_elbow      = _kp(keypoints, "left_elbow")
        r_elbow      = _kp(keypoints, "right_elbow")
        l_wrist      = _kp(keypoints, "left_wrist")
        r_wrist      = _kp(keypoints, "right_wrist")
        l_hip        = _kp(keypoints, "left_hip")
        r_hip        = _kp(keypoints, "right_hip")
        l_ankle      = _kp(keypoints, "left_ankle")
        r_ankle      = _kp(keypoints, "right_ankle")

        # Require at minimum both shoulders to produce meaningful features
        if l_shoulder[2] == 0.0 and r_shoulder[2] == 0.0:
            feats.valid = False
            return feats

        # ── Shoulder width ────────────────────────────────────────────────
        sw = _dist(l_shoulder, r_shoulder)
        feats.shoulder_width = max(sw, 1.0)   # guard against division by zero

        # ── Arm extension ratios ──────────────────────────────────────────
        # Upper arm = shoulder → elbow; forearm = elbow → wrist
        # Extension = (shoulder-to-wrist) / (upper-arm + forearm)
        feats.left_arm_extension = self._arm_extension(l_shoulder, l_elbow, l_wrist)
        feats.right_arm_extension = self._arm_extension(r_shoulder, r_elbow, r_wrist)

        # ── Wrist positions ───────────────────────────────────────────────
        feats.left_wrist_xy  = (l_wrist[0], l_wrist[1]) if l_wrist[2] > 0 else (0.0, 0.0)
        feats.right_wrist_xy = (r_wrist[0], r_wrist[1]) if r_wrist[2] > 0 else (0.0, 0.0)

        # ── Head position (use nose; fall back to ear midpoint) ───────────
        if nose[2] > 0:
            feats.head_xy = (nose[0], nose[1])
        else:
            l_ear = _kp(keypoints, "left_ear")
            r_ear = _kp(keypoints, "right_ear")
            if l_ear[2] > 0 and r_ear[2] > 0:
                feats.head_xy = ((l_ear[0] + r_ear[0]) / 2, (l_ear[1] + r_ear[1]) / 2)

        # ── Hip centre ────────────────────────────────────────────────────
        if l_hip[2] > 0 and r_hip[2] > 0:
            feats.hip_centre_xy = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        elif l_hip[2] > 0:
            feats.hip_centre_xy = (l_hip[0], l_hip[1])
        elif r_hip[2] > 0:
            feats.hip_centre_xy = (r_hip[0], r_hip[1])

        # ── Hip height normalised ─────────────────────────────────────────
        hip_y = feats.hip_centre_xy[1]
        feats.hip_height_norm = hip_y / max(frame_height, 1)

        # ── Torso lean (angle of shoulder midpoint → hip midpoint vector) ─
        if (l_shoulder[2] > 0 and r_shoulder[2] > 0 and
                feats.hip_centre_xy != (0.0, 0.0)):
            shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
            shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
            dx = feats.hip_centre_xy[0] - shoulder_mid_x
            dy = feats.hip_centre_xy[1] - shoulder_mid_y
            feats.torso_lean_deg = float(np.degrees(np.arctan2(dx, dy)))

        # ── Guard detection ───────────────────────────────────────────────
        guard_dist = guard_wrist_head_ratio * feats.shoulder_width
        if feats.head_xy != (0.0, 0.0):
            if l_wrist[2] > 0:
                d = np.hypot(l_wrist[0] - feats.head_xy[0],
                             l_wrist[1] - feats.head_xy[1])
                feats.left_guard = bool(d < guard_dist)
            if r_wrist[2] > 0:
                d = np.hypot(r_wrist[0] - feats.head_xy[0],
                             r_wrist[1] - feats.head_xy[1])
                feats.right_guard = bool(d < guard_dist)

        # ── Stance width (normalised by shoulder width) ───────────────────
        if l_ankle[2] > 0 and r_ankle[2] > 0:
            ankle_dist = np.hypot(l_ankle[0] - r_ankle[0], l_ankle[1] - r_ankle[1])
            feats.stance_width_norm = ankle_dist / feats.shoulder_width

        return feats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _arm_extension(
        shoulder: tuple[float, float, float],
        elbow:    tuple[float, float, float],
        wrist:    tuple[float, float, float],
    ) -> float:
        """
        Compute arm extension ratio in [0, ∞).

        0   = fully bent / keypoints missing
        ~1  = fully extended
        """
        if shoulder[2] == 0.0 or wrist[2] == 0.0:
            return 0.0
        shoulder_wrist = np.hypot(shoulder[0] - wrist[0], shoulder[1] - wrist[1])
        if elbow[2] > 0:
            upper = np.hypot(shoulder[0] - elbow[0], shoulder[1] - elbow[1])
            fore  = np.hypot(elbow[0]   - wrist[0],  elbow[1]   - wrist[1])
            total = upper + fore
        else:
            # Elbow not visible — approximate total arm length from shoulder width
            # (typical human arm ≈ 1.5× shoulder width)
            total = np.hypot(shoulder[0] - wrist[0], shoulder[1] - wrist[1])
        return float(shoulder_wrist / max(total, 1.0))
