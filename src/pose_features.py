"""
src/pose_features.py — Keypoint → numerical feature extraction and Geometry layer.

Responsibilities
----------------
- Provide robust geometric primitives that handle missing data (`None`) gracefully.
- Convert raw (17, 3) COCO keypoint arrays into named, interpretable features.
- Calculate joint centers, body orientations, arm extensions, and normalization scales.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from config import CFG

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Point = Tuple[float, float]
Vector = Tuple[float, float]

KP = CFG.KP
MIN_CONF = CFG.KP_CONFIDENCE_THRESHOLD

# ---------------------------------------------------------------------------
# Geometry Functions
# ---------------------------------------------------------------------------

def distance(p1: Optional[Point], p2: Optional[Point]) -> Optional[float]:
    """Euclidean distance between two points."""
    if p1 is None or p2 is None:
        return None
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))

def vector(p1: Optional[Point], p2: Optional[Point]) -> Optional[Vector]:
    """Vector pointing from p1 to p2."""
    if p1 is None or p2 is None:
        return None
    return (float(p2[0] - p1[0]), float(p2[1] - p1[1]))

def magnitude(v: Optional[Vector]) -> Optional[float]:
    """Magnitude (length) of a vector."""
    if v is None:
        return None
    return float(math.hypot(v[0], v[1]))

def normalize(v: Optional[Vector]) -> Optional[Vector]:
    """Return a unit vector. Returns None if length is zero."""
    if v is None:
        return None
    mag = magnitude(v)
    if mag is None or mag == 0.0:
        return None
    return (v[0] / mag, v[1] / mag)

def angle_between_vectors(v1: Optional[Vector], v2: Optional[Vector]) -> Optional[float]:
    """Returns the smallest angle between two vectors in degrees [0, 180]."""
    v1_n = normalize(v1)
    v2_n = normalize(v2)
    if v1_n is None or v2_n is None:
        return None
    dot = v1_n[0] * v2_n[0] + v1_n[1] * v2_n[1]
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))

def angle_between_points(p1: Optional[Point], p2: Optional[Point], p3: Optional[Point]) -> Optional[float]:
    """Angle at p2 formed by p1-p2-p3 in degrees [0, 180]."""
    v1 = vector(p2, p1)
    v2 = vector(p2, p3)
    return angle_between_vectors(v1, v2)

def velocity(p_prev: Optional[Point], p_curr: Optional[Point]) -> Optional[Vector]:
    """Velocity vector assuming 1 frame timestep (pixels/frame)."""
    return vector(p_prev, p_curr)

def acceleration(v_prev: Optional[Vector], v_curr: Optional[Vector]) -> Optional[Vector]:
    """Acceleration vector assuming 1 frame timestep (pixels/frame^2)."""
    return vector(v_prev, v_curr)

def midpoint(p1: Optional[Point], p2: Optional[Point]) -> Optional[Point]:
    """Midpoint between two points. If one is None, returns the other."""
    if p1 is None and p2 is None: return None
    if p1 is None: return p2
    if p2 is None: return p1
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

def _kp(keypoints: np.ndarray, name: str) -> Optional[Point]:
    """Extract a named keypoint from the YOLO pose array if confidence is sufficient."""
    idx = KP.get(name)
    if idx is None or idx >= len(keypoints):
        return None
    x, y, c = keypoints[idx]
    if c < MIN_CONF:
        return None
    return (float(x), float(y))

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PoseFeatures:
    """All derived features and geometric points for a single fighter in a single frame."""
    track_id: int = -1
    valid: bool = True

    # ── Basic Body Points ───────────────────────────────────────────────
    head_center: Optional[Point] = None
    shoulder_center: Optional[Point] = None
    hip_center: Optional[Point] = None
    body_center: Optional[Point] = None
    bbox_center: Optional[Point] = None

    left_wrist: Optional[Point] = None
    right_wrist: Optional[Point] = None
    left_elbow: Optional[Point] = None
    right_elbow: Optional[Point] = None
    left_knee: Optional[Point] = None
    right_knee: Optional[Point] = None
    left_ankle: Optional[Point] = None
    right_ankle: Optional[Point] = None

    # ── Derived Dimensions ──────────────────────────────────────────────
    shoulder_width: Optional[float] = None
    hip_width: Optional[float] = None
    body_orientation: Optional[float] = None   # angle from vertical (degrees)
    
    # ── Action specific features ─────────────────────────────────────────
    left_arm_extension: float = 0.0
    right_arm_extension: float = 0.0
    torso_lean_deg: float = 0.0
    left_guard: bool = False
    right_guard: bool = False
    stance_width_norm: float = 0.0
    hip_height_norm: float = 0.0
    
    # Keeping old properties pointing to new fields for backward-compatibility conceptually
    @property
    def left_wrist_xy(self) -> Optional[Point]: return self.left_wrist
    
    @property
    def right_wrist_xy(self) -> Optional[Point]: return self.right_wrist

    @property
    def head_xy(self) -> Optional[Point]: return self.head_center
    
    @property
    def hip_centre_xy(self) -> Optional[Point]: return self.hip_center

# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class PoseFeatureExtractor:
    
    def extract(
        self,
        keypoints: np.ndarray,
        track_id: int,
        frame_height: int,
        bbox_center: Optional[Point] = None,
        guard_wrist_head_ratio: float = 0.60,
    ) -> PoseFeatures:
        
        feats = PoseFeatures(track_id=track_id)
        feats.bbox_center = bbox_center

        # ── Raw Points ──────────────────────────────────────────────────
        nose         = _kp(keypoints, "nose")
        l_ear        = _kp(keypoints, "left_ear")
        r_ear        = _kp(keypoints, "right_ear")
        l_shoulder   = _kp(keypoints, "left_shoulder")
        r_shoulder   = _kp(keypoints, "right_shoulder")
        feats.left_elbow   = _kp(keypoints, "left_elbow")
        feats.right_elbow  = _kp(keypoints, "right_elbow")
        feats.left_wrist   = _kp(keypoints, "left_wrist")
        feats.right_wrist  = _kp(keypoints, "right_wrist")
        l_hip        = _kp(keypoints, "left_hip")
        r_hip        = _kp(keypoints, "right_hip")
        feats.left_knee    = _kp(keypoints, "left_knee")
        feats.right_knee   = _kp(keypoints, "right_knee")
        feats.left_ankle   = _kp(keypoints, "left_ankle")
        feats.right_ankle  = _kp(keypoints, "right_ankle")

        # Require at minimum both shoulders to consider pose remotely valid
        if l_shoulder is None and r_shoulder is None:
            feats.valid = False
            return feats

        # ── Centers ──────────────────────────────────────────────────────
        feats.head_center = nose if nose is not None else midpoint(l_ear, r_ear)
        feats.shoulder_center = midpoint(l_shoulder, r_shoulder)
        feats.hip_center = midpoint(l_hip, r_hip)
        
        if feats.shoulder_center is not None and feats.hip_center is not None:
            feats.body_center = midpoint(feats.shoulder_center, feats.hip_center)
        else:
            feats.body_center = None
        
        feats.shoulder_width = distance(l_shoulder, r_shoulder)
        if feats.shoulder_width == 0.0:
            feats.shoulder_width = None
            
        feats.hip_width = distance(l_hip, r_hip)

        # ── Arm Extension ───────────────────────────────────────────────
        feats.left_arm_extension = self._arm_extension(l_shoulder, feats.left_elbow, feats.left_wrist)
        feats.right_arm_extension = self._arm_extension(r_shoulder, feats.right_elbow, feats.right_wrist)

        # ── Torso lean / Body orientation ───────────────────────────────
        if feats.shoulder_center is not None and feats.hip_center is not None:
            dx = feats.hip_center[0] - feats.shoulder_center[0]
            dy = feats.hip_center[1] - feats.shoulder_center[1]
            # Torso lean: angle from vertical (y-axis)
            feats.torso_lean_deg = float(math.degrees(math.atan2(dx, dy)))
            feats.body_orientation = feats.torso_lean_deg

        # ── Hip height normalised ───────────────────────────────────────
        if feats.hip_center is not None:
            feats.hip_height_norm = feats.hip_center[1] / max(frame_height, 1)

        # ── Guard detection ─────────────────────────────────────────────
        if feats.head_center is not None and feats.shoulder_width is not None:
            guard_dist = guard_wrist_head_ratio * feats.shoulder_width
            
            d_l = distance(feats.left_wrist, feats.head_center)
            if d_l is not None:
                feats.left_guard = bool(d_l < guard_dist)
                
            d_r = distance(feats.right_wrist, feats.head_center)
            if d_r is not None:
                feats.right_guard = bool(d_r < guard_dist)

        # ── Stance width ────────────────────────────────────────────────
        if feats.shoulder_width is not None and feats.shoulder_width > 0:
            ankle_dist = distance(feats.left_ankle, feats.right_ankle)
            if ankle_dist is not None:
                feats.stance_width_norm = ankle_dist / feats.shoulder_width

        return feats

    @staticmethod
    def _arm_extension(
        shoulder: Optional[Point],
        elbow:    Optional[Point],
        wrist:    Optional[Point],
    ) -> float:
        if shoulder is None or wrist is None:
            return 0.0
        
        shoulder_wrist = distance(shoulder, wrist)
        if shoulder_wrist is None:
            return 0.0
            
        if elbow is not None:
            upper = distance(shoulder, elbow)
            fore = distance(elbow, wrist)
            if upper is not None and fore is not None:
                total = upper + fore
            else:
                total = shoulder_wrist
        else:
            # Fallback estimation
            total = shoulder_wrist
            
        if total == 0.0:
            return 0.0
            
        return float(shoulder_wrist / total)
