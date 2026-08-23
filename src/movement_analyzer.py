"""
src/movement_analyzer.py — Calculates relative movement, advancing/retreating, and stance.

Responsibilities
----------------
- Estimate stance (Orthodox/Southpaw) via heuristic foot distance.
- Calculate relative movement vectors (advancing/retreating/lateral).
- Accumulate spatial movement volume (head, feet, center).
- Calculate simple activity scores.

Changes (audit fixes)
---------------------
- Head movement: uses body-relative head velocity (head_shoulder_velocity from
  SmoothedFeatures) normalized by shoulder width → result is in shoulder-widths/frame,
  dimensionless and scale-invariant.  Frames with relative head displacement above
  HEAD_MOVEMENT_JITTER_GATE_SW shoulder-widths are rejected as residual jitter.
  `total_head_movement` now accumulates normalized dimensionless values, not pixels.
- Foot movement: uses real ankle velocities (left_ankle_velocity / right_ankle_velocity)
  from SmoothedFeatures.  Frames with no valid ankle data are tracked separately in
  `ankle_frames_missing`.  No alias to center movement.
- `MovementStats` gains `ankle_frames_valid` and `ankle_frames_missing` counters.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Optional, Dict

from config import CFG
from src.pose_features import PoseFeatures, distance, magnitude
from src.temporal_features import SmoothedFeatures


@dataclass
class MovementStats:
    """Cumulative movement metrics for a single fighter."""
    fighter_id: int

    # State tracking
    current_stance: str = "UNKNOWN"
    current_movement_state: str = "STATIONARY"  # ADVANCING / RETREATING / LATERAL / STATIONARY

    # Distance
    fighter_separation: Optional[float] = None

    # Accumulators
    # total_head_movement: dimensionless (shoulder-widths/frame accumulated).
    #   Was pixels/frame — now normalized and jitter-gated.
    total_head_movement: float = 0.0
    # total_foot_movement: real ankle velocity magnitude (pixels/frame accumulated).
    #   Was a hardcoded alias to total_center_movement — now independent.
    total_foot_movement: float = 0.0
    total_center_movement: float = 0.0

    # Ankle data quality counters
    ankle_frames_valid: int = 0    # frames where at least one ankle velocity was valid
    ankle_frames_missing: int = 0  # frames where no ankle data was available

    # Frame counters
    frames_advancing: int = 0
    frames_retreating: int = 0
    frames_stationary: int = 0

    # Activity score heuristic
    activity_score: float = 0.0


class MovementAnalyzer:
    def __init__(self):
        self.stance_window    = getattr(CFG, 'STANCE_CONFIDENCE_FRAMES', 30)
        self.stance_foot_ratio = getattr(CFG, 'STANCE_FOOT_DIST_RATIO', 0.1)
        self.advance_vel      = getattr(CFG, 'MOVEMENT_ADVANCE_VELOCITY', 4.0)
        self.retreat_vel      = getattr(CFG, 'MOVEMENT_RETREAT_VELOCITY', -4.0)
        self.min_pixels       = getattr(CFG, 'MOVEMENT_MIN_PIXELS', 3.0)
        self.head_jitter_gate = getattr(CFG, 'HEAD_MOVEMENT_JITTER_GATE_SW', 0.30)
        self.min_ankle_conf   = getattr(CFG, 'MIN_ANKLE_KP_CONF', 0.25)

        self.stats: Dict[int, MovementStats] = {}
        self._stance_buffers: Dict[int, collections.deque] = {}

    def update(
        self,
        all_features: Dict[int, PoseFeatures],
        all_smoothed: Dict[int, SmoothedFeatures],
        frame_idx: int,
    ) -> Dict[int, MovementStats]:

        # Initialize missing state
        for tid in all_features.keys():
            if tid not in self.stats:
                self.stats[tid] = MovementStats(fighter_id=tid)
                self._stance_buffers[tid] = collections.deque(maxlen=self.stance_window)

        # Relative movement needs exactly 2 fighters
        tids = list(all_smoothed.keys())
        if len(tids) == 2:
            f1, f2 = tids[0], tids[1]
            sf1, sf2 = all_smoothed[f1], all_smoothed[f2]

            if sf1.body_center and sf2.body_center:
                sep = distance(sf1.body_center, sf2.body_center)
                self.stats[f1].fighter_separation = sep
                self.stats[f2].fighter_separation = sep

                self._analyze_relative_movement(f1, sf1, sf2)
                self._analyze_relative_movement(f2, sf2, sf1)

                self._estimate_stance(f1, sf1, sf2, all_features[f1].shoulder_width)
                self._estimate_stance(f2, sf2, sf1, all_features[f2].shoulder_width)
        else:
            for tid in tids:
                self.stats[tid].current_movement_state = "STATIONARY"
                self.stats[tid].current_stance = "UNKNOWN"

        # ── Accumulate volume metrics ───────────────────────────────────────
        for tid, sf in all_smoothed.items():
            stat = self.stats[tid]
            feat = all_features.get(tid)
            sw = feat.shoulder_width if feat and feat.shoulder_width else None

            # ── Head movement (body-relative, normalized, jitter-gated) ──────
            # head_shoulder_velocity is the frame-to-frame change in
            # (head_center − shoulder_center), which removes body translation.
            # Normalizing by shoulder_width makes it scale-invariant.
            if sf.head_shoulder_velocity is not None and sw and sw > 0:
                rel_m = magnitude(sf.head_shoulder_velocity)
                if rel_m is not None:
                    norm_m = rel_m / sw  # shoulder-widths / frame
                    # Reject residual jitter above gate threshold
                    if norm_m <= self.head_jitter_gate:
                        stat.total_head_movement += norm_m

            # ── Center volume (unchanged) ─────────────────────────────────────
            if sf.body_velocity:
                m = magnitude(sf.body_velocity)
                if m and m > self.min_pixels and m < 120.0:
                    stat.total_center_movement += m

            # ── Foot movement (real ankle velocities) ─────────────────────────
            # Uses left_ankle_velocity and right_ankle_velocity from SmoothedFeatures.
            # Both have been jitter-gated inside TemporalFeatureManager (_PointHistory).
            # We apply an additional hard cap at 120 px/frame and a minimum motion floor.
            left_v  = magnitude(sf.left_ankle_velocity)  if sf.left_ankle_velocity  else None
            right_v = magnitude(sf.right_ankle_velocity) if sf.right_ankle_velocity else None

            valid_ankle_vels = [
                v for v in [left_v, right_v]
                if v is not None and self.min_pixels < v < 120.0
            ]

            if valid_ankle_vels:
                avg_ankle_vel = sum(valid_ankle_vels) / len(valid_ankle_vels)
                stat.total_foot_movement += avg_ankle_vel
                stat.ankle_frames_valid += 1
            else:
                stat.ankle_frames_missing += 1

        return self.stats

    def _analyze_relative_movement(
        self,
        tid: int,
        sf: SmoothedFeatures,
        opp_sf: SmoothedFeatures,
    ):
        """Calculate projection of fighter's velocity onto vector pointing at opponent."""
        stat = self.stats[tid]

        if not sf.body_center or not opp_sf.body_center or not sf.body_velocity:
            stat.current_movement_state = "STATIONARY"
            stat.frames_stationary += 1
            return

        bx, by = sf.body_center
        ox, oy = opp_sf.body_center
        vx, vy = sf.body_velocity

        v_mag = magnitude((vx, vy))
        if v_mag is None or v_mag < self.min_pixels:
            stat.current_movement_state = "STATIONARY"
            stat.frames_stationary += 1
            return

        dx = ox - bx
        dy = oy - by
        dist = magnitude((dx, dy))

        if dist and dist > 0:
            nx = dx / dist
            ny = dy / dist
            projection = (vx * nx) + (vy * ny)

            if projection > self.advance_vel:
                stat.current_movement_state = "ADVANCING"
                stat.frames_advancing += 1
            elif projection < self.retreat_vel:
                stat.current_movement_state = "RETREATING"
                stat.frames_retreating += 1
            else:
                stat.current_movement_state = "LATERAL"
                stat.frames_stationary += 1
        else:
            stat.current_movement_state = "STATIONARY"
            stat.frames_stationary += 1

    def _estimate_stance(
        self,
        tid: int,
        sf: SmoothedFeatures,
        opp_sf: SmoothedFeatures,
        shoulder_width: Optional[float],
    ):
        """
        Estimate Orthodox vs Southpaw using foot proximity to opponent.
        Left foot closer = Orthodox. Right foot closer = Southpaw.
        """
        stat   = self.stats[tid]
        buffer = self._stance_buffers[tid]

        if not sf.left_ankle or not sf.right_ankle or not opp_sf.body_center or not shoulder_width:
            buffer.append("UNKNOWN")
        else:
            d_left  = distance(sf.left_ankle,  opp_sf.body_center)
            d_right = distance(sf.right_ankle, opp_sf.body_center)

            if d_left is not None and d_right is not None:
                diff  = d_right - d_left
                ratio = diff / shoulder_width

                if ratio > self.stance_foot_ratio:
                    buffer.append("ORTHODOX")
                elif ratio < -self.stance_foot_ratio:
                    buffer.append("SOUTHPAW")
                else:
                    buffer.append("UNKNOWN")
            else:
                buffer.append("UNKNOWN")

        if len(buffer) > 0:
            counts = collections.Counter(buffer)
            most_common, count = counts.most_common(1)[0]
            if count > len(buffer) * 0.5:
                stat.current_stance = most_common
