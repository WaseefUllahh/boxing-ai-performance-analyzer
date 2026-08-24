"""
src/temporal_features.py — Stateful temporal smoothing and motion derivation.

Responsibilities
----------------
- Maintain bounded rolling histories of keypoints per fighter.
- Compute Exponential Moving Average (EMA) for noisy points with Kinematic Coasting.
- Compute velocity and acceleration.
- Gracefully handle micro-occlusions (up to 5 frames) using damped velocity extrapolation.

Features & Fixes
----------------
- _PointHistory: 5-frame velocity-guided kinematic coasting buffer with 0.85 decay.
- _PointHistory: first-frame guard — vel/acc are None on initialization (no velocity
  without a prior frame), preventing spurious first-frame dodge/movement events.
- _PointHistory: adaptive jitter gate with multi-frame recovery — single-frame isolated
  teleportation spikes are rejected, while genuine tracker shifts (>1 frame) safely
  re-anchor without permanently locking the EMA.
- Joint-appropriate jump limits: wrists are given a wider dynamic range (250 px) to
  accommodate fast punch acceleration, while head/torso have stricter limits (120 px).
- SmoothedFeatures: includes shoulder_center, head_shoulder_velocity (body-relative head
  motion for dodge detection), left_ankle_velocity, right_ankle_velocity (for real
  foot movement tracking).
- TemporalFeatureManager: persistent emission of canonical fighters across micro-occlusions.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Optional, Dict, List

from config import CFG
from src.pose_features import PoseFeatures, Point, Vector, distance, velocity, acceleration

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SmoothedFeatures:
    """Smoothed points and their derivatives for a single fighter in a single frame."""
    track_id: int

    # Smoothed positions (EMA)
    head_center: Optional[Point] = None
    shoulder_center: Optional[Point] = None
    body_center: Optional[Point] = None
    left_wrist: Optional[Point] = None
    right_wrist: Optional[Point] = None
    left_elbow: Optional[Point] = None
    right_elbow: Optional[Point] = None
    left_ankle: Optional[Point] = None
    right_ankle: Optional[Point] = None

    # Velocities (pixels/frame)
    head_velocity: Optional[Vector] = None
    body_velocity: Optional[Vector] = None
    left_wrist_velocity: Optional[Vector] = None
    right_wrist_velocity: Optional[Vector] = None

    # Body-relative head motion (head position minus shoulder center, frame-to-frame change).
    # Cancels body translation/camera pan and isolates head bob/weave.
    head_shoulder_velocity: Optional[Vector] = None

    # Ankle velocities for real foot movement tracking
    left_ankle_velocity: Optional[Vector] = None
    right_ankle_velocity: Optional[Vector] = None

    # Accelerations (pixels/frame^2)
    left_wrist_acceleration: Optional[Vector] = None
    right_wrist_acceleration: Optional[Vector] = None


class _PointHistory:
    """
    Tracks and smoothes a single body point across frames using EMA
    with a 5-frame velocity-guided kinematic coasting buffer for micro-occlusions.
    """

    def __init__(self, maxlen: int, alpha: float, max_jump_px: Optional[float] = 120.0, max_coast_frames: int = 5):
        self.maxlen = maxlen
        self.alpha = alpha
        self.max_jump_px = max_jump_px
        self._max_jump_sq = (max_jump_px * max_jump_px) if max_jump_px is not None else None
        self.max_coast_frames = max_coast_frames
        self.coast_count = 0
        self._initialized = False
        self._outlier_count = 0

        self.raw_buffer: collections.deque[Optional[Point]] = collections.deque(maxlen=maxlen)
        self.ema_pos: Optional[Point] = None
        self.prev_ema_pos: Optional[Point] = None
        self.vel: Optional[Vector] = None
        self.prev_vel: Optional[Vector] = None
        self.acc: Optional[Vector] = None

    def update(self, pt: Optional[Point]) -> None:
        self.raw_buffer.append(pt)

        # ── 1. Handle Missing Keypoint (Kinematic Coasting) ───────────────
        if pt is None:
            if self._initialized and self.ema_pos is not None and self.coast_count < self.max_coast_frames:
                self.coast_count += 1
                self.prev_ema_pos = self.ema_pos
                self.prev_vel = self.vel
                
                # Extrapolate position using damped velocity
                if self.vel is not None:
                    decay = 0.85
                    self.ema_pos = (
                        self.ema_pos[0] + self.vel[0] * decay,
                        self.ema_pos[1] + self.vel[1] * decay,
                    )
                    self.vel = (self.vel[0] * decay, self.vel[1] * decay)
                    self.acc = (0.0, 0.0)
                else:
                    self.vel = (0.0, 0.0)
                    self.acc = (0.0, 0.0)
                return
            else:
                # Expired past grace period -> Reset state
                self.ema_pos = None
                self.prev_ema_pos = None
                self.vel = None
                self.prev_vel = None
                self.acc = None
                self._initialized = False
                self.coast_count = 0
                return

        # ── 2. Handle Valid Keypoint ──────────────────────────────────────
        self.coast_count = 0

        # First-frame initialization
        if not self._initialized or self.ema_pos is None:
            self.ema_pos = pt
            self.prev_ema_pos = None
            self.vel = None
            self.acc = None
            self._initialized = True
            self._outlier_count = 0
            return

        # Jitter Gate
        if self._max_jump_sq is not None:
            dx = pt[0] - self.ema_pos[0]
            dy = pt[1] - self.ema_pos[1]
            if (dx * dx + dy * dy) > self._max_jump_sq:
                self._outlier_count += 1
                if self._outlier_count >= 2:
                    # 2 consecutive frames far away -> Genuine relocation / re-acquisition
                    self.prev_ema_pos = None
                    self.ema_pos = pt
                    self.vel = None
                    self.acc = None
                    self._outlier_count = 0
                    return
                else:
                    # Isolated spike -> Hold EMA position
                    self.prev_ema_pos = self.ema_pos
                    return
            else:
                self._outlier_count = 0

        # EMA Smoothing & Derivatives
        self.prev_ema_pos = self.ema_pos
        self.prev_vel = self.vel

        self.ema_pos = (
            self.ema_pos[0] * (1.0 - self.alpha) + pt[0] * self.alpha,
            self.ema_pos[1] * (1.0 - self.alpha) + pt[1] * self.alpha,
        )

        if self.prev_ema_pos is not None:
            self.vel = velocity(self.prev_ema_pos, self.ema_pos)
        if self.prev_vel is not None and self.vel is not None:
            self.acc = acceleration(self.prev_vel, self.vel)


class _FighterTemporalState:
    """Holds the independent _PointHistory for all tracked joints of a fighter."""
    def __init__(self, maxlen: int, alpha: float):
        # Joint-specific maximum single-frame jump limits
        self.head     = _PointHistory(maxlen, alpha, max_jump_px=120.0)
        self.shoulder = _PointHistory(maxlen, alpha, max_jump_px=120.0)
        self.body     = _PointHistory(maxlen, alpha, max_jump_px=120.0)
        self.l_wrist  = _PointHistory(maxlen, alpha, max_jump_px=250.0)  # Wrists move fast during strikes
        self.r_wrist  = _PointHistory(maxlen, alpha, max_jump_px=250.0)
        self.l_elbow  = _PointHistory(maxlen, alpha, max_jump_px=200.0)
        self.r_elbow  = _PointHistory(maxlen, alpha, max_jump_px=200.0)
        self.l_ankle  = _PointHistory(maxlen, alpha, max_jump_px=180.0)
        self.r_ankle  = _PointHistory(maxlen, alpha, max_jump_px=180.0)
        self.head_rel = _PointHistory(maxlen, alpha, max_jump_px=80.0)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class TemporalFeatureManager:
    """
    Stateful manager for temporal features across all fighters in a fight.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        maxlen: int = 15,
    ):
        self.alpha = alpha
        self.maxlen = maxlen
        self._states: Dict[int, _FighterTemporalState] = {}
        self.fighter_distance_history: collections.deque[Optional[float]] = collections.deque(maxlen=maxlen)
        self.fighter_distance: Optional[float] = None

    def update(self, features_list: List[PoseFeatures]) -> Dict[int, SmoothedFeatures]:
        """
        Takes a list of PoseFeatures for the current frame, updates histories,
        and returns a dictionary mapping track_id -> SmoothedFeatures with
        persistent emission and kinematic coasting for micro-occlusions.
        """
        smoothed_dict: Dict[int, SmoothedFeatures] = {}

        # Compute global distance if exactly 2 valid fighters exist
        valid_feats = [f for f in features_list if f.valid and f.body_center is not None]
        if len(valid_feats) == 2:
            dist = distance(valid_feats[0].body_center, valid_feats[1].body_center)
            if dist is not None:
                if self.fighter_distance is None:
                    self.fighter_distance = dist
                else:
                    self.fighter_distance = (
                        self.fighter_distance * (1 - self.alpha) + dist * self.alpha
                    )
        else:
            dist = None

        self.fighter_distance_history.append(dist)

        present_tids = {f.track_id: f for f in features_list if f.track_id > 0}

        # Ensure states exist for canonical fighters (1 and 2)
        for tid in [1, 2]:
            if tid not in self._states:
                self._states[tid] = _FighterTemporalState(self.maxlen, self.alpha)

        # Also add any dynamic track IDs present in features_list
        for tid in present_tids.keys():
            if tid not in self._states:
                self._states[tid] = _FighterTemporalState(self.maxlen, self.alpha)

        for tid, state in self._states.items():
            feats = present_tids.get(tid)

            if feats is not None and feats.valid:
                # Update individual joint histories with valid observations
                state.head.update(feats.head_center)
                state.shoulder.update(feats.shoulder_center)
                state.body.update(feats.body_center)
                state.l_wrist.update(feats.left_wrist)
                state.r_wrist.update(feats.right_wrist)
                state.l_elbow.update(feats.left_elbow)
                state.r_elbow.update(feats.right_elbow)
                state.l_ankle.update(feats.left_ankle)
                state.r_ankle.update(feats.right_ankle)
            else:
                # Advance kinematic coasting buffer for missing / occluded fighter
                state.head.update(None)
                state.shoulder.update(None)
                state.body.update(None)
                state.l_wrist.update(None)
                state.r_wrist.update(None)
                state.l_elbow.update(None)
                state.r_elbow.update(None)
                state.l_ankle.update(None)
                state.r_ankle.update(None)

            # ── Body-relative head offset ──────────────────────────────────
            if state.head.ema_pos is not None and state.shoulder.ema_pos is not None:
                rel_offset: Optional[Point] = (
                    state.head.ema_pos[0] - state.shoulder.ema_pos[0],
                    state.head.ema_pos[1] - state.shoulder.ema_pos[1],
                )
            else:
                rel_offset = None
            state.head_rel.update(rel_offset)

            # Package smoothed results if core body anchor is alive
            if state.shoulder.ema_pos is not None or state.head.ema_pos is not None or state.body.ema_pos is not None:
                sf = SmoothedFeatures(
                    track_id=tid,
                    head_center=state.head.ema_pos,
                    shoulder_center=state.shoulder.ema_pos,
                    body_center=state.body.ema_pos,
                    left_wrist=state.l_wrist.ema_pos,
                    right_wrist=state.r_wrist.ema_pos,
                    left_elbow=state.l_elbow.ema_pos,
                    right_elbow=state.r_elbow.ema_pos,
                    left_ankle=state.l_ankle.ema_pos,
                    right_ankle=state.r_ankle.ema_pos,
                    head_velocity=state.head.vel,
                    body_velocity=state.body.vel,
                    left_wrist_velocity=state.l_wrist.vel,
                    right_wrist_velocity=state.r_wrist.vel,
                    left_wrist_acceleration=state.l_wrist.acc,
                    right_wrist_acceleration=state.r_wrist.acc,
                    head_shoulder_velocity=state.head_rel.vel,
                    left_ankle_velocity=state.l_ankle.vel,
                    right_ankle_velocity=state.r_ankle.vel,
                )
                smoothed_dict[tid] = sf

        return smoothed_dict

    def get_fighter_distance(self) -> Optional[float]:
        """Returns the EMA smoothed distance between the two primary fighters."""
        return self.fighter_distance
