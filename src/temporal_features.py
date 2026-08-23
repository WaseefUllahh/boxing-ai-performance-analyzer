"""
src/temporal_features.py — Stateful temporal smoothing and motion derivation.

Responsibilities
----------------
- Maintain bounded rolling histories of keypoints per fighter.
- Compute Exponential Moving Average (EMA) for noisy points.
- Compute velocity and acceleration.
- Gracefully handle occlusions/missing points without corrupting the history buffer.

Features & Fixes
----------------
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
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Optional, Dict

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
    Tracks and smoothes a single body point across frames using EMA.

    Jitter gate & Outlier Handling:
    - If a point jumps more than max_jump_px from current EMA:
      - Single-frame spike: treated as an outlier (EMA is preserved, vel/acc zeroed).
      - Multi-frame relocation: tracker/fighter moved, re-anchors EMA to the new point.
    
    First-frame guard:
    - First valid point initializes EMA with vel = None, acc = None.
    """

    def __init__(self, maxlen: int, alpha: float, max_jump_px: Optional[float] = 120.0):
        self.maxlen = maxlen
        self.alpha = alpha
        self.max_jump_px = max_jump_px
        self._max_jump_sq = (max_jump_px * max_jump_px) if max_jump_px is not None else None
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

        if pt is None:
            # Point missing this frame
            self.prev_ema_pos = self.ema_pos
            self.prev_vel = self.vel
            self.vel = (0.0, 0.0) if self.prev_ema_pos is not None else None
            self.acc = (0.0, 0.0) if self.prev_vel is not None else None
            self._outlier_count = 0
            return

        # ── First-frame initialisation ─────────────────────────────────────
        if not self._initialized or self.ema_pos is None:
            self.ema_pos = pt
            self.prev_ema_pos = None
            self.vel = None
            self.acc = None
            self._initialized = True
            self._outlier_count = 0
            return

        # ── Jitter Gate ────────────────────────────────────────────────────
        if self._max_jump_sq is not None:
            dx = pt[0] - self.ema_pos[0]
            dy = pt[1] - self.ema_pos[1]
            if (dx * dx + dy * dy) > self._max_jump_sq:
                self._outlier_count += 1
                if self._outlier_count >= 2:
                    # 2 consecutive frames far away → Genuine relocation / re-acquisition
                    self.prev_ema_pos = None
                    self.ema_pos = pt
                    self.vel = None
                    self.acc = None
                    self._outlier_count = 0
                    return
                else:
                    # Isolated 1-frame jump → Outlier rejection
                    self.prev_ema_pos = self.ema_pos
                    self.prev_vel = self.vel
                    self.vel = (0.0, 0.0) if self.prev_ema_pos is not None else None
                    self.acc = (0.0, 0.0) if self.prev_vel is not None else None
                    return
            else:
                self._outlier_count = 0

        # ── Update EMA & Derivatives ───────────────────────────────────────
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
    Maintains temporal state for all fighters.
    Processes PoseFeatures frame-by-frame and outputs SmoothedFeatures.
    """
    def __init__(
        self,
        history_length: int = CFG.HISTORY_LENGTH,
        smoothing_factor: float = CFG.TEMPORAL_SMOOTHING_FACTOR,
    ):
        self.maxlen = history_length
        self.alpha = smoothing_factor
        self._states: Dict[int, _FighterTemporalState] = {}

        # Track distance between fighters (needs 2 fighters)
        self.fighter_distance: Optional[float] = None
        self.fighter_distance_history: collections.deque[Optional[float]] = collections.deque(maxlen=history_length)

    def update(self, features_list: list[PoseFeatures]) -> dict[int, SmoothedFeatures]:
        """
        Takes a list of PoseFeatures for the current frame, updates histories,
        and returns a dictionary mapping track_id -> SmoothedFeatures.
        """
        smoothed_dict = {}

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

        # Update per-fighter points
        for feats in features_list:
            tid = feats.track_id
            if tid not in self._states:
                self._states[tid] = _FighterTemporalState(self.maxlen, self.alpha)

            state = self._states[tid]

            if not feats.valid:
                # Decay all histories with None
                state.head.update(None)
                state.shoulder.update(None)
                state.body.update(None)
                state.l_wrist.update(None)
                state.r_wrist.update(None)
                state.l_elbow.update(None)
                state.r_elbow.update(None)
                state.l_ankle.update(None)
                state.r_ankle.update(None)
                state.head_rel.update(None)
                continue

            # Update individual joint histories
            state.head.update(feats.head_center)
            state.shoulder.update(feats.shoulder_center)
            state.body.update(feats.body_center)
            state.l_wrist.update(feats.left_wrist)
            state.r_wrist.update(feats.right_wrist)
            state.l_elbow.update(feats.left_elbow)
            state.r_elbow.update(feats.right_elbow)
            state.l_ankle.update(feats.left_ankle)
            state.r_ankle.update(feats.right_ankle)

            # ── Body-relative head offset ──────────────────────────────────
            if state.head.ema_pos is not None and state.shoulder.ema_pos is not None:
                rel_offset: Optional[Point] = (
                    state.head.ema_pos[0] - state.shoulder.ema_pos[0],
                    state.head.ema_pos[1] - state.shoulder.ema_pos[1],
                )
            else:
                rel_offset = None
            state.head_rel.update(rel_offset)

            # Package the smoothed results
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
