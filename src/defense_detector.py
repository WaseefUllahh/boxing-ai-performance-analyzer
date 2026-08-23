"""
src/defense_detector.py — Strike Outcome and Defensive Action Estimator.

Responsibilities
----------------
- Track detected strikes for N frames to find minimum distance to target.
- Classify strike outcome: POSSIBLE_LANDED, POSSIBLE_BLOCKED, POSSIBLE_MISSED.
- Detect defensive movements: DODGE, BLOCK.

Changes (audit fixes)
---------------------
- Dodge detection completely reworked:
    * Uses head_shoulder_velocity (body-relative head motion) instead of raw
      head_velocity, eliminating false positives from camera pan and body translation.
    * Velocity normalized by shoulder width → dimensionless, scale-invariant.
    * Requires the threshold to be met in TWO consecutive frames in the same lateral
      direction, blocking single-frame YOLO nose jitter spikes.
    * Requires head keypoint confidence >= MIN_HEAD_KP_CONF_FOR_DODGE.
    * Skips fighters whose track is younger than 3 frames (no reliable velocity yet),
      preventing the first-frame initialization artefact.
- Block detection: unchanged (still requires bilateral high guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

from config import CFG
from src.pose_features import PoseFeatures, distance
from src.temporal_features import SmoothedFeatures
from src.events import FightEvent

# ---------------------------------------------------------------------------
# Output Structures
# ---------------------------------------------------------------------------

@dataclass
class _TrackedStrike:
    """Internal state to track a strike over a time window."""
    event: FightEvent
    frames_remaining: int
    min_target_distance: float
    was_blocked: bool
    resolved_outcome: Optional[str]

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DefenseAndOutcomeDetector:

    def __init__(self):
        self.outcome_window   = getattr(CFG, 'OUTCOME_WINDOW_FRAMES', 10)
        self.hit_dist_ratio   = getattr(CFG, 'HIT_DISTANCE_RATIO', 0.8)
        self.guard_dist_ratio = getattr(CFG, 'GUARD_DISTANCE_RATIO', 0.6)

        # New dodge thresholds (body-relative, normalized by shoulder width)
        self.dodge_rel_threshold = getattr(CFG, 'DODGE_RELATIVE_VELOCITY_THRESHOLD', 0.15)
        self.dodge_min_frames    = getattr(CFG, 'DODGE_MIN_CONSECUTIVE_FRAMES', 2)
        self.min_head_conf       = getattr(CFG, 'MIN_HEAD_KP_CONF_FOR_DODGE', 0.40)

        # Track pending strikes
        self.active_strikes: List[_TrackedStrike] = []

        # Per-fighter cooldowns
        self._block_cooldowns: Dict[int, int] = {}
        self._dodge_cooldowns: Dict[int, int] = {}

        # New: track age (frames seen) to skip first 2 frames
        self._track_ages: Dict[int, int] = {}

        # New: previous frame's normalised lateral velocity for consecutive-frame check
        self._prev_rel_lateral: Dict[int, float] = {}

    def update(
        self,
        new_strikes: List[FightEvent],
        all_features: Dict[int, PoseFeatures],
        all_smoothed: Dict[int, SmoothedFeatures],
        frame_idx: int,
        fps: float
    ) -> List[FightEvent]:
        """
        Ingests new strikes, updates tracking for existing strikes to find outcomes,
        and detects new defense events.

        Returns:
            resolved_events: FightEvents that include resolved strikes and new defenses.
        """

        resolved_events: List[FightEvent] = []
        timestamp = frame_idx / max(fps, 1.0)

        # ── Increment track ages ────────────────────────────────────────────
        for tid in all_smoothed.keys():
            self._track_ages[tid] = self._track_ages.get(tid, 0) + 1

        # ── Decrement cooldowns ─────────────────────────────────────────────
        for tid in list(self._block_cooldowns.keys()):
            if self._block_cooldowns[tid] > 0:
                self._block_cooldowns[tid] -= 1
        for tid in list(self._dodge_cooldowns.keys()):
            if self._dodge_cooldowns[tid] > 0:
                self._dodge_cooldowns[tid] -= 1

        # ── 1. Register new strikes ─────────────────────────────────────────
        for strike in new_strikes:
            self.active_strikes.append(_TrackedStrike(
                event=strike,
                frames_remaining=self.outcome_window,
                min_target_distance=float('inf'),
                was_blocked=False,
                resolved_outcome=None,
            ))

        # ── 2. Track active strikes and resolve outcomes ────────────────────
        remaining_strikes = []
        for ts in self.active_strikes:
            attacker_tid = ts.event.fighter_id

            # Find opponent ID
            opponent_tid = None
            for tid in all_smoothed.keys():
                if tid != attacker_tid:
                    opponent_tid = tid
                    break

            if opponent_tid is None or opponent_tid not in all_smoothed or attacker_tid not in all_smoothed:
                ts.frames_remaining -= 1
            else:
                att_smooth = all_smoothed[attacker_tid]
                opp_smooth = all_smoothed[opponent_tid]
                opp_feat   = all_features[opponent_tid]

                wrist = att_smooth.left_wrist if ts.event.hand == "left" else att_smooth.right_wrist

                target = None
                if ts.event.target_zone_estimate == "HEAD":
                    target = opp_smooth.head_center
                elif ts.event.target_zone_estimate == "BODY":
                    target = opp_smooth.body_center

                shoulder_width = opp_feat.shoulder_width or 50.0

                if wrist and target:
                    dist = distance(wrist, target)
                    if dist is not None:
                        if dist < ts.min_target_distance:
                            ts.min_target_distance = dist

                        hit_threshold   = shoulder_width * self.hit_dist_ratio
                        guard_threshold = shoulder_width * self.guard_dist_ratio

                        if dist <= hit_threshold:
                            blocked = False
                            for opp_wrist in [opp_smooth.left_wrist, opp_smooth.right_wrist]:
                                if opp_wrist:
                                    w_dist = distance(opp_wrist, target)
                                    if w_dist is not None and w_dist <= guard_threshold:
                                        blocked = True
                                        break
                            if blocked:
                                ts.was_blocked = True

                ts.frames_remaining -= 1

            # Resolve if time is up
            if ts.frames_remaining <= 0:
                if opponent_tid is None or ts.min_target_distance == float('inf'):
                    ts.event.event_type = "POSSIBLE_MISSED"
                else:
                    opp_feat = all_features.get(opponent_tid)
                    sw = opp_feat.shoulder_width if opp_feat and opp_feat.shoulder_width else 50.0
                    hit_thresh = sw * self.hit_dist_ratio

                    if ts.min_target_distance <= hit_thresh:
                        ts.event.event_type = (
                            "POSSIBLE_BLOCKED" if ts.was_blocked else "POSSIBLE_LANDED"
                        )
                    else:
                        ts.event.event_type = "POSSIBLE_MISSED"

                resolved_events.append(ts.event)
            else:
                remaining_strikes.append(ts)

        self.active_strikes = remaining_strikes

        # ── 3. Detect Defense Actions ───────────────────────────────────────
        for tid, feat in all_features.items():
            smooth = all_smoothed.get(tid)
            if not smooth or not feat.valid:
                self._prev_rel_lateral[tid] = 0.0
                continue

            sw = feat.shoulder_width or 50.0

            # ── Dodge detection (body-relative head motion) ─────────────────
            if self._dodge_cooldowns.get(tid, 0) == 0:
                # Skip tracks that are too young — velocity is not yet reliable
                if self._track_ages.get(tid, 0) < 3:
                    self._prev_rel_lateral[tid] = 0.0
                else:
                    rel_vel = smooth.head_shoulder_velocity

                    if rel_vel is not None:
                        dx, dy = rel_vel
                        # Normalize lateral displacement by shoulder width
                        lateral_norm = abs(dx) / sw if sw > 0 else 0.0

                        prev_lateral = self._prev_rel_lateral.get(tid, 0.0)
                        self._prev_rel_lateral[tid] = lateral_norm

                        # Fire dodge only when:
                        # 1. Current frame exceeds threshold
                        # 2. Previous frame also exceeded threshold (consecutive)
                        # 3. Movement is laterally dominant (not mostly vertical)
                        if (lateral_norm > self.dodge_rel_threshold
                                and prev_lateral > self.dodge_rel_threshold
                                and abs(dx) > abs(dy)):
                            resolved_events.append(FightEvent(
                                fighter_id=tid,
                                frame_number=frame_idx,
                                timestamp=timestamp,
                                category="DEFENSE",
                                action="DODGE",
                                confidence=round(min(0.85, 0.45 + lateral_norm), 2),
                                supporting_features=f"rel_lateral_sw: {lateral_norm:.3f}",
                            ))
                            self._dodge_cooldowns[tid] = CFG.STRIKE_COOLDOWN_FRAMES
                    else:
                        # head_shoulder_velocity unavailable (first frame or jitter-gated)
                        self._prev_rel_lateral[tid] = 0.0

            # ── Block / Guard detection (both wrists near head) ─────────────
            # This logic is unchanged: requires a bilateral high guard.
            if self._block_cooldowns.get(tid, 0) == 0:
                head = smooth.head_center
                lw   = smooth.left_wrist
                rw   = smooth.right_wrist
                if head and lw and rw:
                    d_lw = distance(lw, head)
                    d_rw = distance(rw, head)
                    guard_dist = sw * CFG.GUARD_WRIST_HEAD_RATIO

                    if d_lw is not None and d_rw is not None:
                        if d_lw <= guard_dist and d_rw <= guard_dist:
                            resolved_events.append(FightEvent(
                                fighter_id=tid,
                                frame_number=frame_idx,
                                timestamp=timestamp,
                                category="DEFENSE",
                                action="BLOCK",
                                confidence=0.9,
                                supporting_features="High guard",
                            ))
                            self._block_cooldowns[tid] = int(CFG.STRIKE_COOLDOWN_FRAMES * 1.5)

        return resolved_events
