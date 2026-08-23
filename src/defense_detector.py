"""
src/defense_detector.py — Strike Outcome and Defensive Action Estimator.

Responsibilities
----------------
- Track detected strikes over a multi-frame temporal window (N frames).
- Classify punch outcomes into four disciplined categories:
    * LANDED: Fist entered close target radius with trajectory convergence and impact proximity.
    * BLOCKED: Fist intercepted by opponent's guard glove / defensive forearm.
    * MISSED: Fist bypassed target or was out-of-range / slipped.
    * UNCERTAIN: Keypoints occluded / low-confidence during terminal phase, or ambiguous geometry.
- Detect defensive movements: DODGE (lateral head slip), BLOCK (high guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

from config import CFG
from src.pose_features import PoseFeatures, distance, magnitude
from src.temporal_features import SmoothedFeatures
from src.events import FightEvent

# ---------------------------------------------------------------------------
# Output Structures
# ---------------------------------------------------------------------------

@dataclass
class _TrackedStrike:
    """Internal state to track a strike over a temporal window."""
    event: FightEvent
    frames_remaining: int
    min_target_distance: float
    initial_target_distance: Optional[float]
    was_blocked: bool
    had_low_confidence: bool
    had_target_approach: bool


class DefenseAndOutcomeDetector:

    def __init__(self):
        self.outcome_window = getattr(CFG, 'OUTCOME_WINDOW_FRAMES', 10)
        self.hit_dist_ratio = getattr(CFG, 'HIT_DISTANCE_RATIO', 0.55)
        self.guard_dist_ratio = getattr(CFG, 'GUARD_DISTANCE_RATIO', 0.60)
        self.dodge_rel_threshold = getattr(CFG, 'DODGE_RELATIVE_VELOCITY_THRESHOLD', 0.035)
        self.dodge_min_frames = getattr(CFG, 'DODGE_MIN_CONSECUTIVE_FRAMES', 2)
        self.min_head_conf = getattr(CFG, 'MIN_HEAD_KP_CONF_FOR_DODGE', 0.40)

        self.active_strikes: List[_TrackedStrike] = []
        self._block_cooldowns: Dict[int, int] = {}
        self._dodge_cooldowns: Dict[int, int] = {}
        self._track_ages: Dict[int, int] = {}
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
        Updates tracking for existing strikes to determine outcomes (LANDED/BLOCKED/MISSED/UNCERTAIN),
        and detects new defense events.
        """
        resolved_events: List[FightEvent] = []
        timestamp = frame_idx / max(fps, 1.0)

        # Increment track ages
        for tid in all_smoothed.keys():
            self._track_ages[tid] = self._track_ages.get(tid, 0) + 1

        # Decrement cooldowns
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
                initial_target_distance=None,
                was_blocked=False,
                had_low_confidence=False,
                had_target_approach=False,
            ))

        # ── 2. Track active strikes and resolve outcomes ────────────────────
        remaining_strikes = []
        for ts in self.active_strikes:
            attacker_tid = ts.event.fighter_id
            opponent_tid = next((tid for tid in all_smoothed.keys() if tid != attacker_tid), None)

            if opponent_tid is None or opponent_tid not in all_smoothed or attacker_tid not in all_smoothed:
                ts.frames_remaining -= 1
                ts.had_low_confidence = True
            else:
                att_smooth = all_smoothed[attacker_tid]
                opp_smooth = all_smoothed[opponent_tid]
                opp_feat = all_features.get(opponent_tid)

                wrist = att_smooth.left_wrist if ts.event.hand == "left" else att_smooth.right_wrist
                target = opp_smooth.head_center if ts.event.target_zone_estimate == "HEAD" else opp_smooth.body_center
                shoulder_width = opp_feat.shoulder_width if opp_feat and opp_feat.shoulder_width else 50.0

                if wrist and target:
                    dist = distance(wrist, target)
                    if dist is not None:
                        if ts.initial_target_distance is None:
                            ts.initial_target_distance = dist

                        if dist < ts.min_target_distance:
                            ts.min_target_distance = dist
                            if ts.initial_target_distance and dist < ts.initial_target_distance - 15.0:
                                ts.had_target_approach = True

                        hit_threshold = shoulder_width * self.hit_dist_ratio
                        guard_threshold = shoulder_width * self.guard_dist_ratio

                        # Check if opponent glove intercepts
                        if dist <= hit_threshold * 1.3:
                            for opp_wrist in [opp_smooth.left_wrist, opp_smooth.right_wrist]:
                                if opp_wrist:
                                    w_dist = distance(opp_wrist, target)
                                    inter_dist = distance(wrist, opp_wrist)
                                    if (w_dist is not None and w_dist <= guard_threshold) and \
                                       (inter_dist is not None and inter_dist <= guard_threshold * 1.1):
                                        ts.was_blocked = True
                                        break
                else:
                    ts.had_low_confidence = True

                ts.frames_remaining -= 1

            # Resolve outcome when observation window expires
            if ts.frames_remaining <= 0:
                opp_feat = all_features.get(opponent_tid) if opponent_tid else None
                sw = opp_feat.shoulder_width if opp_feat and opp_feat.shoulder_width else 50.0
                hit_thresh = sw * self.hit_dist_ratio
                outer_thresh = sw * 0.90

                if opponent_tid is None or ts.min_target_distance == float('inf'):
                    ts.event.event_type = "UNCERTAIN"
                elif ts.initial_target_distance and ts.initial_target_distance > sw * 2.5 and ts.min_target_distance > sw * 1.2:
                    # Out-of-range feint or aborted jab
                    ts.event.event_type = "MISSED"
                elif ts.was_blocked:
                    ts.event.event_type = "BLOCKED"
                elif ts.min_target_distance <= hit_thresh and ts.had_target_approach:
                    if ts.had_low_confidence:
                        ts.event.event_type = "UNCERTAIN"
                    else:
                        ts.event.event_type = "LANDED"
                elif ts.min_target_distance <= outer_thresh:
                    # Transition / ambiguous boundary zone
                    ts.event.event_type = "UNCERTAIN" if ts.had_low_confidence else "MISSED"
                else:
                    ts.event.event_type = "MISSED"

                resolved_events.append(ts.event)
            else:
                remaining_strikes.append(ts)

        self.active_strikes = remaining_strikes

        # ── 3. Detect Defense Actions (DODGE / BLOCK) ───────────────────────
        for tid, feat in all_features.items():
            smooth = all_smoothed.get(tid)
            if not smooth or not feat.valid:
                self._prev_rel_lateral[tid] = 0.0
                continue

            sw = feat.shoulder_width or 50.0

            # Dodge detection (body-relative head slip)
            if self._dodge_cooldowns.get(tid, 0) == 0:
                if self._track_ages.get(tid, 0) < 3:
                    self._prev_rel_lateral[tid] = 0.0
                else:
                    rel_vel = smooth.head_shoulder_velocity
                    if rel_vel is not None:
                        dx, dy = rel_vel
                        lateral_norm = abs(dx) / sw if sw > 0 else 0.0
                        prev_lateral = self._prev_rel_lateral.get(tid, 0.0)
                        self._prev_rel_lateral[tid] = lateral_norm

                        if (lateral_norm > self.dodge_rel_threshold and
                            prev_lateral > self.dodge_rel_threshold and
                            abs(dx) > abs(dy)):
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
                        self._prev_rel_lateral[tid] = 0.0

            # Block detection (bilateral high guard)
            if self._block_cooldowns.get(tid, 0) == 0:
                head = smooth.head_center
                lw = smooth.left_wrist
                rw = smooth.right_wrist
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
