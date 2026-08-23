"""
src/defense_detector.py — Strike Outcome and Defensive Action Estimator (Frozen MVP Baseline).

Responsibilities
----------------
- Track detected strikes over a multi-frame temporal window (N frames).
- Classify punch outcomes using RobustScaleManager:
    * Timestamped frame-age expiration (30 frames / 1.2s at 25 FPS).
    * Anthropometric torso-ratio gating (rejects clinch merge artifacts).
    * Robust 80th-percentile scale reference over valid recent frames.
- Four disciplined outcome categories:
    * LANDED: Fist entered close target radius (<= 0.55 * SW_ref) with trajectory convergence.
    * BLOCKED: Fist intercepted by opponent's guard glove / defensive forearm (guard closer to target).
    * MISSED: Fist bypassed target, slipped, or was out-of-range (> 2.50 * SW_ref).
    * UNCERTAIN: Keypoints occluded / low-confidence during terminal phase, or ambiguous boundary zone.
- Detect defensive movements: DODGE (lateral head slip), BLOCK (high guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np

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
    had_target_approach: bool
    valid_at_min: bool
    ref_sw_at_min: float
    inst_sw_at_min: float


@dataclass
class _SWEntry:
    """Timestamped shoulder width measurement."""
    frame_idx: int
    sw_value: float


class RobustScaleManager:
    """
    Maintains a robust, time-decayed shoulder width reference.
    Rejects clinch-merge landmark spikes using anthropometric torso ratios
    and purges entries older than max_age_frames.
    """
    def __init__(self, max_age_frames: int = 30):
        self.max_age_frames = max_age_frames
        self._history: Dict[int, List[_SWEntry]] = {1: [], 2: []}
        self._established_baseline: Dict[int, float] = {1: 150.0, 2: 150.0}

    def update(self, tid: int, current_frame: int, feat: Optional[PoseFeatures]):
        if tid not in self._history:
            self._history[tid] = []

        # 1. Purge entries strictly older than max_age_frames (1.2s at 25 FPS)
        self._history[tid] = [
            e for e in self._history[tid]
            if (current_frame - e.frame_idx) <= self.max_age_frames
        ]

        # 2. Anthropometric validation (reject clinch merges & multi-body spans)
        if feat and feat.shoulder_width:
            sw = feat.shoulder_width
            th = distance(feat.shoulder_center, feat.hip_center) if (feat.shoulder_center and feat.hip_center) else None

            # Anatomical check: human bi-deltoid span is 0.35 to 1.15 x torso height
            is_valid_ratio = True
            if th and th > 30.0:
                ratio = sw / th
                if ratio < 0.35 or ratio > 1.15:
                    is_valid_ratio = False

            if is_valid_ratio and 20.0 < sw < 320.0:
                self._history[tid].append(_SWEntry(current_frame, sw))
                # Update long-term established fighter baseline (EMA)
                self._established_baseline[tid] = (
                    0.95 * self._established_baseline[tid] + 0.05 * sw
                )

    def get_reference_sw(self, tid: int, current_frame: int) -> float:
        # Purge expired entries
        if tid in self._history:
            self._history[tid] = [
                e for e in self._history[tid]
                if (current_frame - e.frame_idx) <= self.max_age_frames
            ]
            valid_values = [e.sw_value for e in self._history[tid]]
            if len(valid_values) >= 3:
                return float(np.percentile(valid_values, 80))

        return self._established_baseline.get(tid, 150.0)


class DefenseAndOutcomeDetector:

    def __init__(self):
        self.outcome_window = getattr(CFG, 'OUTCOME_WINDOW_FRAMES', 10)
        self.hit_dist_ratio = getattr(CFG, 'HIT_DISTANCE_RATIO', 0.55)
        self.guard_dist_ratio = getattr(CFG, 'GUARD_DISTANCE_RATIO', 0.60)
        self.ambiguous_outer_ratio = getattr(CFG, 'AMBIGUOUS_OUTER_RATIO', 0.90)
        self.reach_feint_ratio = getattr(CFG, 'REACH_FEINT_RATIO', 2.50)
        self.dodge_rel_threshold = getattr(CFG, 'DODGE_RELATIVE_VELOCITY_THRESHOLD', 0.035)
        self.dodge_min_frames = getattr(CFG, 'DODGE_MIN_CONSECUTIVE_FRAMES', 2)
        self.min_head_conf = getattr(CFG, 'MIN_HEAD_KP_CONF_FOR_DODGE', 0.40)

        self.scale_mgr = RobustScaleManager(max_age_frames=30)
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

        # Update scale reference manager for all present fighters
        for tid, feat in all_features.items():
            self.scale_mgr.update(tid, frame_idx, feat)

        # ── 1. Register new strikes ─────────────────────────────────────────
        for strike in new_strikes:
            self.active_strikes.append(_TrackedStrike(
                event=strike,
                frames_remaining=self.outcome_window,
                min_target_distance=float('inf'),
                initial_target_distance=None,
                was_blocked=False,
                had_target_approach=False,
                valid_at_min=False,
                ref_sw_at_min=150.0,
                inst_sw_at_min=150.0,
            ))

        # ── 2. Track active strikes and resolve outcomes ────────────────────
        remaining_strikes = []
        for ts in self.active_strikes:
            attacker_tid = ts.event.fighter_id
            opponent_tid = 2 if attacker_tid == 1 else (1 if attacker_tid == 2 else next((tid for tid in all_smoothed.keys() if tid != attacker_tid), None))

            if opponent_tid is not None and attacker_tid in all_smoothed and opponent_tid in all_smoothed:
                att_smooth = all_smoothed[attacker_tid]
                opp_smooth = all_smoothed[opponent_tid]
                opp_feat = all_features.get(opponent_tid)

                inst_sw = opp_feat.shoulder_width if (opp_feat and opp_feat.shoulder_width) else 150.0
                ref_sw = self.scale_mgr.get_reference_sw(opponent_tid, frame_idx)

                wrist = att_smooth.left_wrist if ts.event.hand == "left" else att_smooth.right_wrist
                target = opp_smooth.head_center if ts.event.target_zone_estimate == "HEAD" else opp_smooth.body_center

                if wrist and target:
                    dist = distance(wrist, target)
                    if dist is not None:
                        if ts.initial_target_distance is None:
                            ts.initial_target_distance = dist

                        if dist < ts.min_target_distance:
                            ts.min_target_distance = dist
                            ts.valid_at_min = True
                            ts.ref_sw_at_min = ref_sw
                            ts.inst_sw_at_min = inst_sw
                            if ts.initial_target_distance and dist < ts.initial_target_distance - 15.0:
                                ts.had_target_approach = True

                        hit_threshold = ref_sw * self.hit_dist_ratio
                        guard_threshold = ref_sw * self.guard_dist_ratio

                        # Defensive Glove Interception check (guard must be closer to target than attacking fist)
                        if dist <= hit_threshold * 1.3:
                            for opp_wrist in [opp_smooth.left_wrist, opp_smooth.right_wrist]:
                                if opp_wrist:
                                    w_dist = distance(opp_wrist, target)
                                    inter_dist = distance(wrist, opp_wrist)
                                    if (w_dist is not None and w_dist <= guard_threshold) and \
                                       (inter_dist is not None and inter_dist <= guard_threshold * 1.1):
                                        if w_dist < dist:  # Guard priority
                                            ts.was_blocked = True
                                            break

            ts.frames_remaining -= 1

            # Resolve outcome when observation window expires
            if ts.frames_remaining <= 0:
                ref_sw = ts.ref_sw_at_min
                hit_thresh = ref_sw * self.hit_dist_ratio
                outer_thresh = ref_sw * self.ambiguous_outer_ratio

                if opponent_tid is None or ts.min_target_distance == float('inf') or not ts.valid_at_min:
                    ts.event.event_type = "UNCERTAIN"
                    reason = "occluded/missing target"
                elif ts.initial_target_distance and ts.initial_target_distance > ref_sw * self.reach_feint_ratio and ts.min_target_distance > ref_sw * 1.2:
                    ts.event.event_type = "MISSED"
                    reason = f"out-of-range feint (init={ts.initial_target_distance:.1f}px > {ref_sw * self.reach_feint_ratio:.1f}px)"
                elif ts.was_blocked:
                    ts.event.event_type = "BLOCKED"
                    reason = f"guard interception (d_min={ts.min_target_distance:.1f}px, guard_thr={ref_sw * self.guard_dist_ratio:.1f}px)"
                elif ts.min_target_distance <= hit_thresh and ts.had_target_approach:
                    ts.event.event_type = "LANDED"
                    reason = f"clean contact (d_min={ts.min_target_distance:.1f}px <= hit_thr={hit_thresh:.1f}px, SW_ref={ref_sw:.1f}px)"
                elif ts.min_target_distance <= outer_thresh:
                    ts.event.event_type = "UNCERTAIN"
                    reason = f"ambiguous boundary zone ({hit_thresh:.1f}px < d_min={ts.min_target_distance:.1f}px <= {outer_thresh:.1f}px)"
                else:
                    ts.event.event_type = "MISSED"
                    reason = f"clear miss (d_min={ts.min_target_distance:.1f}px > {outer_thresh:.1f}px)"

                ts.event.supporting_features = reason
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

            ref_sw = self.scale_mgr.get_reference_sw(tid, frame_idx)

            # Dodge detection (body-relative head slip)
            if self._dodge_cooldowns.get(tid, 0) == 0:
                if self._track_ages.get(tid, 0) < 3:
                    self._prev_rel_lateral[tid] = 0.0
                else:
                    rel_vel = smooth.head_shoulder_velocity
                    if rel_vel is not None:
                        dx, dy = rel_vel
                        lateral_norm = abs(dx) / ref_sw if ref_sw > 0 else 0.0
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
                    guard_dist = ref_sw * CFG.GUARD_WRIST_HEAD_RATIO

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
