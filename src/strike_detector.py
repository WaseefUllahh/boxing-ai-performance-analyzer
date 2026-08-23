"""
src/strike_detector.py — Rule-based strike classifier.

Responsibilities
----------------
- Consume PoseFeatures and SmoothedFeatures.
- Detect strikes using configurable geometric heuristics.
- Apply cooldown debouncing to ensure 1 punch = 1 event.
- Differentiate Jab, Cross, Hook, Uppercut based on trajectory and posture.

Changes (audit fixes)
---------------------
- Arm extension: ext is Optional[float].  A None value (missing elbow) ALWAYS
  rejects the candidate — never substituted with a passing numeric value.
- Confidence formula: multi-signal bounded score replacing the old formula that
  saturated at 1.0 under normal movement.  Components: velocity score, extension
  score, elbow keypoint confidence, opponent proximity.  Practical range: 0.2–0.8.
- Debug logging: every candidate that meets velocity threshold is logged with the
  reason it was accepted or rejected (enabled by CFG.DEBUG_STRIKES).
"""

from __future__ import annotations

from typing import Optional, List

from config import CFG
from src.pose_features import PoseFeatures, distance, magnitude
from src.temporal_features import SmoothedFeatures
from src.events import FightEvent

# ---------------------------------------------------------------------------
# State Tracking
# ---------------------------------------------------------------------------

class _FighterStrikeState:
    """Maintains per-arm debounce cooldowns."""
    def __init__(self):
        self.left_cooldown: int = 0
        self.right_cooldown: int = 0

    def tick(self):
        if self.left_cooldown > 0:
            self.left_cooldown -= 1
        if self.right_cooldown > 0:
            self.right_cooldown -= 1

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class StrikeDetector:

    def __init__(self):
        self._states: dict[int, _FighterStrikeState] = {}
        self.min_velocity   = CFG.STRIKE_MIN_VELOCITY        # 42.0 px/frame (data-driven)
        self.max_velocity   = getattr(CFG, 'STRIKE_MAX_VELOCITY', 120.0)
        self.min_extension  = CFG.STRIKE_MIN_EXTENSION       # 0.65
        self.cooldown_frames = CFG.STRIKE_COOLDOWN_FRAMES    # 20 frames
        self.debug = CFG.DEBUG_STRIKES

    def detect(
        self,
        features: PoseFeatures,
        smoothed: SmoothedFeatures,
        opponent_smoothed: Optional[SmoothedFeatures],
        frame_idx: int,
        fps: float
    ) -> List[FightEvent]:

        tid = features.track_id
        if tid not in self._states:
            self._states[tid] = _FighterStrikeState()

        state = self._states[tid]
        state.tick()

        events: List[FightEvent] = []

        if not features.valid:
            return events

        timestamp = frame_idx / max(fps, 1.0)

        # Determine lead/rear hand: the hand physically closer to the opponent is lead
        left_is_lead = True
        if (opponent_smoothed and opponent_smoothed.body_center
                and smoothed.left_wrist and smoothed.right_wrist):
            dist_l = distance(smoothed.left_wrist, opponent_smoothed.body_center)
            dist_r = distance(smoothed.right_wrist, opponent_smoothed.body_center)
            if dist_l is not None and dist_r is not None:
                left_is_lead = dist_l < dist_r

        # ── Check Left Arm ──────────────────────────────────────────────────
        if state.left_cooldown == 0:
            vel = magnitude(smoothed.left_wrist_velocity) or 0.0
            ext = features.left_arm_extension          # Optional[float]
            elbow_conf = features.left_elbow_conf

            if self.min_velocity <= vel <= self.max_velocity:
                if ext is not None and ext >= self.min_extension:
                    event = self._classify_punch(
                        arm="left", is_lead=left_is_lead, vel=vel, ext=ext,
                        elbow_conf=elbow_conf, features=features,
                        smoothed=smoothed, opponent_smoothed=opponent_smoothed,
                        frame_idx=frame_idx, timestamp=timestamp,
                    )
                    if event:
                        events.append(event)
                        state.left_cooldown = self.cooldown_frames
                elif self.debug:
                    reason = "ext_none(no_elbow)" if ext is None else f"ext_low({ext:.2f})"
                    print(
                        f"[STRIKE_CAND] frame={frame_idx} tid={tid} arm=left "
                        f"vel={vel:.1f} ext={ext} elbow_conf={elbow_conf:.2f} "
                        f"REJECTED({reason})"
                    )

        # ── Check Right Arm ─────────────────────────────────────────────────
        if state.right_cooldown == 0:
            vel = magnitude(smoothed.right_wrist_velocity) or 0.0
            ext = features.right_arm_extension         # Optional[float]
            elbow_conf = features.right_elbow_conf

            if self.min_velocity <= vel <= self.max_velocity:
                if ext is not None and ext >= self.min_extension:
                    event = self._classify_punch(
                        arm="right", is_lead=not left_is_lead, vel=vel, ext=ext,
                        elbow_conf=elbow_conf, features=features,
                        smoothed=smoothed, opponent_smoothed=opponent_smoothed,
                        frame_idx=frame_idx, timestamp=timestamp,
                    )
                    if event:
                        events.append(event)
                        state.right_cooldown = self.cooldown_frames
                elif self.debug:
                    reason = "ext_none(no_elbow)" if ext is None else f"ext_low({ext:.2f})"
                    print(
                        f"[STRIKE_CAND] frame={frame_idx} tid={tid} arm=right "
                        f"vel={vel:.1f} ext={ext} elbow_conf={elbow_conf:.2f} "
                        f"REJECTED({reason})"
                    )

        return events

    def _classify_punch(
        self,
        arm: str,
        is_lead: bool,
        vel: float,
        ext: float,
        elbow_conf: float,
        features: PoseFeatures,
        smoothed: SmoothedFeatures,
        opponent_smoothed: Optional[SmoothedFeatures],
        frame_idx: int,
        timestamp: float,
    ) -> Optional[FightEvent]:

        # Wrist and velocity vector
        wrist = smoothed.left_wrist if arm == "left" else smoothed.right_wrist
        wrist_vel_vec = (
            smoothed.left_wrist_velocity if arm == "left"
            else smoothed.right_wrist_velocity
        )

        if not wrist or not wrist_vel_vec:
            return None

        dx, dy = wrist_vel_vec

        # Distance to opponent body center
        opp_dist = None
        if opponent_smoothed and opponent_smoothed.body_center:
            opp_dist = distance(wrist, opponent_smoothed.body_center)

        # Target zone estimation
        target = "UNKNOWN"
        if opponent_smoothed:
            opp_head = opponent_smoothed.head_center
            opp_body = opponent_smoothed.body_center
            if opp_head and opp_body:
                dist_to_head = distance(wrist, opp_head) or float('inf')
                dist_to_body = distance(wrist, opp_body) or float('inf')
                target = "HEAD" if dist_to_head < dist_to_body else "BODY"

        # ── Multi-signal confidence (Phase 2 fix) ───────────────────────────
        # Old formula: min(1.0, vel/50*0.5 + ext*0.5) → saturated at 1.0 frequently.
        # New formula: four independent signals, weighted, practical ceiling ~0.80.
        sw = features.shoulder_width or 50.0

        # Signal 1: velocity score — how far above minimum threshold (0–1)
        # Saturates at 3× min_velocity so typical fast punches score ~0.5
        vel_score = min(1.0, (vel - self.min_velocity) / (2.0 * self.min_velocity))

        # Signal 2: extension score — maps [0.55, 0.95] → [0, 1]
        ext_score = min(1.0, max(0.0, (ext - 0.55) / 0.40))

        # Signal 3: elbow keypoint confidence — maps [0, 0.80+] → [0, 1]
        kp_score = min(1.0, elbow_conf / 0.80)

        # Signal 4: opponent proximity — closer = higher, neutral when unknown
        if opp_dist is not None and sw > 0:
            prox_score = max(0.0, 1.0 - (opp_dist / (sw * 3.0)))
        else:
            prox_score = 0.5

        conf = 0.35 * vel_score + 0.30 * ext_score + 0.20 * kp_score + 0.15 * prox_score

        # ── Action classification ───────────────────────────────────────────
        action = "PUNCH"

        # Uppercut: sharply upward wrist movement dominant over lateral
        if dy < -0.5 * sw and abs(dy) > abs(dx):
            action = "UPPERCUT"
        # Hook: lateral dominant and arm not fully extended
        elif abs(dx) > abs(dy) * 1.5 and ext < 0.85:
            action = "HOOK"
        else:
            action = "JAB" if is_lead else "CROSS"

        # ── Debug output ────────────────────────────────────────────────────
        if self.debug:
            print(
                f"[STRIKE_FIRED] frame={frame_idx} tid={features.track_id} arm={arm} "
                f"action={action} vel={vel:.1f} ext={ext:.2f} elbow_conf={elbow_conf:.2f} "
                f"conf={conf:.3f} (vel_s={vel_score:.2f} ext_s={ext_score:.2f} "
                f"kp_s={kp_score:.2f} prox_s={prox_score:.2f})"
            )

        return FightEvent(
            fighter_id=features.track_id,
            frame_number=frame_idx,
            timestamp=timestamp,
            category="STRIKE",
            action=action,
            hand=arm,
            confidence=round(conf, 3),
            wrist_position=wrist,
            opponent_distance=round(opp_dist, 1) if opp_dist is not None else None,
            target_zone_estimate=target,
            event_type="STRIKE",
        )
