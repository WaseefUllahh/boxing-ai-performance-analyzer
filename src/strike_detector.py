"""
src/strike_detector.py — Rule-based strike classifier.

Responsibilities
----------------
- Consume PoseFeatures and SmoothedFeatures.
- Detect strikes using configurable geometric heuristics.
- Apply cooldown debouncing to ensure 1 punch = 1 event.
- Differentiate Jab, Cross, Hook, Uppercut based on trajectory and posture.
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
    """Maintains debounce cooldowns for each arm."""
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
        self.min_velocity = CFG.STRIKE_MIN_VELOCITY
        self.max_velocity = getattr(CFG, 'STRIKE_MAX_VELOCITY', 120.0)
        self.min_extension = CFG.STRIKE_MIN_EXTENSION
        self.cooldown_frames = CFG.STRIKE_COOLDOWN_FRAMES
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
        
        # Determine Lead/Rear hand using distance to opponent (if available)
        # In 2D, the hand physically closer to the opponent's body center is typically the lead hand (jab).
        left_is_lead = True 
        if opponent_smoothed and opponent_smoothed.body_center and smoothed.left_wrist and smoothed.right_wrist:
            dist_l = distance(smoothed.left_wrist, opponent_smoothed.body_center)
            dist_r = distance(smoothed.right_wrist, opponent_smoothed.body_center)
            if dist_l is not None and dist_r is not None:
                left_is_lead = dist_l < dist_r

        # Check Left Arm
        if state.left_cooldown == 0:
            vel = magnitude(smoothed.left_wrist_velocity) or 0.0
            ext = features.left_arm_extension
            if self.min_velocity <= vel <= self.max_velocity and ext >= self.min_extension:
                event = self._classify_punch(
                    arm="left",
                    is_lead=left_is_lead,
                    vel=vel,
                    ext=ext,
                    features=features,
                    smoothed=smoothed,
                    opponent_smoothed=opponent_smoothed,
                    frame_idx=frame_idx,
                    timestamp=timestamp
                )
                if event:
                    events.append(event)
                    state.left_cooldown = self.cooldown_frames

        # Check Right Arm
        if state.right_cooldown == 0:
            vel = magnitude(smoothed.right_wrist_velocity) or 0.0
            ext = features.right_arm_extension
            if self.min_velocity <= vel <= self.max_velocity and ext >= self.min_extension:
                event = self._classify_punch(
                    arm="right",
                    is_lead=not left_is_lead,
                    vel=vel,
                    ext=ext,
                    features=features,
                    smoothed=smoothed,
                    opponent_smoothed=opponent_smoothed,
                    frame_idx=frame_idx,
                    timestamp=timestamp
                )
                if event:
                    events.append(event)
                    state.right_cooldown = self.cooldown_frames
                    
        return events

    def _classify_punch(
        self,
        arm: str,
        is_lead: bool,
        vel: float,
        ext: float,
        features: PoseFeatures,
        smoothed: SmoothedFeatures,
        opponent_smoothed: Optional[SmoothedFeatures],
        frame_idx: int,
        timestamp: float
    ) -> Optional[FightEvent]:
        
        # Get hand specific vectors
        if arm == "left":
            wrist = smoothed.left_wrist
            wrist_vel_vec = smoothed.left_wrist_velocity
        else:
            wrist = smoothed.right_wrist
            wrist_vel_vec = smoothed.right_wrist_velocity
            
        if not wrist or not wrist_vel_vec:
            return None
            
        dx, dy = wrist_vel_vec
        
        # Distance to opponent
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

        # Base confidence calculation (heuristic based on velocity & extension)
        conf = min(1.0, (vel / (self.min_velocity * 2.0)) * 0.5 + ext * 0.5)

        # Action logic
        action = "PUNCH"
        
        sw = features.shoulder_width or 50.0
        
        # Uppercut: wrist moving sharply upwards (negative Y in image space)
        if dy < -0.5 * sw and abs(dy) > abs(dx):
            action = "UPPERCUT"
            
        # Hook: lateral movement dominant, arm not fully extended
        elif abs(dx) > abs(dy) * 1.5 and ext < 0.85:
            action = "HOOK"
            
        # Straight punches
        else:
            if is_lead:
                action = "JAB"
            else:
                action = "CROSS"

        # Debugging
        if self.debug:
            print(f"\n[DEBUG STRIKE] Frame: {frame_idx}")
            print(f"ACTION: {action}")
            print(f"fighter: {features.track_id}")
            print(f"hand: {arm} (lead={is_lead})")
            print(f"wrist velocity: {vel:.1f} (dx:{dx:.1f}, dy:{dy:.1f})")
            print(f"forward extension: {ext:.2f}")
            print(f"confidence: {conf:.2f}")

        return FightEvent(
            fighter_id=features.track_id,
            frame_number=frame_idx,
            timestamp=timestamp,
            category="STRIKE",
            action=action,
            hand=arm,
            confidence=round(conf, 2),
            wrist_position=wrist,
            opponent_distance=round(opp_dist, 1) if opp_dist else None,
            target_zone_estimate=target,
            event_type="STRIKE"
        )
