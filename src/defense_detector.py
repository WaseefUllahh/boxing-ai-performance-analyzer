"""
src/defense_detector.py — Strike Outcome and Defensive Action Estimator.

Responsibilities
----------------
- Track detected strikes for N frames to find minimum distance to target.
- Classify strike outcome: POSSIBLE_LANDED, POSSIBLE_BLOCKED, POSSIBLE_MISSED.
- Detect defensive movements: DODGE, BLOCK.
"""

from __future__ import annotations

import math
import collections
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
        self.outcome_window = getattr(CFG, 'OUTCOME_WINDOW_FRAMES', 10)
        self.hit_dist_ratio = getattr(CFG, 'HIT_DISTANCE_RATIO', 0.8)
        self.guard_dist_ratio = getattr(CFG, 'GUARD_DISTANCE_RATIO', 0.6)
        self.dodge_velocity = getattr(CFG, 'DODGE_VELOCITY_THRESHOLD', 12.0)
        
        # Track pending strikes: list of _TrackedStrike
        self.active_strikes: List[_TrackedStrike] = []
        
        # Track block cooldown to avoid spamming "BLOCK" every frame
        self._block_cooldowns: Dict[int, int] = {}
        # Track dodge cooldown to avoid spamming "DODGE"
        self._dodge_cooldowns: Dict[int, int] = {}

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
            resolved_events: List of FightEvents that include resolved strikes and new defenses.
        """
        
        resolved_events: List[FightEvent] = []
        timestamp = frame_idx / max(fps, 1.0)
        
        # Decrement cooldowns
        for tid in list(self._block_cooldowns.keys()):
            if self._block_cooldowns[tid] > 0:
                self._block_cooldowns[tid] -= 1
        for tid in list(self._dodge_cooldowns.keys()):
            if self._dodge_cooldowns[tid] > 0:
                self._dodge_cooldowns[tid] -= 1
                
        # 1. Register new strikes
        for strike in new_strikes:
            self.active_strikes.append(_TrackedStrike(
                event=strike,
                frames_remaining=self.outcome_window,
                min_target_distance=float('inf'),
                was_blocked=False,
                resolved_outcome=None
            ))
            
        # 2. Track active strikes and resolve outcomes
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
                # Missing data, decay window
                ts.frames_remaining -= 1
            else:
                att_smooth = all_smoothed[attacker_tid]
                opp_smooth = all_smoothed[opponent_tid]
                opp_feat = all_features[opponent_tid]
                
                # Get attacking wrist
                wrist = att_smooth.left_wrist if ts.event.hand == "left" else att_smooth.right_wrist
                
                # Get target zone
                target = None
                if ts.event.target_zone_estimate == "HEAD":
                    target = opp_smooth.head_center
                elif ts.event.target_zone_estimate == "BODY":
                    target = opp_smooth.body_center
                    
                shoulder_width = opp_feat.shoulder_width or 50.0
                
                if wrist and target:
                    dist = distance(wrist, target)
                    if dist is not None:
                        # Update minimum distance
                        if dist < ts.min_target_distance:
                            ts.min_target_distance = dist
                            
                        # Check if blocked in this frame
                        hit_threshold = shoulder_width * self.hit_dist_ratio
                        guard_threshold = shoulder_width * self.guard_dist_ratio
                        
                        if dist <= hit_threshold:
                            # We are close enough to hit. Check if opponent guard is here.
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
                # Missing opponent or never got close
                if opponent_tid is None or ts.min_target_distance == float('inf'):
                    ts.event.event_type = "POSSIBLE_MISSED"
                else:
                    opp_feat = all_features.get(opponent_tid)
                    sw = opp_feat.shoulder_width if opp_feat and opp_feat.shoulder_width else 50.0
                    hit_thresh = sw * self.hit_dist_ratio
                    
                    if ts.min_target_distance <= hit_thresh:
                        if ts.was_blocked:
                            ts.event.event_type = "POSSIBLE_BLOCKED"
                        else:
                            ts.event.event_type = "POSSIBLE_LANDED"
                    else:
                        ts.event.event_type = "POSSIBLE_MISSED"
                        
                resolved_events.append(ts.event)
            else:
                remaining_strikes.append(ts)
                
        self.active_strikes = remaining_strikes
        
        # 3. Detect Defense Actions (Independent of strikes)
        for tid, feat in all_features.items():
            smooth = all_smoothed.get(tid)
            if not smooth or not feat.valid:
                continue
                
            sw = feat.shoulder_width or 50.0
            
            # Dodge detection (rapid lateral head movement)
            if self._dodge_cooldowns.get(tid, 0) == 0:
                if smooth.head_velocity:
                    dx, dy = smooth.head_velocity
                    if abs(dx) >= self.dodge_velocity and abs(dx) > abs(dy):
                        resolved_events.append(FightEvent(
                            fighter_id=tid,
                            frame_number=frame_idx,
                            timestamp=timestamp,
                            category="DEFENSE",
                            action="DODGE",
                            confidence=0.8,
                            supporting_features=f"head_dx: {dx:.1f}"
                        ))
                        self._dodge_cooldowns[tid] = CFG.STRIKE_COOLDOWN_FRAMES
            
            # Block/Guard detection (hands tight to head)
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
                                supporting_features=f"High guard"
                            ))
                            # Longer cooldown for static block
                            self._block_cooldowns[tid] = int(CFG.STRIKE_COOLDOWN_FRAMES * 1.5)
                            
        return resolved_events
