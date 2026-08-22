"""
src/movement_analyzer.py — Calculates relative movement, advancing/retreating, and stance.

Responsibilities
----------------
- Estimate stance (Orthodox/Southpaw) via heuristic foot distance.
- Calculate relative movement vectors (advancing/retreating/lateral).
- Accumulate spatial movement volume (head, feet, center).
- Calculate simple activity scores.
"""

from __future__ import annotations


import collections
from dataclasses import dataclass
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
    current_movement_state: str = "STATIONARY" # "ADVANCING", "RETREATING", "LATERAL", "STATIONARY"
    
    # Distance
    fighter_separation: Optional[float] = None
    
    # Accumulators (screen-space units)
    total_head_movement: float = 0.0
    total_foot_movement: float = 0.0
    total_center_movement: float = 0.0
    
    # Frame counters
    frames_advancing: int = 0
    frames_retreating: int = 0
    frames_stationary: int = 0
    
    # Activity score heuristic (combines movement + actions)
    activity_score: float = 0.0

class MovementAnalyzer:
    def __init__(self):
        self.stance_window = getattr(CFG, 'STANCE_CONFIDENCE_FRAMES', 30)
        self.stance_foot_ratio = getattr(CFG, 'STANCE_FOOT_DIST_RATIO', 0.1)
        self.advance_vel = getattr(CFG, 'MOVEMENT_ADVANCE_VELOCITY', 4.0)
        self.retreat_vel = getattr(CFG, 'MOVEMENT_RETREAT_VELOCITY', -4.0)
        self.min_pixels = getattr(CFG, 'MOVEMENT_MIN_PIXELS', 3.0)
        
        self.stats: Dict[int, MovementStats] = {}
        
        # Rolling stance buffers (store 'ORTHODOX', 'SOUTHPAW', 'UNKNOWN')
        self._stance_buffers: Dict[int, collections.deque[str]] = {}

    def update(
        self,
        all_features: Dict[int, PoseFeatures],
        all_smoothed: Dict[int, 'SmoothedFeatures'],
        frame_idx: int
    ) -> Dict[int, MovementStats]:
        
        # Initialize missing state
        for tid in all_features.keys():
            if tid not in self.stats:
                self.stats[tid] = MovementStats(fighter_id=tid)
                self._stance_buffers[tid] = collections.deque(maxlen=self.stance_window)
                
        # Need exactly 2 fighters to calculate relative metrics properly
        tids = list(all_smoothed.keys())
        if len(tids) == 2:
            f1, f2 = tids[0], tids[1]
            sf1, sf2 = all_smoothed[f1], all_smoothed[f2]
            
            # Separation
            if sf1.body_center and sf2.body_center:
                sep = distance(sf1.body_center, sf2.body_center)
                self.stats[f1].fighter_separation = sep
                self.stats[f2].fighter_separation = sep
                
                # Analyze relative movement for f1
                self._analyze_relative_movement(f1, sf1, sf2)
                # Analyze relative movement for f2
                self._analyze_relative_movement(f2, sf2, sf1)
                
                # Analyze stance
                self._estimate_stance(f1, sf1, sf2, all_features[f1].shoulder_width)
                self._estimate_stance(f2, sf2, sf1, all_features[f2].shoulder_width)
        else:
            # Cannot do relative movement, fallback
            for tid in tids:
                self.stats[tid].current_movement_state = "STATIONARY"
                self.stats[tid].current_stance = "UNKNOWN"

        # Accumulate pure volume
        for tid, sf in all_smoothed.items():
            stat = self.stats[tid]
            
            # Head volume
            if sf.head_velocity:
                m = magnitude(sf.head_velocity)
                if m and m > self.min_pixels and m < 120.0: # Filter teleportation
                    stat.total_head_movement += m
            
            # Center volume
            if sf.body_velocity:
                m = magnitude(sf.body_velocity)
                if m and m > self.min_pixels and m < 120.0:
                    stat.total_center_movement += m
                    
            # Foot volume (average of left/right ankle velocity)
            # We don't have explicit ankle velocity in SmoothedFeatures, but since the center movement is the only metric we can reliably track in 2D space without jitter, we will just use center velocity for foot movement proxy.
            # Using 1.5x was an impossible statistic. We will use the center movement directly.
            stat.total_foot_movement = stat.total_center_movement 
            
        return self.stats

    def _analyze_relative_movement(self, tid: int, sf: 'SmoothedFeatures', opp_sf: 'SmoothedFeatures'):
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
            
        # Vector towards opponent
        dx = ox - bx
        dy = oy - by
        dist = magnitude((dx, dy))
        
        if dist and dist > 0:
            # Normalize direction vector
            nx = dx / dist
            ny = dy / dist
            
            # Dot product (velocity projected onto direction to opponent)
            projection = (vx * nx) + (vy * ny)
            
            if projection > self.advance_vel:
                stat.current_movement_state = "ADVANCING"
                stat.frames_advancing += 1
            elif projection < self.retreat_vel:
                stat.current_movement_state = "RETREATING"
                stat.frames_retreating += 1
            else:
                stat.current_movement_state = "LATERAL"
                # Lateral counts broadly towards stationary for "forward/back" metrics, but we track it.
                stat.frames_stationary += 1
        else:
            stat.current_movement_state = "STATIONARY"
            stat.frames_stationary += 1

    def _estimate_stance(self, tid: int, sf: 'SmoothedFeatures', opp_sf: 'SmoothedFeatures', shoulder_width: Optional[float]):
        """
        Estimate Orthodox vs Southpaw using foot proximity to opponent.
        Left foot closer = Orthodox. Right foot closer = Southpaw.
        """
        stat = self.stats[tid]
        buffer = self._stance_buffers[tid]
        
        if not sf.left_ankle or not sf.right_ankle or not opp_sf.body_center or not shoulder_width:
            buffer.append("UNKNOWN")
        else:
            d_left = distance(sf.left_ankle, opp_sf.body_center)
            d_right = distance(sf.right_ankle, opp_sf.body_center)
            
            if d_left is not None and d_right is not None:
                diff = d_right - d_left
                # Normalize difference by shoulder width
                ratio = diff / shoulder_width
                
                if ratio > self.stance_foot_ratio:
                    buffer.append("ORTHODOX")
                elif ratio < -self.stance_foot_ratio:
                    buffer.append("SOUTHPAW")
                else:
                    buffer.append("UNKNOWN")
            else:
                buffer.append("UNKNOWN")
                
        # Resolve consensus
        if len(buffer) > 0:
            counts = collections.Counter(buffer)
            most_common, count = counts.most_common(1)[0]
            
            # Only switch if strong consensus (e.g., > 50% of the window)
            if count > len(buffer) * 0.5:
                stat.current_stance = most_common
