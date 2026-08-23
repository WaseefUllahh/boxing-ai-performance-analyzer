"""
src/fight_analyzer.py — Central Aggregation Engine

Responsibilities
----------------
- Aggregate all frame-level (Movement) and event-level (Strikes, Defense) data.
- Calculate per-fighter and per-round statistics.
- Handle zero division and missing fighter edge cases safely.
- Sanitize JSON outputs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Dict, Any

from config import CFG
from src.events import FightEvent
from src.movement_analyzer import MovementStats

def _safe_div(a: float, b: float) -> float:
    return a / b if b and b > 0 else 0.0

def _sanitize_dict(obj: Any) -> Any:
    """Recursively replaces math.isnan and float('inf') with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_dict(x) for x in obj]
    return obj

class FightAnalyzer:
    def __init__(self):
        self.assumed_round_duration = getattr(CFG, 'ASSUMED_ROUND_DURATION', 180.0)

    def aggregate(
        self,
        events: List[FightEvent],
        final_movement: Dict[int, MovementStats],
        fps: float,
        total_frames: int
    ) -> Dict[str, Any]:
        """
        Aggregates all raw data into comprehensive statistics.
        Returns a dictionary suitable for JSON serialization and Streamlit.
        """
        total_seconds = total_frames / max(fps, 1.0)
        total_minutes = total_seconds / 60.0
        
        # Determine all unique fighter IDs across all sources
        fighter_ids = set(final_movement.keys())
        for e in events: fighter_ids.add(e.fighter_id)
        
        # If no fighters, return empty shell
        if not fighter_ids:
            return {"fighters": {}, "rounds": {}, "total_time_seconds": total_seconds}
            
        # 1. Initialize data structures
        fighter_stats = {}
        for fid in fighter_ids:
            fighter_stats[fid] = {
                # Attack
                "total_punches": 0,
                "jabs": 0,
                "crosses": 0,
                "hooks": 0,
                "uppercuts": 0,
                "attack_frequency": 0.0,
                # Outcomes
                "landed": 0,
                "blocked": 0,
                "missed": 0,
                "uncertain": 0,
                # Backwards-compatible legacy keys
                "possible_landed": 0,
                "possible_blocked": 0,
                "possible_missed": 0,
                "estimated_punch_effectiveness": 0.0,
                # Defense
                "blocks": 0,
                "dodges": 0,
                "defensive_movements": 0,
                # Movement
                # normalized_head_movement: body-relative head motion in shoulder-widths/s (dimensionless).
                # Much smaller than the old px/s values (typical range 0.1-2.0 vs old 400-700).
                "normalized_head_movement": 0.0,
                "normalized_center_movement": 0.0,
                # normalized_foot_movement: real ankle velocity / total_seconds (px/s).
                # None means no ankle data was available.
                "normalized_foot_movement": None,
                "time_advancing_pct": 0.0,
                "time_retreating_pct": 0.0,
                "time_stationary_pct": 0.0,
                "activity_score": 0.0,
                # Other
                "stance": "UNKNOWN",
                "average_separation": None,
                "active_time_seconds": total_seconds
            }
            
        round_stats = defaultdict(lambda: defaultdict(lambda: {
            "total_punches": 0,
            "possible_landed": 0,
            "blocks": 0,
            "dodges": 0
        }))
        
        # 2. Process Events
        for e in events:
            fid = e.fighter_id
            stats = fighter_stats[fid]
            r_idx = int((e.frame_number / fps) // self.assumed_round_duration) + 1
            
            if e.category == "STRIKE":
                stats["total_punches"] += 1
                if e.action == "JAB": stats["jabs"] += 1
                elif e.action == "CROSS": stats["crosses"] += 1
                elif e.action == "HOOK": stats["hooks"] += 1
                elif e.action == "UPPERCUT": stats["uppercuts"] += 1
                
                evt = e.event_type.upper()
                if "LANDED" in evt:
                    stats["landed"] += 1
                    stats["possible_landed"] += 1
                    round_stats[r_idx][fid]["possible_landed"] += 1
                elif "BLOCK" in evt:
                    stats["blocked"] += 1
                    stats["possible_blocked"] += 1
                elif "MISS" in evt:
                    stats["missed"] += 1
                    stats["possible_missed"] += 1
                elif "UNCERTAIN" in evt:
                    stats["uncertain"] += 1
                
                # Round bucketing
                round_stats[r_idx][fid]["total_punches"] += 1
                    
            elif e.category == "DEFENSE":
                stats["defensive_movements"] += 1
                if e.action == "BLOCK": stats["blocks"] += 1
                elif e.action == "DODGE": stats["dodges"] += 1
                
                if e.action == "BLOCK": round_stats[r_idx][fid]["blocks"] += 1
                if e.action == "DODGE": round_stats[r_idx][fid]["dodges"] += 1
            
        # 4. Process Movement & Calculate Derived Metrics
        for fid, stats in fighter_stats.items():
            # Derive attack & effectiveness
            stats["attack_frequency"] = _safe_div(stats["total_punches"], total_minutes)
            stats["estimated_punch_effectiveness"] = _safe_div(stats["possible_landed"], stats["total_punches"])
            
            mov = final_movement.get(fid)
            if mov:
                stats["stance"] = mov.current_stance
                stats["average_separation"] = mov.fighter_separation
                # normalized_head_movement is now in shoulder-widths/s (dimensionless)
                stats["normalized_head_movement"] = mov.total_head_movement / max(total_seconds, 1.0)
                stats["normalized_center_movement"] = mov.total_center_movement / max(total_seconds, 1.0)

                # normalized_foot_movement: real ankle-based; None if no ankle data
                if mov.ankle_frames_valid > 0:
                    stats["normalized_foot_movement"] = mov.total_foot_movement / max(total_seconds, 1.0)
                else:
                    stats["normalized_foot_movement"] = None  # no ankle data available

                total_mov_frames = mov.frames_advancing + mov.frames_retreating + mov.frames_stationary
                stats["time_advancing_pct"] = _safe_div(mov.frames_advancing, total_mov_frames)
                stats["time_retreating_pct"] = _safe_div(mov.frames_retreating, total_mov_frames)
                stats["time_stationary_pct"] = _safe_div(mov.frames_stationary, total_mov_frames)

                action_vol = (stats["total_punches"] * 2.0) + stats["defensive_movements"]
                mov_vol = (stats["time_advancing_pct"] + stats["time_retreating_pct"]) * 50.0
                stats["activity_score"] = min(100.0, action_vol + mov_vol)
                
        # 5. Sanitize and package
        output = {
            "total_time_seconds": total_seconds,
            "assumed_round_duration": self.assumed_round_duration,
            "fighters": fighter_stats,
            "rounds": dict(round_stats)
        }
        
        return _sanitize_dict(output)
        

