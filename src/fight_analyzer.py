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
import json
import csv
from collections import defaultdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import CFG
from src.strike_detector import StrikeEvent
from src.defense_detector import DefenseEvent
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
        strikes: List[StrikeEvent],
        defenses: List[DefenseEvent],
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
        for s in strikes: fighter_ids.add(s.fighter_id)
        for d in defenses: fighter_ids.add(d.fighter_id)
        
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
                "possible_landed": 0,
                "possible_blocked": 0,
                "possible_missed": 0,
                "estimated_punch_effectiveness": 0.0,
                # Defense
                "blocks": 0,
                "dodges": 0,
                "defensive_movements": 0,
                # Movement
                "normalized_head_movement": 0.0,
                "normalized_center_movement": 0.0,
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
        
        # 2. Process Strikes
        for s in strikes:
            fid = s.fighter_id
            stats = fighter_stats[fid]
            
            stats["total_punches"] += 1
            if s.action == "JAB": stats["jabs"] += 1
            elif s.action == "CROSS": stats["crosses"] += 1
            elif s.action == "HOOK": stats["hooks"] += 1
            elif s.action == "UPPERCUT": stats["uppercuts"] += 1
            
            if s.event_type == "POSSIBLE_LANDED": stats["possible_landed"] += 1
            elif s.event_type == "POSSIBLE_BLOCKED": stats["possible_blocked"] += 1
            elif s.event_type == "POSSIBLE_MISSED": stats["possible_missed"] += 1
            
            # Round bucketing
            r_idx = int((s.frame_number / fps) // self.assumed_round_duration) + 1
            round_stats[r_idx][fid]["total_punches"] += 1
            if s.event_type == "POSSIBLE_LANDED":
                round_stats[r_idx][fid]["possible_landed"] += 1

        # 3. Process Defenses
        for d in defenses:
            fid = d.fighter_id
            stats = fighter_stats[fid]
            
            stats["defensive_movements"] += 1
            if d.action == "BLOCK": stats["blocks"] += 1
            elif d.action == "DODGE": stats["dodges"] += 1
            
            r_idx = int((d.frame_number / fps) // self.assumed_round_duration) + 1
            if d.action == "BLOCK": round_stats[r_idx][fid]["blocks"] += 1
            if d.action == "DODGE": round_stats[r_idx][fid]["dodges"] += 1
            
        # 4. Process Movement & Calculate Derived Metrics
        for fid, stats in fighter_stats.items():
            # Derive attack & effectiveness
            stats["attack_frequency"] = _safe_div(stats["total_punches"], total_minutes)
            stats["estimated_punch_effectiveness"] = _safe_div(stats["possible_landed"], stats["total_punches"])
            
            mov = final_movement.get(fid)
            if mov:
                stats["stance"] = mov.current_stance
                stats["average_separation"] = mov.fighter_separation
                stats["normalized_head_movement"] = mov.total_head_movement / max(total_seconds, 1.0)
                stats["normalized_center_movement"] = mov.total_center_movement / max(total_seconds, 1.0)
                
                total_mov_frames = mov.frames_advancing + mov.frames_retreating + mov.frames_stationary
                stats["time_advancing_pct"] = _safe_div(mov.frames_advancing, total_mov_frames)
                stats["time_retreating_pct"] = _safe_div(mov.frames_retreating, total_mov_frames)
                stats["time_stationary_pct"] = _safe_div(mov.frames_stationary, total_mov_frames)
                
                # Simple heuristic activity score (0-100 scale approximation)
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
        

