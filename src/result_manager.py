"""
src/result_manager.py — Result Management Layer

Responsibilities
----------------
- Create a timestamped run directory for each analysis (e.g., outputs/analysis_20260822_131500/).
- Provide paths for video generation.
- Safely export events, movement, and fight stats to CSV and JSON without overwriting previous runs.
- Collect metadata regarding the run (model, FPS, time taken).
- Handle missing data or partial failure gracefully.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from config import CFG
from src.strike_detector import StrikeEvent
from src.defense_detector import DefenseEvent
from src.movement_analyzer import MovementStats

class ResultManager:
    def __init__(self, input_filename: str):
        self.input_filename = input_filename
        
        # Create timestamped directory safely handling Windows filename conventions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = input_filename.replace(' ', '_').replace('.', '_')
        dir_name = f"analysis_{safe_name}_{timestamp}"
        
        self.output_dir = CFG.OUTPUT_DIR / dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self.metadata = {
            "input_filename": input_filename,
            "analysis_timestamp": timestamp,
            "model_name": getattr(CFG, 'MODEL_NAME', 'unknown'),
            "tracker": getattr(CFG, 'TRACKER', 'unknown'),
            "confidence_threshold": getattr(CFG, 'CONFIDENCE_THRESHOLD', 0.0),
            "video_resolution": None,
            "fps": None,
            "total_frames": 0,
            "duration_seconds": 0.0,
            "processing_time_seconds": 0.0
        }

    def set_video_metadata(self, width: int, height: int, fps: float, total_frames: int):
        self.metadata["video_resolution"] = f"{width}x{height}"
        self.metadata["fps"] = fps
        self.metadata["total_frames"] = total_frames
        self.metadata["duration_seconds"] = total_frames / max(fps, 1.0)
        
    def get_video_output_path(self) -> Path:
        """Returns the safe path to write the annotated video file."""
        return self.output_dir / "boxing_analysis.mp4"
        
    def export_results(self, 
                       fight_stats: Dict[str, Any], 
                       strikes: List[StrikeEvent], 
                       defenses: List[DefenseEvent],
                       final_movement: Dict[int, MovementStats]):
        """Safely exports all collected metrics to the timestamped directory."""
        self.metadata["processing_time_seconds"] = round(time.time() - self.start_time, 2)
        
        self._export_json(fight_stats, "fight_stats.json")
        self._export_metadata()
        self._export_fight_stats_csv(fight_stats)
        self._export_round_stats_csv(fight_stats)
        self._export_events_csv(strikes, defenses)
        self._export_movement_csv(final_movement)
        
    def _export_json(self, data: Dict[str, Any], filename: str):
        try:
            path = self.output_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to export {filename}: {e}")

    def _export_metadata(self):
        self._export_json(self.metadata, "summary.json")

    def _write_csv(self, filename: str, headers: List[str], rows: List[List[Any]]):
        if not headers and not rows:
            return # Skip entirely empty tables
            
        try:
            path = self.output_dir / filename
            # newline='' avoids blank lines in Windows Excel
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                writer.writerows(rows)
        except Exception as e:
            print(f"Failed to export CSV {filename}: {e}")

    def _export_fight_stats_csv(self, data: Dict[str, Any]):
        fighters = data.get("fighters", {})
        if not fighters:
            self._write_csv("fight_stats.csv", ["fighter_id", "status"], [[0, "No data"]])
            return
            
        headers = ["fighter_id"] + list(list(fighters.values())[0].keys())
        rows = []
        for fid, stats in fighters.items():
            row = [fid] + [stats.get(k) for k in headers[1:]]
            rows.append(row)
        self._write_csv("fight_stats.csv", headers, rows)

    def _export_round_stats_csv(self, data: Dict[str, Any]):
        headers = ["round", "fighter_id", "total_punches", "possible_landed", "blocks", "dodges"]
        rows = []
        for r_idx, r_data in data.get("rounds", {}).items():
            for fid, f_stats in r_data.items():
                rows.append([
                    r_idx, fid,
                    f_stats.get("total_punches", 0), 
                    f_stats.get("possible_landed", 0),
                    f_stats.get("blocks", 0), 
                    f_stats.get("dodges", 0)
                ])
        self._write_csv("round_stats.csv", headers, rows)

    def _export_events_csv(self, strikes: List[StrikeEvent], defenses: List[DefenseEvent]):
        headers = [
            "fighter_id", "frame_number", "event_type", "action", "hand", 
            "target_zone", "confidence", "x", "y"
        ]
        rows = []
        
        for s in strikes:
            rows.append([
                s.fighter_id, s.frame_number, s.event_type, s.action, s.hand,
                s.target_zone_estimate, round(s.confidence, 3),
                round(s.wrist_position[0], 2) if s.wrist_position else None,
                round(s.wrist_position[1], 2) if s.wrist_position else None
            ])
            
        for d in defenses:
            rows.append([
                d.fighter_id, d.frame_number, "DEFENSE", d.action, None,
                None, 1.0, None, None
            ])
            
        # Sort by frame number
        rows.sort(key=lambda r: r[1])
        self._write_csv("events.csv", headers, rows)

    def _export_movement_csv(self, movement_dict: Dict[int, MovementStats]):
        if not movement_dict:
            self._write_csv("movement.csv", ["fighter_id", "status"], [[0, "No data"]])
            return
            
        headers = [
            "fighter_id", "current_stance", "fighter_separation",
            "frames_advancing", "frames_retreating", "frames_stationary",
            "total_head_movement", "total_center_movement"
        ]
        rows = []
        for fid, stats in movement_dict.items():
            rows.append([
                fid, stats.current_stance, 
                round(stats.fighter_separation, 2) if stats.fighter_separation else None,
                stats.frames_advancing, stats.frames_retreating, stats.frames_stationary,
                round(stats.total_head_movement, 2), round(stats.total_center_movement, 2)
            ])
        self._write_csv("movement.csv", headers, rows)
