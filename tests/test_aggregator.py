"""
test_aggregator.py — Standalone debug runner for the Fight Analyzer.

Processes the first 300 frames of the video, runs the full analytical stack
(Tracking, Features, Strikes, Defenses, Movement), and aggregates the final 
results into the central FightAnalyzer.

Exports to fight_stats.json, fight_stats.csv, and round_stats.csv.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CFG
CFG.DEBUG_STRIKES = False

from src.video_io import VideoReader
from src.tracker import FighterTracker
from src.pose_features import PoseFeatureExtractor
from src.temporal_features import TemporalFeatureManager
from src.strike_detector import StrikeDetector
from src.defense_detector import DefenseAndOutcomeDetector
from src.movement_analyzer import MovementAnalyzer
from src.fight_analyzer import FightAnalyzer
from src.result_manager import ResultManager

def main():
    print("=" * 60)
    print("  Testing Fight Aggregator (Full Stack)")
    print("=" * 60)
    
    video_path = CFG.VIDEO_PATH
    max_frames = 300
    
    tracker = FighterTracker(
        model_name=CFG.MODEL_NAME,
        tracker_cfg=CFG.TRACKER,
        confidence=CFG.CONFIDENCE_THRESHOLD,
        iou=CFG.IOU_THRESHOLD,
        max_fighters=CFG.MAX_FIGHTERS,
        device=""
    )
    extractor = PoseFeatureExtractor()
    temporal_mgr = TemporalFeatureManager()
    strike_det = StrikeDetector()
    defense_det = DefenseAndOutcomeDetector()
    movement_mgr = MovementAnalyzer()
    aggregator = FightAnalyzer()
    
    all_events = []
    final_movement = {}
    
    with VideoReader(video_path) as reader:
        meta = reader.meta
        print(f"Reading video: {video_path}")
        print(f"Frames to process: {max_frames}\n")
        
        for frame_idx, frame in reader.frames(start_frame=0, end_frame=max_frames):
            
            tracked = tracker.update(frame)
            
            feats_dict = {}
            for fighter in tracked:
                tid = fighter.get("track_id", fighter.get("bot_sort_id", -1))
                kps = fighter["keypoints"]
                feat = extractor.extract(kps, tid, meta.height, fighter.get("center"))
                feats_dict[tid] = feat
                
            smoothed_dict = temporal_mgr.update(list(feats_dict.values()))
            
            # Strikes
            new_strikes = []
            for tid, feat in feats_dict.items():
                smoothed = smoothed_dict.get(tid)
                if not smoothed: continue
                opp_smoothed = next((sf for oid, sf in smoothed_dict.items() if oid != tid), None)
                
                events = strike_det.detect(feat, smoothed, opp_smoothed, frame_idx, meta.fps)
                new_strikes.extend(events)
                
            # Defense & Outcomes
            resolved_events = defense_det.update(
                new_strikes, feats_dict, smoothed_dict, frame_idx, meta.fps
            )
            all_events.extend(resolved_events)
            
            # Movement
            final_movement = movement_mgr.update(feats_dict, smoothed_dict, frame_idx)
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{max_frames} frames...")

    print("\nAggregating final results...")
    stats = aggregator.aggregate(
        events=all_events,
        final_movement=final_movement,
        fps=meta.fps,
        total_frames=max_frames
    )
    
    print("\nExporting files...")
    rm = ResultManager(video_path.name)
    rm.set_video_metadata(meta.width, meta.height, meta.fps, max_frames)
    rm.export_results(stats, all_events, final_movement)
    
    print(f"[DONE] Outputs saved to {rm.output_dir}")
    print("\nPreview of JSON output:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
