"""
test_movement.py — Standalone debug runner for the Movement & Stance Analyzer.

Processes the first 300 frames of the video, runs tracking, feature extraction,
temporal smoothing, and movement analytics.
Outputs to movement.csv for numerical sanity check and review.
"""

import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CFG
CFG.DEBUG_STRIKES = False

from src.video_io import VideoReader
from src.tracker import FighterTracker
from src.pose_features import PoseFeatureExtractor
from src.temporal_features import TemporalFeatureManager
from src.movement_analyzer import MovementAnalyzer
from src.result_manager import ResultManager

def main():
    print("=" * 60)
    print("  Testing Movement & Stance Analyzer")
    print("=" * 60)
    
    video_path = CFG.VIDEO_PATH
    max_frames = 300
    rm = ResultManager(video_path.name)
    
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
    movement_mgr = MovementAnalyzer()
    
    all_data = []
    
    with VideoReader(video_path) as reader:
        meta = reader.meta
        print(f"Reading video: {video_path}")
        print(f"Frames to process: {max_frames}")
        print(f"Writing CSV to: {rm.output_dir}\n")
        
        for frame_idx, frame in reader.frames(start_frame=0, end_frame=max_frames):
            
            tracked = tracker.update(frame)
            
            feats_dict = {}
            for fighter in tracked:
                tid = fighter.get("track_id", fighter.get("bot_sort_id", -1))
                kps = fighter["keypoints"]
                feat = extractor.extract(kps, tid, meta.height, fighter.get("center"))
                feats_dict[tid] = feat
                
            smoothed_dict = temporal_mgr.update(list(feats_dict.values()))
            
            # Movement Analytics
            stats = movement_mgr.update(
                all_features=feats_dict,
                all_smoothed=smoothed_dict,
                frame_idx=frame_idx
            )
            
            for tid, stat in stats.items():
                all_data.append({
                    "frame": frame_idx,
                    "fighter_id": tid,
                    "stance": stat.current_stance,
                    "state": stat.current_movement_state,
                    "separation": f"{stat.fighter_separation:.1f}" if stat.fighter_separation else "",
                    "head_mov": f"{stat.total_head_movement:.1f}",
                    "center_mov": f"{stat.total_center_movement:.1f}",
                    "adv_frames": stat.frames_advancing,
                    "ret_frames": stat.frames_retreating,
                    "stat_frames": stat.frames_stationary
                })
            
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx}/{max_frames} frames...")

    # Write to CSV
    out_csv = rm.output_dir / "movement.csv"
    with open(out_csv, 'w', newline='') as f:
        fieldnames = ["frame", "fighter_id", "stance", "state", "separation", 
                      "head_mov", "center_mov", "adv_frames", "ret_frames", "stat_frames"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
            
    print(f"\n[DONE] Wrote {len(all_data)} movement state records to {out_csv}")

if __name__ == "__main__":
    main()
