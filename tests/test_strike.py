"""
test_strike.py — Standalone debug runner for the Strike Detection Engine.

Processes the first 300 frames of the video, runs tracking, feature extraction,
temporal smoothing, and strike detection. Outputs to CSV for review.
"""

import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CFG
# Force debug on for this test script
CFG.DEBUG_STRIKES = True

from src.video_io import VideoReader
from src.tracker import FighterTracker
from src.pose_features import PoseFeatureExtractor
from src.temporal_features import TemporalFeatureManager
from src.strike_detector import StrikeDetector

def main():
    print("=" * 60)
    print("  Testing Strike Detector")
    print("=" * 60)
    
    video_path = CFG.VIDEO_PATH
    max_frames = 300
    out_csv = CFG.OUTPUT_DIR / "detected_strikes.csv"
    
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
    
    all_strikes = []
    
    with VideoReader(video_path) as reader:
        meta = reader.meta
        print(f"Reading video: {video_path}")
        print(f"Frames to process: {max_frames}")
        print(f"Writing CSV to: {out_csv}\n")
        
        for frame_idx, frame in reader.frames(start_frame=0, end_frame=max_frames):
            
            # 1. Track
            tracked = tracker.update(frame)
            
            # 2. Extract Base Features
            feats_dict = {}
            for fighter in tracked:
                tid = fighter.get("track_id", fighter.get("bot_sort_id", -1))
                kps = fighter["keypoints"]
                feat = extractor.extract(kps, tid, meta.height, fighter.get("center"))
                feats_dict[tid] = feat
                
            # 3. Temporal Smoothing
            smoothed_dict = temporal_mgr.update(list(feats_dict.values()))
            
            # 4. Strike Detection
            for tid, feat in feats_dict.items():
                smoothed = smoothed_dict.get(tid)
                if not smoothed:
                    continue
                    
                # Find opponent
                opponent_smoothed = None
                for other_tid, other_sf in smoothed_dict.items():
                    if other_tid != tid:
                        opponent_smoothed = other_sf
                        break
                        
                events = strike_det.detect(
                    features=feat,
                    smoothed=smoothed,
                    opponent_smoothed=opponent_smoothed,
                    frame_idx=frame_idx,
                    fps=meta.fps
                )
                
                all_strikes.extend(events)

    # Write to CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "timestamp", "fighter_id", "action", "hand", 
            "confidence", "target_zone", "opponent_distance", "wrist_pos_x", "wrist_pos_y"
        ])
        for s in all_strikes:
            wx, wy = s.wrist_position if s.wrist_position else ("", "")
            writer.writerow([
                s.frame_number, f"{s.timestamp:.2f}", s.fighter_id, s.action, s.hand,
                s.confidence, s.target_zone_estimate, s.opponent_distance, wx, wy
            ])
            
    print(f"\n[DONE] Wrote {len(all_strikes)} strikes to {out_csv}")

if __name__ == "__main__":
    main()
