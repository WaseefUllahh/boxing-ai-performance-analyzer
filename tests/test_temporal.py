"""
test_temporal.py — Standalone debug runner for temporal feature smoothing.

Processes the first 100 frames, applies tracking and feature extraction, 
runs the TemporalFeatureManager, and prints debugging info.
"""

import sys
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CFG
from src.video_io import VideoReader
from src.tracker import FighterTracker
from src.pose_features import PoseFeatureExtractor, magnitude
from src.temporal_features import TemporalFeatureManager

def main():
    print("=" * 60)
    print("  Testing Temporal Feature Manager")
    print("=" * 60)
    
    video_path = CFG.VIDEO_PATH
    max_frames = 100
    
    # Initialize components
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
    
    with VideoReader(video_path) as reader:
        meta = reader.meta
        print(f"Reading video: {video_path}")
        print(f"Frames to process: {max_frames}\n")
        
        # Table Header
        print(f"{'Frame':<6} | {'Fighter':<7} | {'L Wrist Vel':<12} | {'R Wrist Vel':<12} | {'Head Vel':<10} | {'Distance':<10}")
        print("-" * 70)
        
        for frame_idx, frame in reader.frames(start_frame=0, end_frame=max_frames):
            
            # 1. Track
            tracked_fighters = tracker.update(frame)
            
            # 2. Extract base features
            feats_list = []
            for fighter in tracked_fighters:
                tid = fighter.get("track_id", fighter.get("bot_sort_id", -1))
                kps = fighter["keypoints"]
                feat = extractor.extract(
                    keypoints=kps,
                    track_id=tid,
                    frame_height=meta.height,
                    bbox_center=fighter.get("center")
                )
                feats_list.append(feat)
                
            # 3. Temporal smoothing
            smoothed_dict = temporal_mgr.update(feats_list)
            dist = temporal_mgr.get_fighter_distance()
            dist_str = f"{dist:.1f}" if dist is not None else "None"
            
            # 4. Debug output
            for tid, sf in smoothed_dict.items():
                l_vel = magnitude(sf.left_wrist_velocity)
                r_vel = magnitude(sf.right_wrist_velocity)
                h_vel = magnitude(sf.head_velocity)
                
                l_vel_str = f"{l_vel:.1f}" if l_vel is not None else "None"
                r_vel_str = f"{r_vel:.1f}" if r_vel is not None else "None"
                h_vel_str = f"{h_vel:.1f}" if h_vel is not None else "None"
                
                print(f"{frame_idx:<6} | {tid:<7} | {l_vel_str:<12} | {r_vel_str:<12} | {h_vel_str:<10} | {dist_str:<10}")

if __name__ == "__main__":
    main()
