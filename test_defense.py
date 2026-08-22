"""
test_defense.py — Standalone debug runner for the Defense & Outcome Detector.

Processes the first 300 frames of the video, runs tracking, feature extraction,
temporal smoothing, strike detection, and outcome/defense estimation.
Outputs to events.csv for review.
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
from src.strike_detector import StrikeDetector
from src.defense_detector import DefenseAndOutcomeDetector
from src.result_manager import ResultManager

def main():
    print("=" * 60)
    print("  Testing Defense & Outcome Detector")
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
    strike_det = StrikeDetector()
    defense_det = DefenseAndOutcomeDetector()
    
    all_strikes = []
    all_defenses = []
    
    with VideoReader(video_path) as reader:
        meta = reader.meta
        print(f"Reading video: {video_path}")
        print(f"Frames to process: {max_frames}")
        print(f"Writing results to: {rm.output_dir}\n")
        
        for frame_idx, frame in reader.frames(start_frame=0, end_frame=max_frames):
            
            tracked = tracker.update(frame)
            
            feats_dict = {}
            for fighter in tracked:
                tid = fighter.get("track_id", fighter.get("bot_sort_id", -1))
                kps = fighter["keypoints"]
                feat = extractor.extract(kps, tid, meta.height, fighter.get("center"))
                feats_dict[tid] = feat
                
            smoothed_dict = temporal_mgr.update(list(feats_dict.values()))
            
            # Strike Detection
            new_strikes = []
            for tid, feat in feats_dict.items():
                smoothed = smoothed_dict.get(tid)
                if not smoothed:
                    continue
                    
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
                new_strikes.extend(events)
                
            # Defense & Outcome Detection
            resolved_strikes, defense_events = defense_det.update(
                new_strikes=new_strikes,
                all_features=feats_dict,
                all_smoothed=smoothed_dict,
                frame_idx=frame_idx,
                fps=meta.fps
            )
            
            all_strikes.extend(resolved_strikes)
            all_defenses.extend(defense_events)
            
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx}/{max_frames} frames...")

    # Write using ResultManager
    rm.set_video_metadata(meta.width, meta.height, meta.fps, max_frames)
    rm._export_events_csv(all_strikes, all_defenses)
            
    print(f"\n[DONE] Wrote {len(all_strikes)} strikes and {len(all_defenses)} defense events to {rm.output_dir}")

if __name__ == "__main__":
    main()
