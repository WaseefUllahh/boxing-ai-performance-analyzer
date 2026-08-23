import sys, json
from pathlib import Path

sys.path.insert(0, '.')
from config import CFG
from src.video_io import VideoReader
from src.tracker import FighterTracker
from src.pose_features import PoseFeatureExtractor
from src.temporal_features import TemporalFeatureManager
from src.strike_detector import StrikeDetector
from src.defense_detector import DefenseAndOutcomeDetector

video_path = "data/fight2.mp4"
print(f"=== PROCESSING UNSEEN EXTERNAL VIDEO: {video_path} ===")

tracker = FighterTracker(model_name=CFG.MODEL_NAME, tracker_cfg=CFG.TRACKER, confidence=0.35, iou=0.45, max_fighters=2, device='')
extractor = PoseFeatureExtractor()
temporal_mgr = TemporalFeatureManager()
strike_det = StrikeDetector()
defense_det = DefenseAndOutcomeDetector()

events = []
frame_count = 0

with VideoReader(video_path) as reader:
    meta = reader.meta
    print(f"Video Info: {meta.width}x{meta.height} @ {meta.fps:.2f} FPS | Total Frames: {meta.frame_count}")
    
    for frame_idx, frame in reader.frames():
        frame_count += 1
        tracked = tracker.update(frame)
        feats_dict = {f.get('track_id'): extractor.extract(f['keypoints'], f.get('track_id'), meta.height, f.get('center')) for f in tracked}
        smoothed_dict = temporal_mgr.update(list(feats_dict.values()))
        
        new_strikes = []
        for tid, feat in feats_dict.items():
            smoothed = smoothed_dict.get(tid)
            if not smoothed: continue
            opp_smoothed = next((sf for oid, sf in smoothed_dict.items() if oid != tid), None)
            new_strikes.extend(strike_det.detect(feat, smoothed, opp_smoothed, frame_idx, meta.fps))
            
        resolved = defense_det.update(new_strikes, feats_dict, smoothed_dict, frame_idx, meta.fps)
        for r in resolved:
            if r.category == "STRIKE":
                events.append({
                    "frame": r.frame_number,
                    "timestamp": round(r.timestamp, 2),
                    "fighter_id": r.fighter_id,
                    "action": r.action,
                    "hand": r.hand,
                    "target_zone": r.target_zone_estimate,
                    "predicted_outcome": r.event_type,
                    "confidence": r.confidence,
                    "trigger_reason": r.supporting_features,
                })
        
        if frame_count % 500 == 0:
            print(f"  Processed {frame_count}/{meta.frame_count} frames ({frame_count/meta.frame_count*100:.1f}%) | Strikes so far: {len(events)}")

print(f"\nProcessing Complete! Total frames: {frame_count}, Detected Strike Events: {len(events)}")

with open("fight2_candidates.json", "w") as f:
    json.dump(events, f, indent=2)

print(f"Saved candidate events to fight2_candidates.json")
