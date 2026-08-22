"""
test_tracker.py — Standalone test runner for tracking subsystem.

Processes the first 200 frames of the video, runs tracking (which includes
IdentityManager mapping), draws annotations, and saves the result to outputs/tracking_test.mp4.
"""

import cv2
import sys
from pathlib import Path

# Ensure we can import src modules
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CFG
from src.video_io import VideoReader
from src.tracker import FighterTracker
from src.video_processor import SKELETON_PAIRS

def main():
    print("=" * 60)
    print("  Testing Fighter Tracker (Identity Management)")
    print("=" * 60)
    
    # Configuration
    video_path = CFG.VIDEO_PATH
    output_path = CFG.OUTPUT_DIR / "tracking_test.mp4"
    max_frames = 200
    
    # Initialize Tracker
    tracker = FighterTracker(
        model_name=CFG.MODEL_NAME,
        tracker_cfg=CFG.TRACKER,
        confidence=CFG.CONFIDENCE_THRESHOLD,
        iou=CFG.IOU_THRESHOLD,
        max_fighters=CFG.MAX_FIGHTERS,
        device="" # auto
    )
    
    # Setup Video Writer
    with VideoReader(video_path) as reader:
        meta = reader.meta
        out_fps = CFG.OUTPUT_VIDEO_FPS if CFG.OUTPUT_VIDEO_FPS > 0 else meta.fps
        fourcc = cv2.VideoWriter_fourcc(*CFG.OUTPUT_VIDEO_CODEC)
        writer = cv2.VideoWriter(
            str(output_path), fourcc, out_fps, (meta.width, meta.height)
        )
        
        if not writer.isOpened():
            print(f"[ERROR] Could not create output video writer at {output_path}")
            return
            
        print(f"Reading video: {video_path}")
        print(f"Output video: {output_path}")
        print(f"Frames to process: {max_frames}")
        
        frames_processed = 0
        for frame_idx, frame in reader.frames(start_frame=0, end_frame=max_frames):
            annotated = frame.copy()
            
            # 1. Run inference + tracking + identity management
            tracked_fighters = tracker.update(frame)
            
            # 2. Draw annotations
            for fighter in tracked_fighters:
                app_id = fighter.get("track_id", 0)
                raw_id = fighter.get("bot_sort_id", -1)
                bbox = fighter["bbox"]
                kps = fighter["keypoints"]
                
                # Colors based on app_id (1=Fighter 1, 2=Fighter 2)
                color_key = f"fighter_{app_id % 2}"
                color = CFG.COLORS.get(color_key, (0, 255, 0))
                
                # Draw bounding box
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw labels (Fighter 1/2 and bot_sort track_id)
                label = f"Fighter {app_id} (Track: {raw_id})"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), CFG.COLORS["label_bg"], -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, CFG.COLORS["label_text"], 1, cv2.LINE_AA)
                
                # Draw skeleton
                skel_color = CFG.COLORS["skeleton"]
                for (idx_i, idx_j) in SKELETON_PAIRS:
                    if idx_i >= len(kps) or idx_j >= len(kps):
                        continue
                    xi, yi, ci = kps[idx_i]
                    xj, yj, cj = kps[idx_j]
                    if ci >= CFG.KP_CONFIDENCE_THRESHOLD and cj >= CFG.KP_CONFIDENCE_THRESHOLD:
                        cv2.line(annotated, (int(xi), int(yi)), (int(xj), int(yj)), skel_color, 1, cv2.LINE_AA)
                
            # 3. Write frame
            writer.write(annotated)
            frames_processed += 1
            if frames_processed % 20 == 0:
                print(f"Processed {frames_processed}/{max_frames} frames...")
                
        writer.release()
        
    print(f"\n[DONE] Test video saved to: {output_path}")

if __name__ == "__main__":
    main()
