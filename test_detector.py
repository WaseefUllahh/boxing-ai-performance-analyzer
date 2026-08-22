"""
Test runner for YOLO Pose inference subsystem.
Processes the first 100 frames of the video, runs inference, draws annotations,
and saves the result to outputs/yolo_pose_test.mp4.
"""

import cv2
import sys
from pathlib import Path

# Ensure we can import src modules
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CFG
from src.video_io import VideoReader
from src.detector import PoseDetector
from src.video_processor import VideoProcessor

def main():
    print("=" * 60)
    print("  Testing YOLO Pose Detector (First 100 Frames)")
    print("=" * 60)
    
    # Configuration
    video_path = CFG.VIDEO_PATH
    output_path = CFG.OUTPUT_DIR / "yolo_pose_test.mp4"
    max_frames = 100
    
    # Initialize Detector
    detector = PoseDetector(
        model_name=CFG.MODEL_NAME,
        confidence=CFG.CONFIDENCE_THRESHOLD,
        iou=CFG.IOU_THRESHOLD,
        device="" # auto
    )
    
    # We will use the skeleton drawing logic from VideoProcessor for convenience
    # SKELETON_PAIRS are defined there
    from src.video_processor import SKELETON_PAIRS
    
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
            
            # 1. Run inference
            detections = detector.detect(frame)
            
            # 2. Draw annotations
            for i, det in enumerate(detections):
                bbox = det["bbox"]
                conf = det["confidence"]
                kps = det["keypoints"]
                
                color = CFG.COLORS.get(f"fighter_{i % 2}", (0, 255, 0))
                
                # Draw bounding box
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw confidence
                label = f"Person {conf:.2f}"
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
                
                # Draw keypoints
                for kp in kps:
                    x, y, c = kp
                    if c >= CFG.KP_CONFIDENCE_THRESHOLD:
                        cv2.circle(annotated, (int(x), int(y)), 3, color, -1)
                        
            # 3. Write frame
            writer.write(annotated)
            frames_processed += 1
            if frames_processed % 10 == 0:
                print(f"Processed {frames_processed}/{max_frames} frames...")
                
        writer.release()
        
    print(f"\n[DONE] Test video saved to: {output_path}")

if __name__ == "__main__":
    main()
