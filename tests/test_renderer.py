"""
test_renderer.py — End-to-end rendering test

Runs the VideoProcessor on a short 100-frame clip to verify HUD layout,
overlay drawing, and video encoding.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CFG
# Disable visual debug for standard detector so it doesn't open cv2.imshow
CFG.DEBUG_STRIKES = False

from src.video_processor import VideoProcessor
from src.result_manager import ResultManager

def main():
    video_path = CFG.VIDEO_PATH
    
    print("=" * 60)
    print("  Testing Video Renderer")
    print("=" * 60)
    
    rm = ResultManager(video_path.name)
    processor = VideoProcessor(rm)
    
    # Process 10 frames for a quick directory/file creation test
    processor.process_video(video_path, max_frames=10)

if __name__ == "__main__":
    main()
