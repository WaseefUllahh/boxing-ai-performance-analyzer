"""
src/video_processor.py — End-to-End Pipeline & Video Renderer

Responsibilities
----------------
- Consume video frames and route them through the full analytics stack.
- Render bounding boxes, full 17-keypoint skeletons, and movement trails.
- Render dynamic HUD (top banner, sidebars, popups, activity metrics).
- Safe temporary output handling: writes to `boxing_analysis.partial.mp4` and
  promotes to `boxing_analysis.mp4` only upon successful completion and verification.
- Guaranteed `VideoWriter` release via `try/finally`.
- Automatic output verification: validates file size, openability, resolution, FPS, and final frame read.
- H.264 remuxing / transcoding when FFmpeg is available, with safe OpenCV fallback.
- Periodic progress reporting with elapsed time, ETA, processing FPS, and memory usage.
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import subprocess
import collections
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from config import CFG
from src.video_io import VideoReader
from src.tracker import FighterTracker
from src.pose_features import PoseFeatureExtractor
from src.events import FightEvent
from src.temporal_features import TemporalFeatureManager
from src.strike_detector import StrikeDetector
from src.defense_detector import DefenseAndOutcomeDetector
from src.movement_analyzer import MovementAnalyzer
from src.fight_analyzer import FightAnalyzer
from src.result_manager import ResultManager

# ─────────────────────────────────────────────────────────────────────────────
# Skeleton Topology (COCO 17 Keypoints)
# ─────────────────────────────────────────────────────────────────────────────
SKELETON_PAIRS: List[Tuple[int, int]] = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Arms / Upper body
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16)
]


# ─────────────────────────────────────────────────────────────────────────────
# Encoding & Verification Utilities
# ─────────────────────────────────────────────────────────────────────────────

def detect_ffmpeg() -> Optional[str]:
    """Detect whether FFmpeg executable is available in PATH."""
    exe = shutil.which("ffmpeg")
    if exe:
        try:
            res = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                return exe
        except Exception:
            pass
    return None


def verify_video(
    path: Path,
    expected_width: int,
    expected_height: int,
    min_frames: int = 1
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Verifies that a generated video file exists, has valid dimensions,
    valid frame count, and can be read back through to the last frame.
    
    Returns:
        (is_valid, details_dict, error_message)
    """
    if not path.exists():
        return False, {}, f"File does not exist: {path}"
        
    file_size = path.stat().st_size
    if file_size < 10240:  # < 10 KB
        return False, {"file_size_bytes": file_size}, f"File size is too small ({file_size} bytes)"
        
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, {"file_size_bytes": file_size}, "OpenCV VideoCapture failed to open the file (corrupted or invalid container)"
        
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        duration_s = frame_count / max(fps, 1.0)
        
        details: Dict[str, Any] = {
            "file_path": str(path),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "width": w,
            "height": h,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration_seconds": round(duration_s, 2),
            "read_first_frame": False,
            "read_last_frame": False,
        }
        
        if frame_count < min_frames:
            return False, details, f"Frame count ({frame_count}) is less than minimum expected ({min_frames})"
            
        if w != expected_width or h != expected_height:
            return False, details, f"Resolution mismatch: expected {expected_width}x{expected_height}, got {w}x{h}"
            
        # Verify reading the first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            return False, details, "Failed to read first frame from video stream"
        details["read_first_frame"] = True
        
        # Verify reading the final frame
        if frame_count > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
            ret_last, frame_last = cap.read()
            if not ret_last or frame_last is None:
                return False, details, "Failed to read final frame from video stream (truncated file)"
            details["read_last_frame"] = True
        else:
            details["read_last_frame"] = True
            
        return True, details, None
    finally:
        cap.release()


def transcode_to_h264(input_path: Path, output_path: Path, ffmpeg_exe: str) -> Tuple[bool, Optional[str]]:
    """Transcodes a video file to browser-compatible H.264 / MP4 with faststart."""
    try:
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(output_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 10240:
            return True, None
        return False, f"FFmpeg error (code {res.returncode}): {res.stderr[:300]}"
    except Exception as e:
        return False, f"FFmpeg execution failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Video Processor
# ─────────────────────────────────────────────────────────────────────────────

class VideoProcessor:
    def __init__(self, result_manager: ResultManager):
        self.result_manager = result_manager
        self.final_output_path = self.result_manager.get_video_output_path()
        self.partial_output_path = self.result_manager.get_partial_video_path()
        
        self.writer: Optional[cv2.VideoWriter] = None
        self.f1_color = getattr(CFG, 'FIGHTER_1_COLOR', (255, 100, 100))  # Blue
        self.f2_color = getattr(CFG, 'FIGHTER_2_COLOR', (100, 100, 255))  # Red
        self.trail_len = getattr(CFG, 'TRAIL_LENGTH', 30)
        self.popup_frames = getattr(CFG, 'EVENT_POPUP_FRAMES', 45)
        
        self.fps = 30.0
        self.total_frames = 0
        self.ffmpeg_exe = detect_ffmpeg()
        
    def _init_writer(self, width: int, height: int, fps: float):
        """Initializes VideoWriter writing to the safe partial output file."""
        if self.writer is None:
            # Ensure parent dir exists
            self.partial_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            codec_str = getattr(CFG, 'OUTPUT_VIDEO_CODEC', 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            self.writer = cv2.VideoWriter(
                str(self.partial_output_path), fourcc, fps, (width, height)
            )
            self.fps = fps
            if not self.writer.isOpened():
                raise IOError(f"Could not open OpenCV VideoWriter at {self.partial_output_path}")

    def _get_color(self, fid: int) -> Tuple[int, int, int]:
        if fid == 1:
            return self.f1_color
        if fid == 2:
            return self.f2_color
        return (100, 255, 100)

    def _draw_hud(self, frame: np.ndarray, width: int, height: int, stats: Dict):
        """Draws transparent HUD banners and fighter statistics."""
        overlay = frame.copy()
        
        # Top banner
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        
        # Sidebars
        sidebar_w = 260
        cv2.rectangle(overlay, (0, 60), (sidebar_w, height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (width - sidebar_w, 60), (width, height), (0, 0, 0), -1)
        
        # Alpha blending
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Top title
        title = "BOXING AI PERFORMANCE ANALYZER"
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(frame, title, (max(10, (width - tw) // 2), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Left Sidebar (Fighter 1)
        f1_stats = stats.get("fighters", {}).get("1", stats.get("fighters", {}).get(1, {}))
        self._draw_sidebar_text(frame, 10, 95, "FIGHTER 1 (Blue)", self.f1_color, f1_stats)
        
        # Right Sidebar (Fighter 2)
        f2_stats = stats.get("fighters", {}).get("2", stats.get("fighters", {}).get(2, {}))
        self._draw_sidebar_text(frame, width - sidebar_w + 10, 95, "FIGHTER 2 (Red)", self.f2_color, f2_stats)

    def _draw_sidebar_text(self, frame: np.ndarray, x: int, y: int, title: str, color: Tuple, stats: Dict):
        cv2.putText(frame, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        y += 35
        
        total_p = stats.get('total_punches', 0)
        landed = stats.get('possible_landed', 0)
        blocked = stats.get('possible_blocked', 0)
        missed = stats.get('possible_missed', 0)
        
        lines = [
            f"Stance: {stats.get('stance', 'UNKNOWN')}",
            f"Punches: {total_p}",
            f"  Jabs: {stats.get('jabs', 0)} | Cross: {stats.get('crosses', 0)}",
            f"  Hooks: {stats.get('hooks', 0)} | Upper: {stats.get('uppercuts', 0)}",
            f"Landed (est): {landed}",
            f"Blocked (est): {blocked}",
            f"Missed (est): {missed}",
            f"Defensive Dodges: {stats.get('dodges', 0)}",
            f"Defensive Blocks: {stats.get('blocks', 0)}",
            f"Activity Score: {stats.get('activity_score', 0.0):.1f}"
        ]
        
        for line in lines:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            y += 24

    def _draw_fighters(self, frame: np.ndarray, tracked: List[Dict], smoothed_dict: Dict, trails: Dict[int, collections.deque]):
        """Draw bounding boxes, full COCO skeleton, and movement trails."""
        kp_conf_thresh = getattr(CFG, 'KP_CONFIDENCE_THRESHOLD', 0.30)
        
        for f in tracked:
            tid = f.get("track_id", f.get("bot_sort_id", -1))
            color = self._get_color(tid)
            
            # Bounding box
            box = f["bbox"]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"Fighter {tid}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), (20, 20, 20), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            
            # Full Skeleton lines
            kps = f["keypoints"]
            skel_color = (210, 210, 210)
            
            for (idx_i, idx_j) in SKELETON_PAIRS:
                if idx_i < len(kps) and idx_j < len(kps):
                    xi, yi, ci = kps[idx_i]
                    xj, yj, cj = kps[idx_j]
                    if ci >= kp_conf_thresh and cj >= kp_conf_thresh:
                        cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), skel_color, 2, cv2.LINE_AA)
            
            # Skeleton Joints
            for i, (kx, ky, kc) in enumerate(kps):
                if kc >= kp_conf_thresh:
                    joint_color = color if i in (9, 10) else (0, 255, 255) # Highlight wrists
                    cv2.circle(frame, (int(kx), int(ky)), 3, joint_color, -1, cv2.LINE_AA)
            
            # Movement Trails
            sf = smoothed_dict.get(tid)
            if sf and sf.body_center:
                trails[tid].append(sf.body_center)
                if len(trails[tid]) > 1:
                    pts = np.array([pt for pt in trails[tid]], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

    def _draw_popups(self, frame: np.ndarray, tracked: List[Dict], popups: Dict[int, List[Dict]]):
        """Render floating action event tags above fighters."""
        for f in tracked:
            tid = f.get("track_id", f.get("bot_sort_id", -1))
            box = f["bbox"]
            x1, y1, x2, y2 = map(int, box)
            
            active_popups = []
            y_offset = y1 - 35
            
            for p in popups[tid]:
                if p["frames_left"] > 0:
                    text = p["text"]
                    c = p["color"]
                    
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(frame, (x1, y_offset - th - 6), (x1 + tw + 6, y_offset + 4), (10, 10, 10), -1)
                    cv2.rectangle(frame, (x1, y_offset - th - 6), (x1 + tw + 6, y_offset + 4), c, 1)
                    cv2.putText(frame, text, (x1 + 3, y_offset - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2, cv2.LINE_AA)
                    
                    p["frames_left"] -= 1
                    y_offset -= (th + 14)
                    active_popups.append(p)
                    
            popups[tid] = active_popups

    def _draw_final_card(self, width: int, height: int, stats: Dict):
        """Render a clean summary card for 5 seconds at the end of the video."""
        card = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Header
        t1 = "BOXING AI PERFORMANCE ANALYZER"
        t2 = "FIGHT SUMMARY & PERFORMANCE METRICS"
        (w1, _), _ = cv2.getTextSize(t1, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        (w2, _), _ = cv2.getTextSize(t2, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.putText(card, t1, ((width - w1) // 2, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(card, t2, ((width - w2) // 2, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100, 200, 255), 2, cv2.LINE_AA)
        
        cv2.line(card, (100, 180), (width - 100, 180), (80, 80, 80), 2)
        
        # Fighter 1 (Left)
        f1 = stats.get("fighters", {}).get("1", stats.get("fighters", {}).get(1, {}))
        self._draw_card_fighter_block(card, width // 4 - 160, 240, "FIGHTER 1 (Blue)", self.f1_color, f1)
        
        # Fighter 2 (Right)
        f2 = stats.get("fighters", {}).get("2", stats.get("fighters", {}).get(2, {}))
        self._draw_card_fighter_block(card, 3 * width // 4 - 160, 240, "FIGHTER 2 (Red)", self.f2_color, f2)
        
        # Footer
        footer = "Heuristic-based CV analysis. Metrics represent physical estimates, not ground truth."
        (wf, _), _ = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(card, footer, ((width - wf) // 2, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Write for 5 seconds
        frames_to_write = int(self.fps * 5)
        for _ in range(frames_to_write):
            self.writer.write(card)

    def _draw_card_fighter_block(self, card: np.ndarray, x: int, y: int, title: str, color: Tuple, fstats: Dict):
        cv2.putText(card, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        y += 45
        
        metrics = [
            f"Stance: {fstats.get('stance', 'UNKNOWN')}",
            f"Total Punches: {fstats.get('total_punches', 0)}",
            f"  Jabs: {fstats.get('jabs', 0)} | Crosses: {fstats.get('crosses', 0)}",
            f"  Hooks: {fstats.get('hooks', 0)} | Uppercuts: {fstats.get('uppercuts', 0)}",
            f"Attack Rate: {fstats.get('attack_frequency', 0.0):.1f} punches/min",
            f"Possible Landed: {fstats.get('possible_landed', 0)}",
            f"Possible Blocked: {fstats.get('possible_blocked', 0)}",
            f"Possible Missed: {fstats.get('possible_missed', 0)}",
            f"Defensive Dodges: {fstats.get('dodges', 0)}",
            f"Defensive Blocks: {fstats.get('blocks', 0)}",
            f"Normalized Head Motion: {fstats.get('normalized_head_movement', 0.0):.3f} SW/s",
            f"Time Advancing: {fstats.get('time_advancing_pct', 0.0)*100:.1f}%",
            f"Time Retreating: {fstats.get('time_retreating_pct', 0.0)*100:.1f}%",
            f"Activity Score: {fstats.get('activity_score', 0.0):.1f} / 100",
        ]
        
        for m in metrics:
            cv2.putText(card, m, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
            y += 30

    def process_video(self, video_path: Path, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes the full video processing pipeline.
        
        Writes to `boxing_analysis.partial.mp4` first.
        Only upon successful completion and verification is it promoted to `boxing_analysis.mp4`.
        """
        tracker = FighterTracker(
            model_name=CFG.MODEL_NAME, tracker_cfg=CFG.TRACKER, 
            confidence=CFG.CONFIDENCE_THRESHOLD, iou=CFG.IOU_THRESHOLD, 
            max_fighters=CFG.MAX_FIGHTERS, device=""
        )
        extractor = PoseFeatureExtractor()
        temporal_mgr = TemporalFeatureManager()
        strike_det = StrikeDetector()
        defense_det = DefenseAndOutcomeDetector()
        movement_mgr = MovementAnalyzer()
        aggregator = FightAnalyzer()
        
        trails: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.trail_len)
        )
        popups: Dict[int, List[Dict]] = collections.defaultdict(list)
        all_events: List[FightEvent] = []
        final_movement: Dict[int, Any] = {}
        
        start_wall_time = time.time()
        process_mem = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        
        print("=" * 65)
        print("  Boxing AI Performance Analyzer — Video Processing Pipeline")
        print("=" * 65)
        print(f"  Input Video   : {video_path}")
        print(f"  Partial Target: {self.partial_output_path.name}")
        print(f"  Final Target  : {self.final_output_path.name}")
        print(f"  FFmpeg Path   : {self.ffmpeg_exe or 'Not Found (OpenCV mp4v fallback)'}")
        print("=" * 65)
        sys.stdout.flush()
        
        meta = None
        frames_rendered = 0
        processing_succeeded = False
        
        try:
            with VideoReader(video_path) as reader:
                meta = reader.meta
                self.result_manager.set_video_metadata(meta.width, meta.height, meta.fps, meta.frame_count)
                self._init_writer(meta.width, meta.height, meta.fps)
                self.total_frames = meta.frame_count if max_frames is None else min(meta.frame_count, max_frames)
                
                print(f"Processing {self.total_frames} frames ({meta.width}x{meta.height} @ {meta.fps:.2f} FPS)...\n")
                sys.stdout.flush()
                
                last_progress_print = time.time()
                
                for frame_idx, frame in reader.frames(start_frame=0, end_frame=max_frames):
                    # 1. Computer Vision & Feature Stack
                    tracked = tracker.update(frame)
                    
                    feats_dict = {}
                    for fighter in tracked:
                        tid = fighter.get("track_id", fighter.get("bot_sort_id", -1))
                        kps = fighter["keypoints"]
                        feat = extractor.extract(kps, tid, meta.height, fighter.get("center"))
                        feats_dict[tid] = feat
                        
                    smoothed_dict = temporal_mgr.update(list(feats_dict.values()))
                    
                    # 2. Action & Defense Detection
                    new_strikes = []
                    for tid, feat in feats_dict.items():
                        smoothed = smoothed_dict.get(tid)
                        if not smoothed:
                            continue
                        opp_smoothed = next((sf for oid, sf in smoothed_dict.items() if oid != tid), None)
                        events = strike_det.detect(feat, smoothed, opp_smoothed, frame_idx, meta.fps)
                        new_strikes.extend(events)
                        
                    resolved_events = defense_det.update(
                        new_strikes, feats_dict, smoothed_dict, frame_idx, meta.fps
                    )
                    all_events.extend(resolved_events)
                    
                    final_movement = movement_mgr.update(feats_dict, smoothed_dict, frame_idx)
                    
                    # 3. Add Live Popups
                    for e in resolved_events:
                        if e.category == "STRIKE":
                            c = (0, 255, 0) if e.event_type == "POSSIBLE_LANDED" else (0, 165, 255)
                            popups[e.fighter_id].insert(0, {
                                "text": f"{e.action}: {e.event_type}",
                                "frames_left": self.popup_frames,
                                "color": c
                            })
                        elif e.category == "DEFENSE":
                            popups[e.fighter_id].insert(0, {
                                "text": e.action,
                                "frames_left": self.popup_frames,
                                "color": (255, 255, 0)
                            })
                        
                    # 4. Live HUD Aggregation
                    live_stats = aggregator.aggregate(all_events, final_movement, meta.fps, frame_idx + 1)
                    
                    # 5. Render Annotations
                    self._draw_fighters(frame, tracked, smoothed_dict, trails)
                    self._draw_hud(frame, meta.width, meta.height, live_stats)
                    self._draw_popups(frame, tracked, popups)
                    
                    # 6. Write Frame
                    self.writer.write(frame)
                    frames_rendered += 1
                    
                    # 7. Progress Reporting
                    now = time.time()
                    if (frame_idx % 50 == 0) or (now - last_progress_print >= 2.5) or (frame_idx == self.total_frames - 1):
                        elapsed = now - start_wall_time
                        curr_fps = frames_rendered / max(elapsed, 0.001)
                        pct = (frames_rendered / max(self.total_frames, 1)) * 100.0
                        rem_sec = (self.total_frames - frames_rendered) / max(curr_fps, 0.001)
                        
                        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                        eta_str = time.strftime("%H:%M:%S", time.gmtime(rem_sec))
                        
                        mem_str = f" | RAM: {process_mem.memory_info().rss / (1024*1024):.1f} MB" if process_mem else ""
                        print(
                            f"Processing: Frame {frames_rendered:4d} / {self.total_frames} ({pct:5.1f}%) | "
                            f"Elapsed: {elapsed_str} | ETA: {eta_str} | Rate: {curr_fps:4.1f} FPS{mem_str}",
                            flush=True
                        )
                        last_progress_print = now

            # 8. Render Final Card (5 seconds)
            print("\nRendering 5-second fight summary card...", flush=True)
            final_stats = aggregator.aggregate(all_events, final_movement, meta.fps, self.total_frames)
            self._draw_final_card(meta.width, meta.height, final_stats)
            
            processing_succeeded = True

        except KeyboardInterrupt:
            print("\n[WARNING] Process interrupted by user (KeyboardInterrupt). Finalizing partial output...", flush=True)
            raise
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed with exception: {e}", flush=True)
            raise
        finally:
            # ── GUARANTEED VideoWriter release in finally block ────────────
            if self.writer is not None:
                self.writer.release()
                self.writer = None
                print("[INFO] VideoWriter released cleanly.", flush=True)

        # ── Output Verification & Promotion ─────────────────────────────────
        total_time_sec = round(time.time() - start_wall_time, 2)
        effective_fps = round(frames_rendered / max(total_time_sec, 0.001), 2)
        
        if not processing_succeeded:
            print(f"\n[INCOMPLETE] Processing was not completed. Partial video retained at: {self.partial_output_path}")
            return {}

        print("\nVerifying rendered output...", flush=True)
        is_valid, details, err = verify_video(
            self.partial_output_path, 
            expected_width=meta.width, 
            expected_height=meta.height, 
            min_frames=frames_rendered
        )
        
        if not is_valid:
            print(f"[ERROR] Render verification failed on partial video: {err}", flush=True)
            self.result_manager.set_video_validation({
                "valid": False, "error": err, "details": details
            })
            raise RuntimeError(f"Render validation failed: {err}")
            
        # ── H.264 Transcoding / Promotion ───────────────────────────────────
        if self.ffmpeg_exe:
            print(f"Transcoding to browser-compatible H.264 using FFmpeg...", flush=True)
            trans_ok, trans_err = transcode_to_h264(self.partial_output_path, self.final_output_path, self.ffmpeg_exe)
            if trans_ok:
                is_valid_h264, details_h264, err_h264 = verify_video(
                    self.final_output_path, meta.width, meta.height, min_frames=frames_rendered
                )
                if is_valid_h264:
                    print(f"H.264 encoding verified successfully ({details_h264['file_size_mb']} MB).", flush=True)
                    details = details_h264
                    details["codec"] = "H.264 / avc1 (FFmpeg)"
                    # Clean up partial video
                    try:
                        self.partial_output_path.unlink()
                    except OSError:
                        pass
                else:
                    print(f"[WARNING] Transcoded video failed verification ({err_h264}). Falling back to raw OpenCV output.", flush=True)
                    shutil.move(str(self.partial_output_path), str(self.final_output_path))
                    details["codec"] = "MPEG-4 / mp4v (OpenCV)"
            else:
                print(f"[WARNING] FFmpeg transcode failed: {trans_err}. Using raw OpenCV output.", flush=True)
                shutil.move(str(self.partial_output_path), str(self.final_output_path))
                details["codec"] = "MPEG-4 / mp4v (OpenCV)"
        else:
            print("[INFO] Promoting verified partial video to final destination...", flush=True)
            # Atomic replace/rename
            if self.final_output_path.exists():
                self.final_output_path.unlink()
            shutil.move(str(self.partial_output_path), str(self.final_output_path))
            details["codec"] = "MPEG-4 / mp4v (OpenCV)"
            print(
                "[NOTICE] FFmpeg not found in PATH. Video encoded with OpenCV MPEG-4 (mp4v).\n"
                "         Native desktop media players (VLC, MPV, Windows Media Player) support this file.\n"
                "         For seamless web browser / Streamlit playback, install FFmpeg.",
                flush=True
            )

        details["valid"] = True
        details["processing_time_seconds"] = total_time_sec
        details["effective_fps"] = effective_fps
        self.result_manager.set_video_validation(details)
        
        # ── Export Final Structured Data ────────────────────────────────────
        print("\nExporting fight analytics (JSON, CSV)...", flush=True)
        self.result_manager.export_results(final_stats, all_events, final_movement)
        
        print("=" * 65)
        print("  ANALYSIS & VIDEO RENDERING COMPLETE")
        print("=" * 65)
        print(f"  Result Directory : {self.result_manager.output_dir}")
        print(f"  Annotated Video  : {self.final_output_path} ({details.get('file_size_mb', 0)} MB)")
        print(f"  Frames Rendered  : {details.get('frame_count', 0)}")
        print(f"  Processing Time  : {total_time_sec}s ({effective_fps} FPS)")
        print("=" * 65)
        sys.stdout.flush()
        
        return final_stats
