"""
src/video_processor.py — End-to-End Pipeline & Video Renderer

Responsibilities
----------------
- Consume video frames and route them through the full analytics stack.
- Render bounding boxes, skeletons, and movement trails.
- Render dynamic HUD (top banner, sidebars, popups, activity graph).
- Write annotated frames to an output video file seamlessly.
- Generate and append a final statistics card to the end of the video.
"""

from __future__ import annotations

import cv2
import numpy as np
import collections
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from config import CFG
from src.video_io import VideoReader
from src.tracker import FighterTracker
from src.pose_features import PoseFeatureExtractor
from src.temporal_features import TemporalFeatureManager
from src.strike_detector import StrikeDetector, StrikeEvent
from src.defense_detector import DefenseAndOutcomeDetector, DefenseEvent
from src.movement_analyzer import MovementAnalyzer
from src.fight_analyzer import FightAnalyzer

class VideoProcessor:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.writer = None
        self.f1_color = getattr(CFG, 'FIGHTER_1_COLOR', (255, 100, 100))
        self.f2_color = getattr(CFG, 'FIGHTER_2_COLOR', (100, 100, 255))
        self.trail_len = getattr(CFG, 'TRAIL_LENGTH', 30)
        self.popup_frames = getattr(CFG, 'EVENT_POPUP_FRAMES', 45)
        
        # State
        self.trails: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.trail_len)
        )
        # Popups: fighter_id -> list of {"text": str, "frames_left": int, "color": tuple}
        self.popups: Dict[int, List[Dict]] = collections.defaultdict(list)
        
        # We need cumulative data for the HUD and final card
        self.all_strikes: List[StrikeEvent] = []
        self.all_defenses: List[DefenseEvent] = []
        self.final_movement = {}
        
        self.fps = 30.0
        self.total_frames = 0
        self.aggregator = FightAnalyzer()
        
    def _init_writer(self, width: int, height: int, fps: float):
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
            self.fps = fps
            
    def _get_color(self, fid: int) -> Tuple[int, int, int]:
        # Fighter 1 is Blue, Fighter 2 is Red. Fallback is Green.
        if fid == 1: return self.f1_color
        if fid == 2: return self.f2_color
        return (100, 255, 100)

    def _draw_hud(self, frame: np.ndarray, width: int, height: int, stats: Dict):
        """Draws the transparent banners and HUD elements."""
        overlay = frame.copy()
        
        # Top banner
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        
        # Sidebars
        sidebar_w = 250
        cv2.rectangle(overlay, (0, 60), (sidebar_w, height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (width - sidebar_w, 60), (width, height), (0, 0, 0), -1)
        
        # Apply transparency
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Top Text
        title = "BOXING AI PERFORMANCE ANALYZER"
        cv2.putText(frame, title, (width//2 - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Left Sidebar (Fighter 1)
        f1_stats = stats.get("fighters", {}).get(1, {})
        self._draw_sidebar_text(frame, 10, 100, "FIGHTER 1", self.f1_color, f1_stats)
        
        # Right Sidebar (Fighter 2)
        f2_stats = stats.get("fighters", {}).get(2, {})
        self._draw_sidebar_text(frame, width - sidebar_w + 10, 100, "FIGHTER 2", self.f2_color, f2_stats)

    def _draw_sidebar_text(self, frame: np.ndarray, x: int, y: int, title: str, color: Tuple, stats: Dict):
        cv2.putText(frame, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 40
        
        lines = [
            f"Stance: {stats.get('stance', 'UNKNOWN')}",
            f"Punches: {stats.get('total_punches', 0)}",
            f"Landed (est): {stats.get('possible_landed', 0)}",
            f"Missed (est): {stats.get('possible_missed', 0)}",
            f"Blocks: {stats.get('blocks', 0)}",
            f"Dodges: {stats.get('dodges', 0)}",
            f"Activity: {stats.get('activity_score', 0.0):.1f}"
        ]
        
        for line in lines:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            y += 30

    def _draw_fighters(self, frame: np.ndarray, tracked: List[Dict], smoothed_dict: Dict):
        """Draw bounding boxes, skeletons, and trails."""
        for f in tracked:
            tid = f.get("track_id", f.get("bot_sort_id", -1))
            color = self._get_color(tid)
            
            # Bounding box
            box = f["bbox"]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Skeleton (simple lines if points exist)
            kps = f["keypoints"]
            # Just draw arms for MVP visualization
            if len(kps) > 10:
                l_s, l_e, l_w = kps[5][:2], kps[7][:2], kps[9][:2]
                r_s, r_e, r_w = kps[6][:2], kps[8][:2], kps[10][:2]
                for (p1, p2) in [(l_s, l_e), (l_e, l_w), (r_s, r_e), (r_e, r_w)]:
                    if p1[0] > 0 and p2[0] > 0:
                        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
            
            # Trails
            sf = smoothed_dict.get(tid)
            if sf and sf.body_center:
                self.trails[tid].append(sf.body_center)
                pts = np.array([pt for pt in self.trails[tid]], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, color, 2)

    def _draw_popups(self, frame: np.ndarray, tracked: List[Dict]):
        """Render floating text events above fighters."""
        for f in tracked:
            tid = f.get("track_id", f.get("bot_sort_id", -1))
            box = f["bbox"]
            x1, y1, x2, y2 = map(int, box)
            
            # Clean up old popups
            active_popups = []
            y_offset = y1 - 40
            
            for p in self.popups[tid]:
                if p["frames_left"] > 0:
                    text = p["text"]
                    c = p["color"]
                    
                    # Draw background rect for readability
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y_offset - th - 5), (x1 + tw, y_offset + 5), (0, 0, 0), -1)
                    cv2.putText(frame, text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
                    
                    p["frames_left"] -= 1
                    y_offset -= 35
                    active_popups.append(p)
                    
            self.popups[tid] = active_popups

    def _draw_final_card(self, width: int, height: int, stats: Dict):
        """Render a solid summary card for 5 seconds at the end."""
        card = np.zeros((height, width, 3), dtype=np.uint8)
        
        cv2.putText(card, "BOXING AI PERFORMANCE ANALYZER", (width//2 - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(card, "FIGHT SUMMARY (AI-ESTIMATED METRICS)", (width//2 - 300, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
        
        # Fighter 1
        f1 = stats.get("fighters", {}).get(1, {})
        self._draw_sidebar_text(card, width//4 - 150, 300, "FIGHTER 1 (Blue)", self.f1_color, f1)
        
        # Fighter 2
        f2 = stats.get("fighters", {}).get(2, {})
        self._draw_sidebar_text(card, 3*width//4 - 150, 300, "FIGHTER 2 (Red)", self.f2_color, f2)
        
        # Write for 5 seconds
        frames_to_write = int(self.fps * 5)
        for _ in range(frames_to_write):
            self.writer.write(card)

    def process_video(self, video_path: Path, max_frames: Optional[int] = None):
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
        
        print(f"Starting Video Processing: {video_path}")
        
        with VideoReader(video_path) as reader:
            meta = reader.meta
            self._init_writer(meta.width, meta.height, meta.fps)
            self.total_frames = meta.frame_count if max_frames is None else min(meta.frame_count, max_frames)
            
            for frame_idx, frame in reader.frames(start_frame=0, end_frame=max_frames):
                # 1. Models
                tracked = tracker.update(frame)
                
                feats_dict = {}
                for fighter in tracked:
                    tid = fighter.get("track_id", fighter.get("bot_sort_id", -1))
                    kps = fighter["keypoints"]
                    feat = extractor.extract(kps, tid, meta.height, fighter.get("center"))
                    feats_dict[tid] = feat
                    
                smoothed_dict = temporal_mgr.update(list(feats_dict.values()))
                
                new_strikes = []
                for tid, feat in feats_dict.items():
                    smoothed = smoothed_dict.get(tid)
                    if not smoothed: continue
                    opp_smoothed = next((sf for oid, sf in smoothed_dict.items() if oid != tid), None)
                    events = strike_det.detect(feat, smoothed, opp_smoothed, frame_idx, meta.fps)
                    new_strikes.extend(events)
                    
                resolved_strikes, defense_events = defense_det.update(
                    new_strikes, feats_dict, smoothed_dict, frame_idx, meta.fps
                )
                self.all_strikes.extend(resolved_strikes)
                self.all_defenses.extend(defense_events)
                
                self.final_movement = movement_mgr.update(feats_dict, smoothed_dict, frame_idx)
                
                # 2. Add Popups
                for s in resolved_strikes:
                    c = (0, 255, 0) if s.event_type == "POSSIBLE_LANDED" else (0, 165, 255)
                    self.popups[s.fighter_id].insert(0, {
                        "text": f"{s.action}: {s.event_type}",
                        "frames_left": self.popup_frames,
                        "color": c
                    })
                    
                for d in defense_events:
                    self.popups[d.fighter_id].insert(0, {
                        "text": d.action,
                        "frames_left": self.popup_frames,
                        "color": (255, 255, 0)
                    })
                    
                # 3. Live HUD Stats (Run aggregator up to this frame)
                live_stats = self.aggregator.aggregate(
                    self.all_strikes, self.all_defenses, self.final_movement, meta.fps, frame_idx+1
                )
                
                # 4. Rendering
                self._draw_fighters(frame, tracked, smoothed_dict)
                self._draw_hud(frame, meta.width, meta.height, live_stats)
                self._draw_popups(frame, tracked)
                
                self.writer.write(frame)
                
                if frame_idx % 50 == 0:
                    print(f"Processed {frame_idx}/{self.total_frames} frames...")

        print("\nRendering Final Stats Card...")
        final_stats = self.aggregator.aggregate(
            self.all_strikes, self.all_defenses, self.final_movement, meta.fps, self.total_frames
        )
        self._draw_final_card(meta.width, meta.height, final_stats)
        
        self.writer.release()
        print(f"[DONE] Video saved to {self.output_path}")
