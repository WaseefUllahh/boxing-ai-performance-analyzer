"""
src/video_processor.py — Orchestrates the full frame-by-frame pipeline.

Pipeline (per frame)
--------------------
1. VideoReader.frames()        → validated, streaming frame iterator
2. FighterTracker.update()     → tracked fighters + keypoints
3. PoseFeatureExtractor.extract()  → PoseFeatures per fighter
4. StrikeDetector.detect()     → StrikeResult per fighter
5. DefenseDetector.detect()    → DefenseResult per fighter
6. FightAnalyzer.update()      → accumulate stats
7. Annotate frame              → draw bboxes, skeletons, labels
8. Write frame to output video
9. (Optional) imshow preview

At end-of-video:
- FightAnalyzer.finalize()     → CSV + JSON
- Release all OpenCV resources
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from config import CFG
from src.video_io import VideoReader, print_video_info
from src.tracker import FighterTracker
from src.pose_features import PoseFeatureExtractor
from src.strike_detector import StrikeDetector
from src.defense_detector import DefenseDetector
from src.fight_analyzer import FightAnalyzer


# ---------------------------------------------------------------------------
# COCO skeleton pairs for drawing
# ---------------------------------------------------------------------------
SKELETON_PAIRS = [
    (5, 6),   # shoulders
    (5, 7),   # l shoulder → l elbow
    (7, 9),   # l elbow → l wrist
    (6, 8),   # r shoulder → r elbow
    (8, 10),  # r elbow → r wrist
    (5, 11),  # l shoulder → l hip
    (6, 12),  # r shoulder → r hip
    (11, 12), # hips
    (11, 13), # l hip → l knee
    (13, 15), # l knee → l ankle
    (12, 14), # r hip → r knee
    (14, 16), # r knee → r ankle
    (0, 5),   # nose → l shoulder
    (0, 6),   # nose → r shoulder
]


class VideoProcessor:
    """
    Orchestrates the end-to-end boxing analysis pipeline.

    Parameters
    ----------
    video_path : Path
        Input video file.
    model_name : str
        Ultralytics YOLO Pose model name.
    output_dir : Path
        Directory where annotated video and stats are saved.
    confidence : float
        YOLO detection confidence threshold.
    max_fighters : int
        Maximum fighters to track per frame.
    show_display : bool
        Whether to show a live preview window (disable for headless runs).
    """

    def __init__(
        self,
        video_path:   Path,
        model_name:   str  = CFG.MODEL_NAME,
        output_dir:   Path = CFG.OUTPUT_DIR,
        confidence:   float = CFG.CONFIDENCE_THRESHOLD,
        max_fighters: int   = CFG.MAX_FIGHTERS,
        show_display: bool  = True,
    ) -> None:
        self.video_path   = Path(video_path)
        self.model_name   = model_name
        self.output_dir   = Path(output_dir)
        self.confidence   = confidence
        self.max_fighters = max_fighters
        self.show_display = show_display

        # Sub-components (created fresh per run)
        self._tracker   = FighterTracker(
            model_name   = model_name,
            tracker_cfg  = CFG.TRACKER,
            confidence   = confidence,
            iou          = CFG.IOU_THRESHOLD,
            max_fighters = max_fighters,
        )
        self._feat_extractor = PoseFeatureExtractor()
        self._strike_det     = StrikeDetector()
        self._defense_det    = DefenseDetector()
        self._analyzer       = FightAnalyzer()

    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute the full pipeline on the configured video.

        Video I/O is fully delegated to VideoReader which:
        - validates file existence and codec compatibility
        - streams frames one-at-a-time (no full-video RAM load)
        - handles corrupted/dropped frames gracefully
        - guarantees VideoCapture.release() via context manager
        """
        # ── Open + validate video (raises VideoIOError on failure) ────────
        with VideoReader(self.video_path) as reader:
            meta = reader.meta

            # Print validated metadata
            print_video_info(self.video_path)

            out_fps = CFG.OUTPUT_VIDEO_FPS if CFG.OUTPUT_VIDEO_FPS > 0 else meta.fps

            # ── Output video writer ───────────────────────────────────────
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / f"annotated_{self.video_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*CFG.OUTPUT_VIDEO_CODEC)
            writer = cv2.VideoWriter(
                str(out_path), fourcc, out_fps, (meta.width, meta.height)
            )

            if not writer.isOpened():
                raise IOError(
                    f"Could not create output video writer at {out_path}\n"
                    f"Codec '{CFG.OUTPUT_VIDEO_CODEC}' may not be available on this system."
                )

            # ── Main loop — frame-by-frame streaming ──────────────────────
            try:
                with tqdm(
                    total=meta.frame_count or None,
                    desc="Processing",
                    unit="frame",
                    dynamic_ncols=True,
                ) as pbar:
                    for frame_idx, frame in reader.frames():
                        annotated = self._process_frame(
                            frame, frame_idx, meta.height
                        )
                        writer.write(annotated)

                        if self.show_display:
                            cv2.imshow(
                                "Boxing Analyzer — press Q to quit", annotated
                            )
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                print("\n[VideoProcessor] User quit early.")
                                break

                        pbar.update(1)

            finally:
                # VideoReader.__exit__ releases cap; release writer here
                writer.release()
                if self.show_display:
                    cv2.destroyAllWindows()

        print(f"[VideoProcessor] Annotated video → {out_path}")

        # ── Finalize stats ────────────────────────────────────────────────
        summary = self._analyzer.finalize()
        self._print_summary(summary)

    # ------------------------------------------------------------------
    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        frame_height: int,
    ) -> np.ndarray:
        """Run the pipeline on a single frame and return annotated copy."""
        annotated = frame.copy()

        # 1. Track
        tracked = self._tracker.update(frame)

        # 2. Features / Strike / Defense per fighter
        features  = {}
        strikes   = {}
        defenses  = {}

        for fighter in tracked:
            tid = fighter["track_id"]
            kps = fighter["keypoints"]
            feat = self._feat_extractor.extract(
                keypoints=kps,
                track_id=tid,
                frame_height=frame_height,
                bbox_center=fighter["center"]
            )
            strike  = self._strike_det.detect(feat)
            defense = self._defense_det.detect(feat)

            features[tid]  = feat
            strikes[tid]   = strike
            defenses[tid]  = defense

        # 3. Aggregate
        self._analyzer.update(frame_idx, tracked, strikes, defenses, features)

        # 4. Annotate
        annotated = self._annotate(annotated, tracked, strikes, defenses, features)

        return annotated

    # ------------------------------------------------------------------
    def _annotate(
        self,
        frame: np.ndarray,
        tracked: list[dict],
        strikes:  dict,
        defenses: dict,
        features: dict,
    ) -> np.ndarray:
        """Draw bounding boxes, skeletons, and action labels on the frame."""

        color_keys = list(CFG.COLORS.keys())

        for idx, fighter in enumerate(tracked):
            tid  = fighter["track_id"]
            bbox = fighter["bbox"]
            kps  = fighter["keypoints"]

            # Pick colour by slot index (0 or 1) for visual consistency
            color_key = f"fighter_{min(idx, 1)}"
            color = CFG.COLORS.get(color_key, (0, 255, 0))

            x1, y1, x2, y2 = bbox

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background + text
            strike  = strikes.get(tid)
            defense = defenses.get(tid)
            action  = strike.label  if (strike and strike.label != "NONE")  else ""
            daction = defense.label if (defense and defense.label != "NONE") else ""
            label   = f"F{tid}"
            if action:  label += f" | {action}"
            if daction: label += f" | {daction}"

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), CFG.COLORS["label_bg"], -1)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CFG.COLORS["label_text"], 1, cv2.LINE_AA)

            # Skeleton
            self._draw_skeleton(frame, kps, color)

        return frame

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_skeleton(
        frame: np.ndarray,
        keypoints: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        """Draw COCO keypoint skeleton on the frame in-place."""
        skel_color = CFG.COLORS["skeleton"]

        # Draw limb connections
        for (i, j) in SKELETON_PAIRS:
            if i >= len(keypoints) or j >= len(keypoints):
                continue
            xi, yi, ci = keypoints[i]
            xj, yj, cj = keypoints[j]
            if ci < CFG.KP_CONFIDENCE_THRESHOLD or cj < CFG.KP_CONFIDENCE_THRESHOLD:
                continue
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), skel_color, 1, cv2.LINE_AA)

        # Draw keypoint dots
        for kp in keypoints:
            x, y, c = kp
            if c < CFG.KP_CONFIDENCE_THRESHOLD:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    # ------------------------------------------------------------------
    @staticmethod
    def _print_summary(summary: dict) -> None:
        print("\n" + "=" * 60)
        print("  FIGHT SUMMARY")
        print("=" * 60)
        print(f"  Frames processed: {summary.get('total_frames_processed', 0)}")
        for fid, stats in summary.get("fighters", {}).items():
            print(f"\n  Fighter {fid} (track_id={stats['track_id']})")
            print(f"    Punches  : {stats['total_punches']}"
                  f"  (J:{stats['jabs']} C:{stats['crosses']}"
                  f" H:{stats['hooks']} U:{stats['uppercuts']})")
            print(f"    Defense  : Guards={stats['guards']}  "
                  f"Slips={stats['slips']}  Ducks={stats['ducks']}")
            print(f"    Distance : {stats['total_distance_px']:.0f} px")
            print(f"    Aggression: {stats['aggression_score']:.4f}")
        print("=" * 60)
