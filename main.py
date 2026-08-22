"""
main.py — CLI entry-point for the Boxing AI Performance Analyzer.

Usage
-----
    python main.py                          # uses VIDEO_PATH from config.py
    python main.py --video data/fight.mp4   # explicit video path
    python main.py --video data/fight.mp4 --model yolov8m-pose --no-display

The module deliberately stays thin: it parses arguments, validates the input,
and delegates all heavy lifting to src.video_processor.VideoProcessor.
"""

import argparse
import sys
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    """Return a configured argument parser."""
    parser = argparse.ArgumentParser(
        prog="boxing-analyzer",
        description="Boxing AI Performance Analyzer — local video analysis pipeline.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Path to the input fight video.  Defaults to config.VIDEO_PATH.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "YOLO Pose model name (e.g. yolov8n-pose, yolov8m-pose).  "
            "Defaults to config.MODEL_NAME."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where outputs are saved.  Defaults to config.OUTPUT_DIR.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold (0-1).  Defaults to config value.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the live preview window (useful on headless servers).",
    )
    parser.add_argument(
        "--max-fighters",
        type=int,
        default=None,
        help="Maximum number of fighters to track (default: 2).",
    )
    return parser


def main() -> int:
    """Parse arguments, validate inputs, and run the pipeline."""
    # ── Imports here (not at module top) so config is importable even when
    #    heavy deps are missing — lets `python main.py --help` still work.
    from config import CFG

    parser = build_arg_parser()
    args = parser.parse_args()

    # ── Resolve paths / settings ──────────────────────────────────────────
    video_path: Path = args.video or CFG.VIDEO_PATH
    model_name: str = args.model or CFG.MODEL_NAME
    output_dir: Path = args.output_dir or CFG.OUTPUT_DIR
    conf: float = args.conf or CFG.CONFIDENCE_THRESHOLD
    max_fighters: int = args.max_fighters or CFG.MAX_FIGHTERS
    show_display: bool = not args.no_display

    # ── Validate input video ──────────────────────────────────────────────
    if not video_path.exists():
        print(
            f"[ERROR] Video file not found: {video_path}\n"
            "Place a video in the data/ folder or pass --video <path>."
        )
        return 1

    # ── Print run summary ─────────────────────────────────────────────────
    print("=" * 60)
    print("  Boxing AI Performance Analyzer")
    print("=" * 60)
    print(f"  Video       : {video_path}")
    print(f"  Model       : {model_name}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Confidence  : {conf}")
    print(f"  Max fighters: {max_fighters}")
    print(f"  Live display: {show_display}")
    print("=" * 60)

    # ── Import pipeline (heavy deps) ──────────────────────────────────────
    try:
        from src.video_processor import VideoProcessor
    except ImportError as exc:
        print(
            f"[ERROR] Could not import pipeline modules: {exc}\n"
            "Run `pip install -r requirements.txt` first."
        )
        return 1

    # ── Run pipeline ──────────────────────────────────────────────────────
    processor = VideoProcessor(
        video_path=video_path,
        model_name=model_name,
        output_dir=output_dir,
        confidence=conf,
        max_fighters=max_fighters,
        show_display=show_display,
    )
    processor.run()

    print("\n[DONE] Analysis complete.")
    print(f"  Annotated video : {output_dir}")
    print(f"  Stats CSV       : {CFG.STATS_CSV}")
    print(f"  Summary JSON    : {CFG.SUMMARY_JSON}")
    print(f"\nLaunch dashboard: streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
