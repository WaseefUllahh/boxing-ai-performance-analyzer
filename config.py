"""
config.py — Central configuration for Boxing AI Performance Analyzer.

All paths use pathlib.Path so they are OS-independent (works on Windows,
macOS, and Linux).  Import this module anywhere in the project:

    from config import CFG
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Project root (always the directory that contains this file)
# ─────────────────────────────────────────────────────────────────────────────
ROOT: Path = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Directory layout
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR: Path = ROOT / "data"
OUTPUT_DIR: Path = ROOT / "outputs"
SRC_DIR: Path = ROOT / "src"

# Ensure output directory exists at import time
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Default input video
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH: Path = DATA_DIR / "fight.mp4"

# ─────────────────────────────────────────────────────────────────────────────
# YOLO Pose model
# Ultralytics will auto-download the weights on first run if not found locally.
# Variants (speed ↔ accuracy): yolov8n-pose  yolov8s-pose  yolov8m-pose
#                               yolov8l-pose  yolov8x-pose
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME: str = "yolov8n-pose"   # lightweight default; swap for accuracy

# ─────────────────────────────────────────────────────────────────────────────
# Detection / tracking thresholds
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.40   # minimum detection confidence
IOU_THRESHOLD: float = 0.45          # NMS IoU threshold

# Tracker config file shipped with Ultralytics.
# "botsort.yaml" or "bytetrack.yaml" (both are included in the package).
TRACKER: str = "botsort.yaml"

# Maximum number of fighters expected in the scene.
# Detections beyond this count are ignored during analysis.
MAX_FIGHTERS: int = 2

# ─────────────────────────────────────────────────────────────────────────────
# Pose / keypoint indices  (COCO 17-keypoint convention)
# ─────────────────────────────────────────────────────────────────────────────
KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Minimum keypoint confidence to consider a keypoint valid
KP_CONFIDENCE_THRESHOLD: float = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# Strike detection thresholds  (heuristic rules, tunable)
# ─────────────────────────────────────────────────────────────────────────────
# Arm extension ratio: (wrist-to-shoulder dist) / (forearm length)
# A value ≥ this threshold is interpreted as "punching extension"
PUNCH_EXTENSION_THRESHOLD: float = 0.75

# Minimum wrist velocity (pixels/frame) to trigger punch detection
PUNCH_VELOCITY_THRESHOLD: float = 8.0

# Minimum number of consecutive frames a punch extension must be held
# before being counted (suppresses noise)
PUNCH_MIN_FRAMES: int = 2

# ─────────────────────────────────────────────────────────────────────────────
# Defense detection thresholds
# ─────────────────────────────────────────────────────────────────────────────
# Wrist-to-head distance (relative to shoulder width) below which
# the fighter is considered in a "guard" position
GUARD_WRIST_HEAD_RATIO: float = 0.60

# Hip drop (relative to standing hip height) that indicates a "duck"
DUCK_HIP_DROP_RATIO: float = 0.15

# Lateral shoulder displacement ratio for "slip" detection
SLIP_LATERAL_RATIO: float = 0.20

# ─────────────────────────────────────────────────────────────────────────────
# Movement analysis
# ─────────────────────────────────────────────────────────────────────────────
# Minimum pixel displacement of the bounding box centre per frame
# to count as "movement" (filters micro-jitter)
MOVEMENT_MIN_PIXELS: float = 3.0

# Smoothing window (frames) for velocity / position smoothing
SMOOTHING_WINDOW: int = 5

# ─────────────────────────────────────────────────────────────────────────────
# Video output
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_VIDEO_CODEC: str = "mp4v"   # fourcc; use "avc1" for H.264 on macOS
OUTPUT_VIDEO_FPS: float = 30.0     # override 0 to use source FPS

# Annotation colours  (BGR for OpenCV)
COLORS = {
    "fighter_0": (0, 255, 128),    # green-ish
    "fighter_1": (0, 128, 255),    # orange-ish
    "skeleton": (200, 200, 200),   # light grey
    "label_bg": (30, 30, 30),      # dark label background
    "label_text": (255, 255, 255), # white text
}

# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────
STATS_CSV: Path = OUTPUT_DIR / "fight_stats.csv"
SUMMARY_JSON: Path = OUTPUT_DIR / "fight_summary.json"

# ─────────────────────────────────────────────────────────────────────────────
# Convenience object — import as `from config import CFG`
# ─────────────────────────────────────────────────────────────────────────────
class _Config:
    """Namespace wrapper so callers can use CFG.MODEL_NAME etc."""
    ROOT = ROOT
    DATA_DIR = DATA_DIR
    OUTPUT_DIR = OUTPUT_DIR
    VIDEO_PATH = VIDEO_PATH
    MODEL_NAME = MODEL_NAME
    CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
    IOU_THRESHOLD = IOU_THRESHOLD
    TRACKER = TRACKER
    MAX_FIGHTERS = MAX_FIGHTERS
    KP = KP
    KP_CONFIDENCE_THRESHOLD = KP_CONFIDENCE_THRESHOLD
    PUNCH_EXTENSION_THRESHOLD = PUNCH_EXTENSION_THRESHOLD
    PUNCH_VELOCITY_THRESHOLD = PUNCH_VELOCITY_THRESHOLD
    PUNCH_MIN_FRAMES = PUNCH_MIN_FRAMES
    GUARD_WRIST_HEAD_RATIO = GUARD_WRIST_HEAD_RATIO
    DUCK_HIP_DROP_RATIO = DUCK_HIP_DROP_RATIO
    SLIP_LATERAL_RATIO = SLIP_LATERAL_RATIO
    MOVEMENT_MIN_PIXELS = MOVEMENT_MIN_PIXELS
    SMOOTHING_WINDOW = SMOOTHING_WINDOW
    OUTPUT_VIDEO_CODEC = OUTPUT_VIDEO_CODEC
    OUTPUT_VIDEO_FPS = OUTPUT_VIDEO_FPS
    COLORS = COLORS
    STATS_CSV = STATS_CSV
    SUMMARY_JSON = SUMMARY_JSON


CFG = _Config()
