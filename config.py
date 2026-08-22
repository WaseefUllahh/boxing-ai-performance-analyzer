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

# Assumed round duration in seconds (used for temporal chunking)
ASSUMED_ROUND_DURATION: float = 180.0

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
# Minimum smoothed wrist velocity (pixels/frame) to trigger a strike
STRIKE_MIN_VELOCITY: float = 25.0

# Maximum smoothed wrist velocity. Anything higher is a tracker teleport artifact.
STRIKE_MAX_VELOCITY: float = 120.0

# Minimum arm extension ratio (wrist-to-shoulder / forearm) to count as punch
STRIKE_MIN_EXTENSION: float = 0.65

# Cooldown frames before the same arm can trigger another strike
STRIKE_COOLDOWN_FRAMES: int = 20

# Enable verbose console debugging for why strikes were triggered
DEBUG_STRIKES: bool = False

# ─────────────────────────────────────────────────────────────────────────────
# Defense & Outcome detection thresholds (heuristic rules, tunable)
# ─────────────────────────────────────────────────────────────────────────────
# Number of frames to track a punch trajectory to estimate outcome
OUTCOME_WINDOW_FRAMES: int = 10

# Max distance (normalized by shoulder width) to consider a strike landed
HIT_DISTANCE_RATIO: float = 0.8

# Max distance (normalized by shoulder width) for hands to be blocking a target
GUARD_DISTANCE_RATIO: float = 0.6

# Minimum head velocity to count as a dodging movement
DODGE_VELOCITY_THRESHOLD: float = 12.0

# Wrist-to-head distance (relative to shoulder width) below which
# the fighter is considered in a "guard" position
GUARD_WRIST_HEAD_RATIO: float = 0.60

# Hip drop (relative to standing hip height) that indicates a "duck"
DUCK_HIP_DROP_RATIO: float = 0.15

# Lateral shoulder displacement ratio for "slip" detection
SLIP_LATERAL_RATIO: float = 0.20

# ─────────────────────────────────────────────────────────────────────────────
# Movement analysis & Stance
# ─────────────────────────────────────────────────────────────────────────────
# Minimum pixel displacement per frame to count as "movement"
MOVEMENT_MIN_PIXELS: float = 3.0

# Minimum velocity (projected towards opponent) to be considered "advancing"
MOVEMENT_ADVANCE_VELOCITY: float = 4.0

# Minimum velocity (projected away from opponent) to be considered "retreating"
MOVEMENT_RETREAT_VELOCITY: float = -4.0

# Window size (frames) for rolling stance consensus
STANCE_CONFIDENCE_FRAMES: int = 30

# Minimum difference in distance-to-opponent (normalized) to classify a stance
STANCE_FOOT_DIST_RATIO: float = 0.1

# Maximum number of frames to store in temporal histories
HISTORY_LENGTH: int = 15

# ─────────────────────────────────────────────────────────────────────────────
# Video Annotation & HUD Rendering
# ─────────────────────────────────────────────────────────────────────────────
# Colors in BGR format for OpenCV
FIGHTER_1_COLOR: tuple[int, int, int] = (255, 100, 100) # Blue
FIGHTER_2_COLOR: tuple[int, int, int] = (100, 100, 255) # Red

# Length of movement trails
TRAIL_LENGTH: int = 30

# How many frames a strike/defense popup stays on screen
EVENT_POPUP_FRAMES: int = 45

# Alpha value for Exponential Moving Average (EMA) smoothing [0, 1]
# Lower means more smoothing, higher means more responsive
TEMPORAL_SMOOTHING_FACTOR: float = 0.3

# Frames to wait before triggering the same discrete action again
ACTION_COOLDOWN: int = 15

# Smoothing window (frames) for basic velocity / position smoothing 
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
    ASSUMED_ROUND_DURATION = getattr(CFG, 'ASSUMED_ROUND_DURATION', 180.0) if 'CFG' in globals() else 180.0
    KP = KP
    KP_CONFIDENCE_THRESHOLD = KP_CONFIDENCE_THRESHOLD
    STRIKE_MIN_VELOCITY = STRIKE_MIN_VELOCITY
    STRIKE_MAX_VELOCITY = STRIKE_MAX_VELOCITY
    STRIKE_MIN_EXTENSION = STRIKE_MIN_EXTENSION
    STRIKE_COOLDOWN_FRAMES = STRIKE_COOLDOWN_FRAMES
    DEBUG_STRIKES = DEBUG_STRIKES
    OUTCOME_WINDOW_FRAMES = OUTCOME_WINDOW_FRAMES
    HIT_DISTANCE_RATIO = HIT_DISTANCE_RATIO
    GUARD_DISTANCE_RATIO = GUARD_DISTANCE_RATIO
    DODGE_VELOCITY_THRESHOLD = DODGE_VELOCITY_THRESHOLD
    GUARD_WRIST_HEAD_RATIO = GUARD_WRIST_HEAD_RATIO
    DUCK_HIP_DROP_RATIO = DUCK_HIP_DROP_RATIO
    SLIP_LATERAL_RATIO = SLIP_LATERAL_RATIO
    MOVEMENT_MIN_PIXELS = MOVEMENT_MIN_PIXELS
    MOVEMENT_ADVANCE_VELOCITY = getattr(CFG, 'MOVEMENT_ADVANCE_VELOCITY', 4.0) if 'CFG' in globals() else 4.0
    MOVEMENT_RETREAT_VELOCITY = getattr(CFG, 'MOVEMENT_RETREAT_VELOCITY', -4.0) if 'CFG' in globals() else -4.0
    STANCE_CONFIDENCE_FRAMES = getattr(CFG, 'STANCE_CONFIDENCE_FRAMES', 30) if 'CFG' in globals() else 30
    STANCE_FOOT_DIST_RATIO = getattr(CFG, 'STANCE_FOOT_DIST_RATIO', 0.1) if 'CFG' in globals() else 0.1
    HISTORY_LENGTH = HISTORY_LENGTH
    FIGHTER_1_COLOR = getattr(CFG, 'FIGHTER_1_COLOR', (255, 100, 100)) if 'CFG' in globals() else (255, 100, 100)
    FIGHTER_2_COLOR = getattr(CFG, 'FIGHTER_2_COLOR', (100, 100, 255)) if 'CFG' in globals() else (100, 100, 255)
    TRAIL_LENGTH = getattr(CFG, 'TRAIL_LENGTH', 30) if 'CFG' in globals() else 30
    EVENT_POPUP_FRAMES = getattr(CFG, 'EVENT_POPUP_FRAMES', 45) if 'CFG' in globals() else 45
    TEMPORAL_SMOOTHING_FACTOR = TEMPORAL_SMOOTHING_FACTOR
    ACTION_COOLDOWN = ACTION_COOLDOWN
    SMOOTHING_WINDOW = SMOOTHING_WINDOW
    OUTPUT_VIDEO_CODEC = OUTPUT_VIDEO_CODEC
    OUTPUT_VIDEO_FPS = OUTPUT_VIDEO_FPS
    COLORS = COLORS
    STATS_CSV = STATS_CSV
    SUMMARY_JSON = SUMMARY_JSON


CFG = _Config()
