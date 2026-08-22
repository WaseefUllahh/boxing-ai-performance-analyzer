"""
src/__init__.py

Public re-exports for the src package.
Import sub-modules explicitly to avoid circular imports.
"""

__version__ = "0.1.0"
__all__ = [
    "video_io",
    "detector",
    "tracker",
    "pose_features",
    "strike_detector",
    "defense_detector",
    "fight_analyzer",
    "video_processor",
]

