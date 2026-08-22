"""
src/events.py — Canonical event structures for the pipeline.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class FightEvent:
    """A canonical event (strike or defense) in the fight."""
    fighter_id: int
    frame_number: int
    timestamp: float
    category: str              # "STRIKE" or "DEFENSE"
    action: str                # e.g., "JAB", "HOOK", "BLOCK", "DODGE"
    confidence: float          # 0.0 - 1.0

    # Optional fields depending on category
    hand: Optional[str] = None
    event_type: Optional[str] = None         # "POSSIBLE_LANDED", "POSSIBLE_BLOCKED", "POSSIBLE_MISSED"
    target_zone_estimate: Optional[str] = None
    wrist_position: Optional[tuple[float, float]] = None
    opponent_distance: Optional[float] = None
    supporting_features: Optional[str] = None
