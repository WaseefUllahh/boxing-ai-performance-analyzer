"""
src/fight_analyzer.py — Frame aggregation, statistics, and export.

Responsibilities
----------------
- Accept one frame's worth of tracked + classified data.
- Maintain per-fighter rolling and cumulative statistics.
- Detect clinches (two fighters very close together).
- Track movement: total distance travelled, aggression index.
- At end-of-fight, produce a summary dict and export CSV / JSON.

No model inference happens here — pure data aggregation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import CFG
from src.pose_features import PoseFeatures
from src.strike_detector import StrikeResult
from src.defense_detector import DefenseResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    """All data captured for a single fighter in a single frame."""
    frame_idx:      int
    track_id:       int
    center_x:       float
    center_y:       float
    strike_label:   str
    strike_arm:     str
    defense_label:  str
    left_ext:       float
    right_ext:      float
    torso_lean:     float
    hip_norm:       float
    stance_norm:    float
    is_clinch:      bool = False


@dataclass
class FighterStats:
    """Cumulative statistics for a single fighter."""
    track_id:           int
    total_punches:      int = 0
    jabs:               int = 0
    crosses:            int = 0
    hooks:              int = 0
    uppercuts:          int = 0
    guards:             int = 0
    slips:              int = 0
    ducks:              int = 0
    clinches:           int = 0
    frames_detected:    int = 0
    total_distance_px:  float = 0.0
    aggression_score:   float = 0.0   # punches / frames_detected
    last_center: Optional[tuple[float, float]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

CLINCH_DISTANCE_RATIO = 0.8   # fraction of shoulder-width: centres this close = clinch

class FightAnalyzer:
    """
    Aggregates per-frame analysis results into fight statistics.

    Usage
    -----
    analyzer = FightAnalyzer()
    for frame_idx, fighters in enumerate(per_frame_data):
        analyzer.update(frame_idx, fighters, strikes, defenses, features)
    summary = analyzer.finalize()
    """

    def __init__(
        self,
        stats_csv:    Path = CFG.STATS_CSV,
        summary_json: Path = CFG.SUMMARY_JSON,
    ) -> None:
        self.stats_csv    = stats_csv
        self.summary_json = summary_json

        self._records: list[FrameRecord] = []
        self._fighter_stats: dict[int, FighterStats] = defaultdict(
            lambda: FighterStats(track_id=-1)
        )

    # ------------------------------------------------------------------
    def update(
        self,
        frame_idx: int,
        tracked_fighters: list[dict],               # from tracker.py
        strikes:   dict[int, StrikeResult],         # track_id → StrikeResult
        defenses:  dict[int, DefenseResult],         # track_id → DefenseResult
        features:  dict[int, PoseFeatures],          # track_id → PoseFeatures
    ) -> None:
        """
        Ingest one frame's worth of analysis data.

        Parameters
        ----------
        frame_idx : int
        tracked_fighters : list[dict]
            Output of FighterTracker.update() — contains track_id, bbox, center.
        strikes : dict[int, StrikeResult]
        defenses : dict[int, DefenseResult]
        features : dict[int, PoseFeatures]
        """
        # ── Clinch detection: are fighter centres very close? ─────────────
        is_clinch = self._detect_clinch(tracked_fighters, features)

        for fighter in tracked_fighters:
            tid = fighter["track_id"]
            cx, cy = fighter["center"]

            strike  = strikes.get(tid,  StrikeResult(track_id=tid))
            defense = defenses.get(tid, DefenseResult(track_id=tid))
            feat    = features.get(tid, PoseFeatures(track_id=tid))

            # ── Update stats ──────────────────────────────────────────────
            stats = self._fighter_stats[tid]
            stats.track_id = tid
            stats.frames_detected += 1

            if strike.label != "NONE":
                stats.total_punches += 1
                if   strike.label == "JAB":      stats.jabs      += 1
                elif strike.label == "CROSS":    stats.crosses   += 1
                elif strike.label == "HOOK":     stats.hooks     += 1
                elif strike.label == "UPPERCUT": stats.uppercuts += 1

            if defense.label == "GUARD":   stats.guards  += 1
            if defense.label == "SLIP":    stats.slips   += 1
            if defense.label == "DUCK":    stats.ducks   += 1
            if is_clinch:                  stats.clinches += 1

            # Distance travelled
            if stats.last_center is not None:
                d = np.hypot(cx - stats.last_center[0], cy - stats.last_center[1])
                if d >= CFG.MOVEMENT_MIN_PIXELS:
                    stats.total_distance_px += d
            stats.last_center = (cx, cy)

            # ── Append frame record ───────────────────────────────────────
            self._records.append(FrameRecord(
                frame_idx     = frame_idx,
                track_id      = tid,
                center_x      = cx,
                center_y      = cy,
                strike_label  = strike.label,
                strike_arm    = strike.arm,
                defense_label = defense.label,
                left_ext      = feat.left_arm_extension,
                right_ext     = feat.right_arm_extension,
                torso_lean    = feat.torso_lean_deg,
                hip_norm      = feat.hip_height_norm,
                stance_norm   = feat.stance_width_norm,
                is_clinch     = is_clinch,
            ))

    # ------------------------------------------------------------------
    def finalize(self) -> dict:
        """
        Compute final statistics, write CSV and JSON, return summary dict.

        Returns
        -------
        dict
            fight_summary with per-fighter stats and metadata.
        """
        # ── Aggression score ──────────────────────────────────────────────
        for tid, stats in self._fighter_stats.items():
            if stats.frames_detected > 0:
                stats.aggression_score = round(
                    stats.total_punches / stats.frames_detected, 4
                )

        # ── Write CSV ─────────────────────────────────────────────────────
        if self._records:
            df = pd.DataFrame([vars(r) for r in self._records])
            df.to_csv(self.stats_csv, index=False)
            print(f"[Analyzer] Saved frame stats → {self.stats_csv}")

        # ── Build summary ─────────────────────────────────────────────────
        summary: dict = {
            "total_frames_processed": len({r.frame_idx for r in self._records}),
            "fighters": {},
        }

        for tid, stats in self._fighter_stats.items():
            summary["fighters"][str(tid)] = {
                "track_id":          stats.track_id,
                "frames_detected":   stats.frames_detected,
                "total_punches":     stats.total_punches,
                "jabs":              stats.jabs,
                "crosses":           stats.crosses,
                "hooks":             stats.hooks,
                "uppercuts":         stats.uppercuts,
                "guards":            stats.guards,
                "slips":             stats.slips,
                "ducks":             stats.ducks,
                "clinches":          stats.clinches,
                "total_distance_px": round(stats.total_distance_px, 1),
                "aggression_score":  stats.aggression_score,
            }

        # ── Write JSON ────────────────────────────────────────────────────
        self.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Analyzer] Saved fight summary → {self.summary_json}")

        return summary

    # ------------------------------------------------------------------
    # Accessors for dashboard
    # ------------------------------------------------------------------
    def get_records_df(self) -> pd.DataFrame:
        """Return all frame records as a DataFrame (for dashboard use)."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([vars(r) for r in self._records])

    def get_fighter_stats(self) -> dict[int, FighterStats]:
        return dict(self._fighter_stats)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_clinch(
        fighters: list[dict],
        features: dict[int, PoseFeatures],
    ) -> bool:
        """
        Returns True if two fighters' centres are within 0.8 × shoulder_width
        of each other (approximate clinch / grappling range).
        """
        if len(fighters) < 2:
            return False
        c1 = fighters[0]["center"]
        c2 = fighters[1]["center"]
        dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])

        # Use average shoulder width as reference
        sw_vals = [
            f.shoulder_width for f in features.values()
            if f.valid and f.shoulder_width > 1.0
        ]
        ref = float(np.mean(sw_vals)) if sw_vals else 100.0
        return bool(dist < CLINCH_DISTANCE_RATIO * ref)
