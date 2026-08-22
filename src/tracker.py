"""
src/tracker.py — BoT-SORT / ByteTrack integration with Identity Management.

Responsibilities
----------------
- Maintain consistent fighter IDs across frames.
- Accept a raw BGR frame + list of raw detections from detector.py.
- Use IdentityManager to filter out background people and handle track switches.
- Return TrackedFighter objects augmented with a stable application-level ``track_id`` (1 or 2).

Design notes
------------
Ultralytics ships BoT-SORT and ByteTrack as built-in tracker configs,
so we leverage ``model.track()`` rather than calling a separate library.

Returned structure per frame
-----------------------------
A list of TrackedFighter dicts:

    {
        "track_id":   int,                 # stable application-level ID (1 or 2)
        "bot_sort_id": int,                # underlying raw tracker ID (for debug)
        "bbox":       [x1, y1, x2, y2],
        "confidence": float,
        "keypoints":  np.ndarray (17, 3),
        "center":     (cx, cy),            # bounding-box centroid (float)
    }
"""

from __future__ import annotations

from typing import Any

import numpy as np

_UltralyticsModel = Any  # ultralytics.YOLO, imported lazily


class IdentityManager:
    """
    Maps volatile BoT-SORT track IDs to stable application identities (1 and 2).
    
    Uses a warm-up phase to identify the two most prominent, persistent fighters,
    then actively attempts to re-associate new track IDs if a fighter is temporarily lost.
    """
    
    def __init__(self, warmup_frames: int = 15, max_primary: int = 2, max_reid_dist: float = 200.0):
        self.warmup_frames = warmup_frames
        self.max_primary = max_primary
        self.max_reid_dist = max_reid_dist
        
        self.frame_count = 0
        self.track_stats: dict[int, dict] = {}     # bot_sort_id -> {"hits": int, "area_sum": float}
        self.primary_tracks: dict[int, int] = {}   # app_id (1,2) -> bot_sort_id
        self.app_state: dict[int, dict] = {}       # app_id -> last known track dict
        
    @staticmethod
    def _bbox_area(bbox: list[int]) -> float:
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
        
    @staticmethod
    def _center_dist(c1: tuple[float, float], c2: tuple[float, float]) -> float:
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

    def update(self, raw_tracked: list[dict]) -> list[dict]:
        self.frame_count += 1
        
        # 1. Update track stats for all detected tracks
        for t in raw_tracked:
            tid = t["track_id"]
            if tid not in self.track_stats:
                self.track_stats[tid] = {"hits": 0, "area_sum": 0.0}
            self.track_stats[tid]["hits"] += 1
            self.track_stats[tid]["area_sum"] += self._bbox_area(t["bbox"])
            
        # 2. Warm-up Phase
        if self.frame_count <= self.warmup_frames:
            # Sort raw tracks by confidence to return consistent temporary IDs
            raw_tracked.sort(key=lambda d: d["confidence"], reverse=True)
            
            if self.frame_count == self.warmup_frames:
                # Lock in primary tracks at the end of warmup
                scores = []
                for tid, stats in self.track_stats.items():
                    avg_area = stats["area_sum"] / max(1, stats["hits"])
                    score = stats["hits"] * avg_area
                    scores.append((tid, score))
                scores.sort(key=lambda x: x[1], reverse=True)
                
                for i in range(min(self.max_primary, len(scores))):
                    app_id = i + 1
                    tid = scores[i][0]
                    self.primary_tracks[app_id] = tid
            
            # During warmup, just map the top N detections to IDs 1, 2, ...
            for i, t in enumerate(raw_tracked[:self.max_primary]):
                t["bot_sort_id"] = t["track_id"]
                t["track_id"] = i + 1
                self.app_state[i+1] = t
            return raw_tracked[:self.max_primary]
            
        # 3. Post-Warmup Phase: Active Identity Management
        present_tids = {t["track_id"]: t for t in raw_tracked}
        assigned_results = []
        missing_app_ids = []
        
        # A. Assign known primary tracks
        for app_id, tid in self.primary_tracks.items():
            if tid in present_tids:
                t = present_tids[tid]
                t["bot_sort_id"] = tid
                t["track_id"] = app_id
                self.app_state[app_id] = t
                assigned_results.append(t)
            else:
                missing_app_ids.append(app_id)
                
        # B. Attempt Re-Identification for missing tracks
        if missing_app_ids:
            unassigned_tracks = [t for t in raw_tracked if t["track_id"] not in self.primary_tracks.values()]
            
            for app_id in missing_app_ids:
                if app_id not in self.app_state: continue
                last_t = self.app_state[app_id]
                last_center = last_t["center"]
                
                best_track = None
                best_dist = self.max_reid_dist
                
                for t in unassigned_tracks:
                    dist = self._center_dist(last_center, t["center"])
                    if dist < best_dist:
                        best_dist = dist
                        best_track = t
                        
                if best_track:
                    new_tid = best_track["track_id"]
                    self.primary_tracks[app_id] = new_tid  # Update mapping
                    best_track["bot_sort_id"] = new_tid
                    best_track["track_id"] = app_id
                    self.app_state[app_id] = best_track
                    assigned_results.append(best_track)
                    unassigned_tracks.remove(best_track)
                    
        # Sort by app_id for consistent processing order
        assigned_results.sort(key=lambda d: d["track_id"])
        return assigned_results


class FighterTracker:
    """
    Wraps Ultralytics' integrated tracker (BoT-SORT or ByteTrack).
    
    The tracker is stateful — it must be called every frame in order,
    and a fresh instance should be created for each new video.
    """

    def __init__(
        self,
        model_name: str = "yolov8n-pose",
        tracker_cfg: str = "botsort.yaml",
        confidence: float = 0.40,
        iou: float = 0.45,
        max_fighters: int = 2,
        device: str = "",
    ) -> None:
        self.model_name = model_name
        self.tracker_cfg = tracker_cfg
        self.confidence = confidence
        self.iou = iou
        self.max_fighters = max_fighters
        self.device = device

        self._model: _UltralyticsModel = None
        self._identity_manager = IdentityManager(max_primary=max_fighters)

    def _load_model(self) -> None:
        from ultralytics import YOLO

        print(f"[Tracker] Loading model: {self.model_name}")
        self._model = YOLO(self.model_name)
        print(f"[Tracker] Using tracker: {self.tracker_cfg}")

    def update(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection + tracking on a single BGR frame.

        Returns a list of tracked fighter dicts with stable track_ids (1 or 2).
        Returns an empty list if no fighters are detected.
        """
        if self._model is None:
            self._load_model()

        # Run Ultralytics tracking
        results = self._model.track(
            frame,
            tracker=self.tracker_cfg,
            conf=self.confidence,
            iou=self.iou,
            persist=True,         # keep tracker state between calls
            verbose=False,
            device=self.device if self.device else None,
        )

        raw_tracked: list[dict] = []

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue

            boxes = result.boxes
            kps = result.keypoints

            for i in range(len(boxes)):
                # Track ID may be None if the tracker hasn't assigned one yet
                if boxes.id is None:
                    continue
                track_id = int(boxes.id[i].cpu().numpy())
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())

                kp_data = kps.data[i].cpu().numpy()  # (17, 3)
                if kp_data.ndim == 2 and kp_data.shape[1] == 2:
                    kp_conf = np.zeros((kp_data.shape[0], 1), dtype=np.float32)
                    kp_data = np.hstack([kp_data, kp_conf])

                # Bounding-box centre
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                raw_tracked.append(
                    {
                        "track_id": track_id,
                        "bbox": xyxy.tolist(),
                        "confidence": round(conf, 4),
                        "keypoints": kp_data,
                        "center": (cx, cy),
                    }
                )

        # Apply identity management to filter background and stabilize IDs
        stable_tracked = self._identity_manager.update(raw_tracked)

        return stable_tracked
