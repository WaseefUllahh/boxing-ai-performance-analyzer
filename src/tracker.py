"""
src/tracker.py — BoT-SORT / ByteTrack integration via Ultralytics.

Responsibilities
----------------
- Maintain consistent fighter IDs across frames.
- Accept a raw BGR frame + list of raw detections from detector.py.
- Return TrackedFighter objects augmented with a stable ``track_id``.

Design notes
------------
Ultralytics ships BoT-SORT and ByteTrack as built-in tracker configs,
so we leverage ``model.track()`` rather than calling a separate library.
This avoids version-pinning headaches with standalone BoT-SORT repos.

Returned structure per frame
-----------------------------
A list of TrackedFighter dicts:

    {
        "track_id":   int,                 # persistent across frames
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
        """
        Parameters
        ----------
        model_name : str
            Ultralytics YOLO Pose model identifier.
        tracker_cfg : str
            Tracker config file name shipped with Ultralytics
            (``botsort.yaml`` or ``bytetrack.yaml``).
        confidence : float
            Detection confidence threshold.
        iou : float
            NMS IoU threshold.
        max_fighters : int
            Only the top-N detections (by confidence) are kept per frame.
        device : str
            Torch device string.
        """
        self.model_name = model_name
        self.tracker_cfg = tracker_cfg
        self.confidence = confidence
        self.iou = iou
        self.max_fighters = max_fighters
        self.device = device

        self._model: _UltralyticsModel = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        from ultralytics import YOLO

        print(f"[Tracker] Loading model: {self.model_name}")
        self._model = YOLO(self.model_name)
        print(f"[Tracker] Using tracker: {self.tracker_cfg}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection + tracking on a single BGR frame.

        Returns a list of tracked fighter dicts sorted by track_id.
        Returns an empty list if no fighters are detected.
        """
        if self._model is None:
            self._load_model()

        results = self._model.track(
            frame,
            tracker=self.tracker_cfg,
            conf=self.confidence,
            iou=self.iou,
            persist=True,         # keep tracker state between calls
            verbose=False,
        )

        tracked: list[dict] = []

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

                tracked.append(
                    {
                        "track_id": track_id,
                        "bbox": xyxy.tolist(),
                        "confidence": round(conf, 4),
                        "keypoints": kp_data,
                        "center": (cx, cy),
                    }
                )

        # Sort by confidence descending, keep only top-N fighters
        tracked.sort(key=lambda d: d["confidence"], reverse=True)
        tracked = tracked[: self.max_fighters]

        # Re-sort by track_id for deterministic frame-by-frame iteration
        tracked.sort(key=lambda d: d["track_id"])

        return tracked
