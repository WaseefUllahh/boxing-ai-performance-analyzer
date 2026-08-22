"""
src/detector.py — YOLO Pose detector wrapper.

Responsibilities
----------------
- Load a pretrained Ultralytics YOLO Pose model once.
- Accept a single BGR frame (np.ndarray) and return structured detections.
- Keep all Ultralytics-specific logic inside this module so the rest of
  the codebase never touches the Ultralytics API directly.

Returned structure per frame
-----------------------------
A list of dicts, one per detected person:

    {
        "bbox":       [x1, y1, x2, y2],   # ints, absolute pixels
        "confidence": float,               # 0-1
        "keypoints":  np.ndarray,          # shape (17, 3) — [x, y, conf]
    }

NOTE: Tracking IDs are NOT assigned here; that is the tracker's job.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Type alias — avoids importing Ultralytics at the module level so unit
# tests can import detector.py without the full GPU stack.
# ---------------------------------------------------------------------------
_UltralyticsModel = Any  # ultralytics.YOLO instance


class PoseDetector:
    """Thin wrapper around Ultralytics YOLO Pose inference."""

    def __init__(
        self,
        model_name: str = "yolov8n-pose",
        confidence: float = 0.40,
        iou: float = 0.45,
        device: str = "",  # "" = auto (GPU if available, else CPU)
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            Ultralytics model identifier or local .pt path.
        confidence : float
            Minimum detection confidence.
        iou : float
            NMS IoU threshold.
        device : str
            Torch device string.  Leave empty for automatic selection.
        """
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self._model: _UltralyticsModel = None

    # ------------------------------------------------------------------
    # Lazy loading — model is loaded on first call to detect()
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Download (if needed) and load the YOLO Pose model."""
        from ultralytics import YOLO  # heavy import, deferred on purpose

        print(f"[Detector] Loading model: {self.model_name}")
        self._model = YOLO(self.model_name)
        # Warm-up: run a dummy inference so the first real frame is fast
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)
        print("[Detector] Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image as returned by cv2.VideoCapture.read().

        Returns
        -------
        list[dict]
            One dict per detected person with keys:
            ``bbox``, ``confidence``, ``keypoints``.
        """
        if self._model is None:
            self._load_model()

        results = self._model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        detections: list[dict] = []

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue

            boxes = result.boxes
            kps = result.keypoints

            for i in range(len(boxes)):
                # Bounding box (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())

                # Keypoints: shape (17, 3) or (17, 2)
                kp_data = kps.data[i].cpu().numpy()  # (17, 3) [x, y, conf]

                # Guard: ensure shape is always (17, 3)
                if kp_data.ndim == 2 and kp_data.shape[1] == 2:
                    # No confidence channel — append zeros
                    kp_conf = np.zeros((kp_data.shape[0], 1), dtype=np.float32)
                    kp_data = np.hstack([kp_data, kp_conf])

                detections.append(
                    {
                        "bbox": xyxy.tolist(),           # [x1, y1, x2, y2]
                        "confidence": round(conf, 4),
                        "keypoints": kp_data,            # np.ndarray (17, 3)
                    }
                )

        return detections
