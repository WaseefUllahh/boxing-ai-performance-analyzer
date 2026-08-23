"""
src/tracker.py — Pairwise Combat Tracking with Robust Identity Management.

Responsibilities
----------------
- Maintain stable, persistent fighter identities (Fighter 1 and Fighter 2).
- Automatically detect and ignore the referee and background spectators.
- Leverage pairwise combat clustering (boxers operate as a mutually engaged pair).
- Pose-based posture scoring (boxer guard vs referee standing posture).
- Occlusion-resistant track re-identification without referee hijacking.
- Provide debug annotations with person roles and combat scores.
"""

from __future__ import annotations

from typing import Any, List, Dict, Tuple, Optional
import numpy as np

_UltralyticsModel = Any


class PairwiseIdentityManager:
    """
    Maintains stable Fighter 1 and Fighter 2 identities using:
    1. Pairwise combat distance and ring centrality.
    2. Boxer posture scoring (guard elevation, hand positioning).
    3. Trajectory coasting and anti-hijacking during occlusions.
    """

    def __init__(
        self,
        warmup_frames: int = 15,
        max_primary: int = 2,
        max_reid_dist: float = 300.0,
        max_pair_dist: float = 950.0,
        min_pair_dist: float = 120.0,
    ):
        self.warmup_frames = warmup_frames
        self.max_primary = max_primary
        self.max_reid_dist = max_reid_dist
        self.max_pair_dist = max_pair_dist
        self.min_pair_dist = min_pair_dist

        self.frame_count = 0
        self.primary_tracks: Dict[int, int] = {}  # app_id (1, 2) -> bot_sort_id
        self.app_state: Dict[int, Dict[str, Any]] = {}  # app_id -> last known track dict
        self.coast_counts: Dict[int, int] = {1: 0, 2: 0}  # frames since last direct detection

    @staticmethod
    def _bbox_area(bbox: List[int]) -> float:
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    @staticmethod
    def _center_dist(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
        return float(((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5)

    def _compute_boxer_posture_score(self, track: Dict[str, Any]) -> float:
        """
        Computes a combat posture score [0.0 - 1.0].
        Boxers have hands raised in guard near chest/chin and wider athletic stance.
        Referees typically have hands hanging low by hips/waist.
        """
        kps = track.get("keypoints")
        if kps is None or len(kps) < 17:
            return 0.5

        # Keypoint indices: 5=l_sh, 6=r_sh, 9=l_wrist, 10=r_wrist, 11=l_hip, 12=r_hip
        l_sh, r_sh = kps[5], kps[6]
        l_wr, r_wr = kps[9], kps[10]
        l_hip, r_hip = kps[11], kps[12]

        score = 0.5
        # Shoulder center & hip center Y
        if l_sh[2] > 0.25 and r_sh[2] > 0.25 and l_hip[2] > 0.25 and r_hip[2] > 0.25:
            sh_y = (l_sh[1] + r_sh[1]) / 2.0
            hip_y = (l_hip[1] + r_hip[1]) / 2.0
            torso_h = max(20.0, hip_y - sh_y)

            # Check wrist elevation relative to torso
            guard_wrists = 0
            if l_wr[2] > 0.25 and l_wr[1] < hip_y + 0.1 * torso_h:
                guard_wrists += 1
            if r_wr[2] > 0.25 and r_wr[1] < hip_y + 0.1 * torso_h:
                guard_wrists += 1

            if guard_wrists == 2:
                score += 0.35
            elif guard_wrists == 1:
                score += 0.15
            else:
                score -= 0.25

        return max(0.0, min(1.0, score))

    def _score_candidate_pair(
        self, t1: Dict[str, Any], t2: Dict[str, Any], frame_w: int, frame_h: int
    ) -> float:
        """
        Evaluates the mutual likelihood that (t1, t2) is the primary boxer pair.
        """
        c1, c2 = t1["center"], t2["center"]
        inter_dist = self._center_dist(c1, c2)

        # 1. Distance score (boxers fight at 200-850px)
        if self.min_pair_dist <= inter_dist <= self.max_pair_dist:
            dist_score = 1.0 - abs(inter_dist - 500.0) / 600.0
        else:
            dist_score = 0.1

        # 2. Ring Centrality score (pair midpoint close to ring center)
        mid_x = (c1[0] + c2[0]) / 2.0
        mid_y = (c1[1] + c2[1]) / 2.0
        norm_cx = abs(mid_x - frame_w / 2.0) / (frame_w / 2.0)
        norm_cy = abs(mid_y - frame_h * 0.55) / (frame_h / 2.0)
        center_score = max(0.0, 1.0 - 0.6 * norm_cx - 0.4 * norm_cy)

        # 3. Area similarity score
        a1 = self._bbox_area(t1["bbox"])
        a2 = self._bbox_area(t2["bbox"])
        area_ratio = min(a1, a2) / max(1.0, max(a1, a2))

        # 4. Posture scores
        p1 = self._compute_boxer_posture_score(t1)
        p2 = self._compute_boxer_posture_score(t2)

        total_score = (
            0.30 * dist_score
            + 0.25 * center_score
            + 0.20 * area_ratio
            + 0.25 * ((p1 + p2) / 2.0)
        )
        return float(total_score)

    def update(
        self, raw_tracked: List[Dict[str, Any]], frame_w: int = 1920, frame_h: int = 1080
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Updates fighter tracking and returns:
            (assigned_fighters, all_annotated_candidates)
        """
        self.frame_count += 1

        # Tag all candidates initially as unassigned / candidate
        for t in raw_tracked:
            t["combat_score"] = self._compute_boxer_posture_score(t)
            t["role"] = "CANDIDATE"

        # ── 1. Warmup Phase (Find the best mutual combat pair) ─────────────
        if self.frame_count <= self.warmup_frames or len(self.primary_tracks) < 2:
            if len(raw_tracked) >= 2:
                best_pair = None
                best_score = -1.0
                for i in range(len(raw_tracked)):
                    for j in range(i + 1, len(raw_tracked)):
                        pair_score = self._score_candidate_pair(
                            raw_tracked[i], raw_tracked[j], frame_w, frame_h
                        )
                        if pair_score > best_score:
                            best_score = pair_score
                            best_pair = (raw_tracked[i], raw_tracked[j])

                if best_pair is not None and (
                    self.frame_count == self.warmup_frames or len(self.primary_tracks) < 2
                ):
                    # Sort left-to-right on initial lock
                    tA, tB = best_pair
                    if tA["center"][0] > tB["center"][0]:
                        tA, tB = tB, tA

                    self.primary_tracks[1] = tA["track_id"]
                    self.primary_tracks[2] = tB["track_id"]

            # During warmup return top detections
            assigned = []
            for i, t in enumerate(raw_tracked[:2]):
                app_id = i + 1
                t["bot_sort_id"] = t["track_id"]
                t["track_id"] = app_id
                t["role"] = f"Fighter {app_id}"
                self.app_state[app_id] = t
                assigned.append(t)

            for t in raw_tracked[2:]:
                t["bot_sort_id"] = t["track_id"]
                t["role"] = "IGNORED (Referee/Spectator)"

            return assigned, raw_tracked

        # ── 2. Post-Warmup Active Tracking & Anti-Hijacking ─────────────────
        present_tids = {t["track_id"]: t for t in raw_tracked}
        assigned_results: List[Dict[str, Any]] = []
        missing_app_ids: List[int] = []

        # A. Match known persistent primary IDs
        for app_id in [1, 2]:
            tid = self.primary_tracks.get(app_id)
            if tid is not None and tid in present_tids:
                t = present_tids[tid]
                t["bot_sort_id"] = tid
                t["track_id"] = app_id
                t["role"] = f"Fighter {app_id}"
                self.app_state[app_id] = t
                self.coast_counts[app_id] = 0
                assigned_results.append(t)
            else:
                missing_app_ids.append(app_id)

        # B. Handle missing fighters with anti-hijacking re-ID
        if missing_app_ids:
            unassigned = [
                t
                for t in raw_tracked
                if t["track_id"] not in [t_res.get("bot_sort_id") for t_res in assigned_results]
            ]

            other_app_id = 2 if 1 in missing_app_ids else 1
            other_fighter_pos = (
                self.app_state[other_app_id]["center"]
                if other_app_id in self.app_state
                else (frame_w / 2, frame_h / 2)
            )

            for app_id in missing_app_ids:
                self.coast_counts[app_id] += 1
                last_known = self.app_state.get(app_id)
                if last_known is None:
                    continue

                last_pos = last_known["center"]
                last_area = self._bbox_area(last_known["bbox"])

                best_candidate = None
                best_cost = float("inf")

                for cand in unassigned:
                    cand_pos = cand["center"]
                    cand_area = self._bbox_area(cand["bbox"])

                    dist_to_last = self._center_dist(last_pos, cand_pos)
                    dist_to_other = self._center_dist(cand_pos, other_fighter_pos)
                    area_ratio = min(last_area, cand_area) / max(1.0, max(last_area, cand_area))
                    posture = cand.get("combat_score", 0.5)

                    # REJECT referee hijack:
                    # If candidate is outside fighting distance from partner, or area is tiny, reject!
                    if dist_to_other > self.max_pair_dist or dist_to_last > self.max_reid_dist:
                        continue

                    # Cost function (lower is better)
                    cost = (
                        (dist_to_last / self.max_reid_dist) * 0.40
                        + (dist_to_other / self.max_pair_dist) * 0.25
                        + (1.0 - area_ratio) * 0.20
                        + (1.0 - posture) * 0.15
                    )

                    if cost < best_cost and cost < 0.65:
                        best_cost = cost
                        best_candidate = cand

                if best_candidate is not None:
                    new_tid = best_candidate["track_id"]
                    self.primary_tracks[app_id] = new_tid
                    best_candidate["bot_sort_id"] = new_tid
                    best_candidate["track_id"] = app_id
                    best_candidate["role"] = f"Fighter {app_id}"
                    self.app_state[app_id] = best_candidate
                    self.coast_counts[app_id] = 0
                    assigned_results.append(best_candidate)
                    unassigned.remove(best_candidate)

        # Label any remaining unassigned detections as IGNORED
        for t in raw_tracked:
            if t not in assigned_results:
                t["bot_sort_id"] = t.get("track_id", -1)
                t["role"] = "IGNORED (Referee/Spectator)"

        assigned_results.sort(key=lambda d: d["track_id"])
        return assigned_results, raw_tracked


class FighterTracker:
    """
    Wraps YOLO Pose + BoT-SORT with PairwiseIdentityManager.
    """

    def __init__(
        self,
        model_name: str = "yolov8n-pose",
        tracker_cfg: str = "botsort.yaml",
        confidence: float = 0.35,
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
        self._identity_manager = PairwiseIdentityManager(max_primary=max_fighters)
        self.last_all_candidates: List[Dict[str, Any]] = []

    def _load_model(self) -> None:
        from ultralytics import YOLO

        print(f"[Tracker] Loading model: {self.model_name}")
        self._model = YOLO(self.model_name)
        print(f"[Tracker] Using tracker: {self.tracker_cfg}")

    def update(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection + tracking on a single BGR frame.
        Returns stable fighter dicts (Fighter 1 and Fighter 2).
        """
        if self._model is None:
            self._load_model()

        h, w = frame.shape[:2]

        results = self._model.track(
            frame,
            tracker=self.tracker_cfg,
            conf=self.confidence,
            iou=self.iou,
            persist=True,
            verbose=False,
            device=self.device if self.device else None,
        )

        raw_tracked: List[Dict[str, Any]] = []

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue

            boxes = result.boxes
            kps = result.keypoints

            for i in range(len(boxes)):
                if boxes.id is None:
                    continue
                track_id = int(boxes.id[i].cpu().numpy())
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())

                kp_data = kps.data[i].cpu().numpy()
                if kp_data.ndim == 2 and kp_data.shape[1] == 2:
                    kp_conf = np.zeros((kp_data.shape[0], 1), dtype=np.float32)
                    kp_data = np.hstack([kp_data, kp_conf])

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

        stable_tracked, all_annotated = self._identity_manager.update(raw_tracked, w, h)
        self.last_all_candidates = all_annotated
        return stable_tracked
