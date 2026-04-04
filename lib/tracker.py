"""
Temporal Smoothing & Object Tracking

Provides Kalman filter and EMA smoothing for bounding box stabilization.
Prevents overlay flickering between frames.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

logger = logging.getLogger("adstream.tracker")


@dataclass
class TrackedObject:
    """A tracked object with smoothed position."""

    track_id: int
    class_id: int
    class_name: str
    bbox: np.ndarray  # Smoothed [x1, y1, x2, y2]
    raw_bbox: np.ndarray  # Raw detection [x1, y1, x2, y2]
    confidence: float
    mask: Optional[np.ndarray] = None
    frames_seen: int = 0
    frames_missing: int = 0
    last_seen: float = 0.0
    is_visible: bool = True


class EMABboxSmoother:
    """Exponential Moving Average smoother for bounding boxes."""

    def __init__(self, alpha: float = 0.35):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive, lower = smoother.
        """
        self.alpha = alpha
        self._prev: Optional[np.ndarray] = None

    def smooth(self, bbox: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to a bounding box [x1, y1, x2, y2]."""
        bbox = np.asarray(bbox, dtype=np.float64)
        if self._prev is None:
            self._prev = bbox.copy()
            return bbox

        smoothed = self.alpha * bbox + (1 - self.alpha) * self._prev
        self._prev = smoothed.copy()
        return smoothed

    def reset(self):
        self._prev = None


class KalmanBboxTracker:
    """
    Kalman filter for smooth bounding box tracking.
    State: [cx, cy, w, h, vcx, vcy, vw, vh] (center + size + velocities)
    """

    def __init__(self):
        # 8 state dims, 4 measurement dims
        self.kf = cv2.KalmanFilter(8, 4)

        # Transition matrix: constant velocity model
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0

        # Measurement matrix: we observe [cx, cy, w, h]
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0

        # Process noise — how much we trust the model
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        # Position components have lower noise
        for i in range(4):
            self.kf.processNoiseCov[i, i] = 5e-3
        # Velocity components have higher noise
        for i in range(4, 8):
            self.kf.processNoiseCov[i, i] = 2e-2

        # Measurement noise — how much we trust detections
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 5e-2

        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1.0
        self.kf.errorCovPre = np.eye(8, dtype=np.float32) * 1.0

        self._initialized = False

    def _xyxy_to_cxcywh(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
        x1, y1, x2, y2 = bbox
        return np.array(
            [
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                x2 - x1,
                y2 - y1,
            ],
            dtype=np.float32,
        )

    def _cxcywh_to_xyxy(self, cxcywh: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
        cx, cy, w, h = cxcywh
        return np.array(
            [
                cx - w / 2,
                cy - h / 2,
                cx + w / 2,
                cy + h / 2,
            ],
            dtype=np.float64,
        )

    def predict(self) -> np.ndarray:
        """Predict next state, return bbox [x1, y1, x2, y2]."""
        state = self.kf.predict()
        return self._cxcywh_to_xyxy(state[:4].flatten())

    def update(self, bbox: np.ndarray) -> np.ndarray:
        """
        Update with new detection.
        Returns smoothed bbox [x1, y1, x2, y2].
        """
        measurement = self._xyxy_to_cxcywh(bbox).reshape(4, 1)

        if not self._initialized:
            self.kf.statePre[:4] = measurement
            self.kf.statePre[4:] = 0  # zero velocity
            self.kf.statePost[:4] = measurement
            self.kf.statePost[4:] = 0
            self._initialized = True
            return self._cxcywh_to_xyxy(measurement.flatten())

        self.kf.predict()
        self.kf.correct(measurement)
        state = self.kf.statePost[:4].flatten()
        return self._cxcywh_to_xyxy(state)


class ObjectTracker:
    """
    Multi-object tracker with temporal smoothing.

    Matches new detections to existing tracks via IoU,
    applies Kalman or EMA smoothing to each track.
    """

    def __init__(
        self,
        smoothing_method: str = "kalman",
        ema_alpha: float = 0.35,
        iou_threshold: float = 0.3,
        max_missing_frames: int = 15,
        min_visible_frames: int = 3,
    ):
        self.smoothing_method = smoothing_method
        self.ema_alpha = ema_alpha
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.min_visible_frames = min_visible_frames

        self._tracks: dict[int, dict] = {}  # track_id -> track data
        self._next_id = 0

    def update(self, detections: list) -> list[TrackedObject]:
        """
        Update tracker with new detections.

        Args:
            detections: List of Detection objects from the detector.

        Returns:
            List of TrackedObject with smoothed positions.
        """
        now = time.time()

        if not detections:
            # No detections — increment missing count for all tracks
            for tid in list(self._tracks.keys()):
                self._tracks[tid]["frames_missing"] += 1
                if self._tracks[tid]["frames_missing"] > self.max_missing_frames:
                    del self._tracks[tid]
                else:
                    # Predict position using smoother
                    track = self._tracks[tid]
                    if self.smoothing_method == "kalman":
                        track["bbox"] = track["smoother"].predict()
            return self._get_visible_tracks()

        # Extract bboxes from detections
        det_bboxes = np.array([d.bbox for d in detections])
        track_ids = list(self._tracks.keys())

        if not track_ids:
            # No existing tracks — create new ones for all detections
            for det in detections:
                self._create_track(det, now)
            return self._get_visible_tracks()

        # Compute IoU matrix between existing tracks and new detections
        track_bboxes = np.array([self._tracks[tid]["bbox"] for tid in track_ids])
        iou_matrix = self._compute_iou_matrix(track_bboxes, det_bboxes)

        # Greedy matching (Hungarian would be better for many objects)
        matched_tracks = set()
        matched_dets = set()

        # Match by highest IoU first
        while True:
            if iou_matrix.size == 0:
                break
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_idx]

            if max_iou < self.iou_threshold:
                break

            track_idx, det_idx = max_idx

            if track_idx not in matched_tracks and det_idx not in matched_dets:
                tid = track_ids[track_idx]
                det = detections[det_idx]
                self._update_track(tid, det, now)
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)

            # Zero out matched row and column
            iou_matrix[track_idx, :] = 0
            iou_matrix[:, det_idx] = 0

        # Unmatched tracks — increment missing
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self._tracks[tid]["frames_missing"] += 1
                if self._tracks[tid]["frames_missing"] > self.max_missing_frames:
                    del self._tracks[tid]
                elif self.smoothing_method == "kalman":
                    self._tracks[tid]["bbox"] = self._tracks[tid]["smoother"].predict()

        # Unmatched detections — create new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(det, now)

        return self._get_visible_tracks()

    def reset(self):
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 0

    # ── Private ──────────────────────────────────────────────

    def _create_track(self, det, now: float):
        """Create a new track from a detection."""
        tid = self._next_id
        self._next_id += 1

        if self.smoothing_method == "kalman":
            smoother = KalmanBboxTracker()
            smoothed = smoother.update(det.bbox)
        else:
            smoother = EMABboxSmoother(alpha=self.ema_alpha)
            smoothed = smoother.smooth(det.bbox)

        self._tracks[tid] = {
            "smoother": smoother,
            "bbox": smoothed,
            "raw_bbox": det.bbox.copy(),
            "class_id": det.class_id,
            "class_name": det.class_name,
            "confidence": det.confidence,
            "mask": det.mask,
            "frames_seen": 1,
            "frames_missing": 0,
            "last_seen": now,
        }

    def _update_track(self, tid: int, det, now: float):
        """Update an existing track with a new detection."""
        track = self._tracks[tid]

        if self.smoothing_method == "kalman":
            track["bbox"] = track["smoother"].update(det.bbox)
        else:
            track["bbox"] = track["smoother"].smooth(det.bbox)

        track["raw_bbox"] = det.bbox.copy()
        track["confidence"] = det.confidence
        track["mask"] = det.mask
        track["frames_seen"] += 1
        track["frames_missing"] = 0
        track["last_seen"] = now

    def _get_visible_tracks(self) -> list[TrackedObject]:
        """Return tracks that have been seen enough frames to be considered visible."""
        result = []
        for tid, track in self._tracks.items():
            is_visible = (
                track["frames_seen"] >= self.min_visible_frames
                and track["frames_missing"] <= 3  # Allow brief occlusions
            )
            result.append(
                TrackedObject(
                    track_id=tid,
                    class_id=track["class_id"],
                    class_name=track["class_name"],
                    bbox=track["bbox"],
                    raw_bbox=track["raw_bbox"],
                    confidence=track["confidence"],
                    mask=track["mask"],
                    frames_seen=track["frames_seen"],
                    frames_missing=track["frames_missing"],
                    last_seen=track["last_seen"],
                    is_visible=is_visible,
                )
            )
        return result

    @staticmethod
    def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of bboxes [x1,y1,x2,y2]."""
        n = len(boxes_a)
        m = len(boxes_b)
        iou = np.zeros((n, m), dtype=np.float64)

        for i in range(n):
            for j in range(m):
                # Intersection
                x1 = max(boxes_a[i, 0], boxes_b[j, 0])
                y1 = max(boxes_a[i, 1], boxes_b[j, 1])
                x2 = min(boxes_a[i, 2], boxes_b[j, 2])
                y2 = min(boxes_a[i, 3], boxes_b[j, 3])

                inter = max(0, x2 - x1) * max(0, y2 - y1)

                # Union
                area_a = (boxes_a[i, 2] - boxes_a[i, 0]) * (
                    boxes_a[i, 3] - boxes_a[i, 1]
                )
                area_b = (boxes_b[j, 2] - boxes_b[j, 0]) * (
                    boxes_b[j, 3] - boxes_b[j, 1]
                )
                union = area_a + area_b - inter

                iou[i, j] = inter / union if union > 0 else 0

        return iou
