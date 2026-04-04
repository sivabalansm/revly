"""
Ring buffer for live stream delay.
Stores N minutes of video frames indexed by timestamp.
Supports segment extraction (as MP4) and frame replacement.
"""

import os
import time
import threading
import tempfile
import logging
from typing import Optional
from collections import OrderedDict

import numpy as np
import cv2

logger = logging.getLogger("adstream.stream_buffer")


class StreamBuffer:
    """
    Time-indexed ring buffer of video frames.
    Frames are stored with their capture timestamp.
    Supports extracting segments as MP4 and replacing frame ranges.
    """

    def __init__(
        self,
        buffer_duration: float = 180.0,
        fps: float = 30.0,
        temp_dir: Optional[str] = None,
    ):
        self.buffer_duration = buffer_duration
        self.fps = fps
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="adstream_buf_")
        os.makedirs(self.temp_dir, exist_ok=True)

        self._frames: OrderedDict[float, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()
        self._start_time: Optional[float] = None
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def duration(self) -> float:
        with self._lock:
            if len(self._frames) < 2:
                return 0.0
            timestamps = list(self._frames.keys())
            return timestamps[-1] - timestamps[0]

    @property
    def oldest_timestamp(self) -> float:
        with self._lock:
            if not self._frames:
                return 0.0
            return next(iter(self._frames))

    @property
    def newest_timestamp(self) -> float:
        with self._lock:
            if not self._frames:
                return 0.0
            return next(reversed(self._frames))

    def push_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        ts = timestamp or time.time()

        if self._start_time is None:
            self._start_time = ts

        with self._lock:
            self._frames[ts] = frame
            self._frame_count += 1
            self._evict_old(ts)

    def get_frame_at(self, target_ts: float) -> Optional[np.ndarray]:
        with self._lock:
            if not self._frames:
                return None

            timestamps = list(self._frames.keys())
            closest = min(timestamps, key=lambda t: abs(t - target_ts))

            if abs(closest - target_ts) > 1.0 / self.fps:
                return None
            return self._frames[closest].copy()

    def get_delayed_frame(self, delay: float) -> Optional[np.ndarray]:
        target_ts = time.time() - delay
        return self.get_frame_at(target_ts)

    def extract_segment_as_mp4(
        self,
        start_ts: float,
        end_ts: float,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        with self._lock:
            segment_frames = [
                (ts, self._frames[ts])
                for ts in self._frames
                if start_ts <= ts <= end_ts
            ]

        if not segment_frames:
            logger.warning("No frames in range [%.1f, %.1f]", start_ts, end_ts)
            return None

        if output_path is None:
            output_path = os.path.join(self.temp_dir, f"segment_{int(start_ts)}.mp4")

        h, w = segment_frames[0][1].shape[:2]
        output_fps = max(24.0, self.fps)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))

        if output_fps > self.fps and len(segment_frames) > 1:
            target_count = int(output_fps * (end_ts - start_ts))
            for i in range(target_count):
                t = i / target_count * (len(segment_frames) - 1)
                idx = min(int(t), len(segment_frames) - 1)
                writer.write(segment_frames[idx][1])
        else:
            for ts, frame in segment_frames:
                writer.write(frame)

        writer.release()

        duration = end_ts - start_ts
        logger.info(
            "Extracted %d frames (%.1fs) to %s",
            len(segment_frames),
            duration,
            output_path,
        )
        return output_path

    def replace_segment_from_mp4(
        self,
        video_path: str,
        start_ts: float,
    ) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return 0

        video_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        frame_interval = 1.0 / video_fps
        replaced = 0
        current_ts = start_ts

        with self._lock:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                closest_ts = self._find_closest_ts(current_ts)
                if closest_ts is not None:
                    self._frames[closest_ts] = frame
                    replaced += 1

                current_ts += frame_interval

        cap.release()
        logger.info("Replaced %d frames starting at %.1f", replaced, start_ts)
        return replaced

    def replace_frames(
        self,
        frames: list[np.ndarray],
        start_ts: float,
    ) -> int:
        frame_interval = 1.0 / self.fps
        replaced = 0

        with self._lock:
            for i, frame in enumerate(frames):
                target_ts = start_ts + i * frame_interval
                closest_ts = self._find_closest_ts(target_ts)
                if closest_ts is not None:
                    self._frames[closest_ts] = frame
                    replaced += 1

        return replaced

    def get_timestamps_in_range(self, start_ts: float, end_ts: float) -> list[float]:
        with self._lock:
            return [ts for ts in self._frames if start_ts <= ts <= end_ts]

    def _evict_old(self, current_ts: float):
        cutoff = current_ts - self.buffer_duration
        while self._frames and next(iter(self._frames)) < cutoff:
            self._frames.popitem(last=False)

    def _find_closest_ts(self, target: float) -> Optional[float]:
        if not self._frames:
            return None
        timestamps = list(self._frames.keys())
        closest = min(timestamps, key=lambda t: abs(t - target))
        if abs(closest - target) < 2.0 / self.fps:
            return closest
        return None
