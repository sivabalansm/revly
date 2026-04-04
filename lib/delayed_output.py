"""
Delayed stream output — reads from the ring buffer at T minus delay
and outputs to the MJPEG stream or OBS overlay.
"""

import time
import threading
import logging
from typing import Optional

import numpy as np
import cv2

from lib.stream_buffer import StreamBuffer

logger = logging.getLogger("adstream.delayed_output")


class DelayedStreamOutput:
    def __init__(
        self,
        stream_buffer: StreamBuffer,
        delay: float = 180.0,
        output_fps: float = 30.0,
    ):
        self.stream_buffer = stream_buffer
        self.delay = delay
        self.output_fps = output_fps

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._mjpeg_server = None
        self._on_frame_callbacks: list = []

    def start(self, mjpeg_port: Optional[int] = None):
        if self._running:
            return

        if mjpeg_port:
            from lib.virtual_cam import VideoStreamOutput

            self._mjpeg_server = VideoStreamOutput(
                width=1920,
                height=1080,
                fps=int(self.output_fps),
                port=mjpeg_port,
            )
            if self._mjpeg_server.start():
                logger.info("Delayed output MJPEG: %s", self._mjpeg_server.stream_url)
            else:
                self._mjpeg_server = None

        self._running = True
        self._thread = threading.Thread(target=self._output_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Delayed output started (delay=%.0fs, fps=%.0f)",
            self.delay,
            self.output_fps,
        )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._mjpeg_server:
            self._mjpeg_server.stop()
            self._mjpeg_server = None

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def on_frame(self, callback):
        self._on_frame_callbacks.append(callback)

    @property
    def is_outputting(self) -> bool:
        return self._running and self._latest_frame is not None

    @property
    def current_output_timestamp(self) -> float:
        return time.time() - self.delay

    def _output_loop(self):
        frame_interval = 1.0 / self.output_fps

        while self._running:
            target_ts = time.time() - self.delay
            frame = self.stream_buffer.get_frame_at(target_ts)

            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
                self._frame_count += 1

                if self._mjpeg_server:
                    self._mjpeg_server.send_frame(frame)

                for cb in self._on_frame_callbacks:
                    try:
                        cb(frame, target_ts)
                    except Exception as e:
                        logger.error("Frame callback error: %s", e)

            time.sleep(frame_interval)
