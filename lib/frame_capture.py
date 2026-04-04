"""
Frame Capture Module

Captures video frames from camera or OBS source for ML processing.
Supports direct camera capture (cv2.VideoCapture) and screen region capture (mss).
"""

import threading
import time
import logging
from typing import Optional

import numpy as np
import cv2

logger = logging.getLogger("adstream.frame_capture")


class CameraCapture:
    """
    Captures frames from a camera via OpenCV.
    Runs in a background thread for non-blocking access.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        capture_interval: float = 0.033,  # ~30fps capture
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.capture_interval = capture_interval

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._frame_ready = threading.Event()

    def start(self) -> bool:
        """Start frame capture. Returns True if camera opened successfully."""
        self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            logger.error("Failed to open camera %d", self.camera_index)
            return False

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera %d opened: %dx%d", self.camera_index, actual_w, actual_h)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop frame capture and release camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera capture stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame (non-blocking). Returns None if not available."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None

    def wait_for_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Wait for a new frame (blocking with timeout)."""
        self._frame_ready.clear()
        if self._frame_ready.wait(timeout=timeout):
            return self.get_frame()
        return None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def resolution(self) -> tuple[int, int]:
        if self._cap and self._cap.isOpened():
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        return (self.width, self.height)

    def _capture_loop(self):
        """Background frame capture loop."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                self._frame_count += 1
                self._frame_ready.set()

            time.sleep(self.capture_interval)


class ScreenRegionCapture:
    """
    Captures a screen region (e.g., OBS preview window).
    Uses mss for fast screen capture.
    """

    def __init__(
        self,
        region: Optional[dict] = None,
        monitor_index: int = 0,
        capture_interval: float = 0.1,
    ):
        """
        Args:
            region: {"left": x, "top": y, "width": w, "height": h} or None for full monitor
            monitor_index: Which monitor to capture
            capture_interval: Time between captures in seconds
        """
        self.region = region
        self.monitor_index = monitor_index
        self.capture_interval = capture_interval

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count = 0

    def start(self) -> bool:
        """Start screen capture."""
        try:
            import mss  # noqa: F401
        except ImportError:
            logger.error("mss not installed. Run: pip install mss")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Screen capture started (region=%s)", self.region)
        return True

    def stop(self):
        """Stop screen capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("Screen capture stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest captured frame."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None

    def set_region(self, left: int, top: int, width: int, height: int):
        """Update the capture region."""
        self.region = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

    def _capture_loop(self):
        """Background screen capture loop."""
        import mss

        with mss.mss() as sct:
            while self._running:
                try:
                    if self.region:
                        monitor = self.region
                    else:
                        monitor = sct.monitors[self.monitor_index + 1]

                    screenshot = sct.grab(monitor)
                    frame = np.array(screenshot)

                    # mss captures BGRA, convert to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    with self._lock:
                        self._latest_frame = frame
                    self._frame_count += 1

                except Exception as e:
                    logger.error("Screen capture error: %s", e)

                time.sleep(self.capture_interval)


class FrameCaptureManager:
    """
    Unified frame capture interface.
    Supports camera, screen region, or manual frame feeding.
    """

    def __init__(self, mode: str = "camera", **kwargs):
        """
        Args:
            mode: "camera", "screen", or "manual"
            **kwargs: Passed to the capture backend
        """
        self.mode = mode
        self._backend = None

        if mode == "camera":
            self._backend = CameraCapture(**kwargs)
        elif mode == "screen":
            self._backend = ScreenRegionCapture(**kwargs)
        elif mode == "manual":
            self._lock = threading.Lock()
            self._manual_frame: Optional[np.ndarray] = None
        else:
            raise ValueError(f"Unknown capture mode: {mode}")

    def start(self) -> bool:
        """Start capture (no-op for manual mode)."""
        if self.mode == "manual":
            return True
        return self._backend.start()

    def stop(self):
        """Stop capture."""
        if self.mode != "manual" and self._backend:
            self._backend.stop()

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame."""
        if self.mode == "manual":
            with self._lock:
                return (
                    self._manual_frame.copy()
                    if self._manual_frame is not None
                    else None
                )
        return self._backend.get_frame()

    def feed_frame(self, frame: np.ndarray):
        """Manually feed a frame (only for manual mode)."""
        if self.mode != "manual":
            raise RuntimeError("feed_frame only works in manual mode")
        with self._lock:
            self._manual_frame = frame.copy()
