"""
Video output for Replace Mode — serves ML-processed frames as a local
MJPEG HTTP stream that OBS reads via Media Source or Browser Source.

No special drivers or platform-specific packages required.
"""

import threading
import time
import logging
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import numpy as np
import cv2

logger = logging.getLogger("adstream.virtual_cam")


class _StreamHandler(BaseHTTPRequestHandler):
    """Serves the latest JPEG frame as a multipart MJPEG stream."""

    def do_GET(self):
        if self.path == "/stream":
            self._serve_mjpeg()
        elif self.path == "/snapshot":
            self._serve_snapshot()
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        server: MJPEGStreamServer = self.server
        while server.running:
            jpeg = server.get_jpeg()
            if jpeg is None:
                time.sleep(0.01)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
            time.sleep(1.0 / server.target_fps)

    def _serve_snapshot(self):
        server: MJPEGStreamServer = self.server
        jpeg = server.get_jpeg()
        if jpeg is None:
            self.send_response(503)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def log_message(self, format, *args):
        pass


class MJPEGStreamServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server that holds the latest JPEG frame."""

    daemon_threads = True

    def __init__(self, host: str, port: int, fps: int = 30, jpeg_quality: int = 85):
        super().__init__((host, port), _StreamHandler)
        self.target_fps = fps
        self.jpeg_quality = jpeg_quality
        self.running = False
        self._lock = threading.Lock()
        self._jpeg_buffer: Optional[bytes] = None

    def update_frame(self, frame: np.ndarray):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            with self._lock:
                self._jpeg_buffer = jpeg.tobytes()

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg_buffer


class VideoStreamOutput:
    """
    Serves ML-processed frames as a local MJPEG HTTP stream.

    OBS setup: Add a "Media Source" pointing to http://localhost:{port}/stream
    or a "Browser Source" with the same URL.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        host: str = "127.0.0.1",
        port: int = 8765,
        jpeg_quality: int = 85,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.host = host
        self.port = port
        self._server: Optional[MJPEGStreamServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._jpeg_quality = jpeg_quality

    def start(self) -> bool:
        try:
            self._server = MJPEGStreamServer(
                self.host,
                self.port,
                fps=self.fps,
                jpeg_quality=self._jpeg_quality,
            )
            self._server.running = True
            self._running = True
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._thread.start()
            logger.info(
                "MJPEG stream started at http://%s:%d/stream (%dx%d@%dfps)",
                self.host,
                self.port,
                self.width,
                self.height,
                self.fps,
            )
            return True
        except OSError as e:
            logger.error("Failed to start stream server on port %d: %s", self.port, e)
            return False

    def stop(self):
        self._running = False
        if self._server:
            self._server.running = False
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("MJPEG stream stopped")

    def send_frame(self, frame: np.ndarray):
        if not self._running or self._server is None:
            return
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self._server.update_frame(frame)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stream_url(self) -> str:
        return f"http://{self.host}:{self.port}/stream"

    @property
    def snapshot_url(self) -> str:
        return f"http://{self.host}:{self.port}/snapshot"


class ReplaceModeProcessor:
    """
    Full replacement pipeline: captures frames, detects objects,
    composites replacements, and outputs to MJPEG stream.
    """

    def __init__(
        self,
        detector,
        tracker,
        replacement_manager,
        frame_capture,
        video_output: VideoStreamOutput,
    ):
        self._detector = detector
        self._tracker = tracker
        self._replacement_mgr = replacement_manager
        self._frame_capture = frame_capture
        self._video_output = video_output
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("Replace mode processor started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Replace mode processor stopped")

    def _process_loop(self):
        last_result = None

        while self._running:
            frame = self._frame_capture.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            self._detector.submit_frame(frame)
            result = self._detector.get_result()

            if result is not None:
                last_result = result

            output_frame = frame.copy()

            if last_result is not None:
                tracked = self._tracker.update(last_result.detections)

                for obj in tracked:
                    if not obj.is_visible:
                        continue

                    asset = self._replacement_mgr.get_asset(obj.class_id)
                    if asset is None:
                        continue

                    output_frame = self._replacement_mgr.replacer.composite_frame(
                        output_frame,
                        asset,
                        obj.bbox,
                        mask=obj.mask,
                    )

            self._video_output.send_frame(output_frame)
