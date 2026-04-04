#!/usr/bin/env python3
"""
Detection Bridge — captures camera frames, runs YOLOv8 detection,
and sends bounding box results to the Node.js overlay server via Socket.IO.

Usage:
    python detection_bridge.py --token YOUR_TOKEN
    python detection_bridge.py                      # auto-fetches token from /api/token
"""

import argparse
import json
import os
import shutil
import signal
import sys
import time
import logging

import cv2
import numpy as np
import socketio

# Add project root to path so we can import lib modules
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from lib.detector import DetectionEngine
from lib.tracker import ObjectTracker
from lib.config import DEFAULT_CONFIG, load_class_mapping, ASSETS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("detection_bridge")

SERVER_URL = "http://localhost:3000"
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "frontend", "public", "uploads")
TARGET_FPS = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS


def fetch_token_from_server() -> str:
    """Auto-fetch auth token from the Node.js server."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{SERVER_URL}/api/token")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("token", "")
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
        logger.error("Failed to fetch token from server: %s", e)
        sys.exit(1)


def copy_sponsor_images(class_mapping: dict):
    """Copy sponsor images from assets/ to frontend/public/uploads/ on startup."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    copied = set()
    for class_id, image_file in class_mapping.items():
        if image_file in copied:
            continue
        src = os.path.join(ASSETS_DIR, image_file)
        dst = os.path.join(UPLOADS_DIR, image_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied.add(image_file)
            logger.info("Copied sponsor image: %s", image_file)
        else:
            logger.warning("Sponsor image not found: %s", src)


def build_detection_payload(tracked_objects, class_mapping: dict) -> dict:
    """Convert tracked objects to the Socket.IO event format."""
    items = []
    for obj in tracked_objects:
        if not obj.is_visible:
            continue

        x1, y1, x2, y2 = obj.bbox
        w = x2 - x1
        h = y2 - y1

        sponsor_image = class_mapping.get(str(obj.class_id))
        sponsor_url = f"/uploads/{sponsor_image}" if sponsor_image else None

        item = {
            "id": f"track-{obj.track_id}",
            "label": obj.class_name,
            "class_id": obj.class_id,
            "bbox": {
                "x": round(float(x1)),
                "y": round(float(y1)),
                "w": round(float(w)),
                "h": round(float(h)),
            },
            "confidence": round(obj.confidence, 3),
        }
        if sponsor_url:
            item["sponsorImageUrl"] = sponsor_url

        items.append(item)

    return {"type": "detection_update", "detections": items}


def main():
    parser = argparse.ArgumentParser(description="Detection Bridge — camera to Socket.IO")
    parser.add_argument("--token", type=str, default=None, help="Auth token for Socket.IO server")
    parser.add_argument("--camera", type=int, default=DEFAULT_CONFIG["camera_index"], help="Camera index")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIG["confidence_threshold"])
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"])
    args = parser.parse_args()

    # Get auth token
    token = args.token
    if not token:
        logger.info("No --token provided, fetching from server...")
        token = fetch_token_from_server()

    # Load class mapping and copy sponsor images
    class_mapping = load_class_mapping()
    logger.info("Class mapping: %s", class_mapping)
    copy_sponsor_images(class_mapping)

    # Set up Socket.IO client
    sio = socketio.Client(reconnection=True, reconnection_attempts=0, reconnection_delay=2)

    connected = False

    @sio.event
    def connect():
        nonlocal connected
        connected = True
        logger.info("Connected to server at %s", SERVER_URL)

    @sio.event
    def disconnect():
        nonlocal connected
        connected = False
        logger.warning("Disconnected from server")

    @sio.event
    def connect_error(data):
        logger.error("Connection error: %s", data)

    # Connect to server
    try:
        sio.connect(SERVER_URL, auth={"token": token})
    except socketio.exceptions.ConnectionError as e:
        logger.error("Cannot connect to server at %s: %s", SERVER_URL, e)
        logger.error("Is the Node.js server running?")
        sys.exit(1)

    # Open camera
    logger.info("Opening camera %d...", args.camera)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Cannot open camera %d", args.camera)
        sio.disconnect()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CONFIG["capture_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CONFIG["capture_height"])
    logger.info(
        "Camera opened: %dx%d",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # Set up detection engine + tracker
    target_classes = DEFAULT_CONFIG["target_classes"]
    logger.info("Target classes: %s", target_classes)

    engine = DetectionEngine(
        model_name=args.model,
        confidence_threshold=args.confidence,
        target_classes=target_classes,
    )
    engine.start()

    tracker = ObjectTracker(smoothing_method="kalman", min_visible_frames=2)

    logger.info("Detection engine started, waiting for model load...")

    # Graceful shutdown
    running = True

    def shutdown(signum, frame):
        nonlocal running
        running = False
        logger.info("Shutting down...")

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Main loop
    frame_count = 0
    detection_count = 0
    fps_timer = time.perf_counter()
    fps_frames = 0

    logger.info("Starting detection loop at ~%d FPS...", TARGET_FPS)

    try:
        while running:
            loop_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            # Submit frame to detector
            engine.submit_frame(frame)
            frame_count += 1

            # Check for results and run through tracker
            result = engine.get_result()
            if result is not None:
                tracked = tracker.update(result.detections)
                visible = [obj for obj in tracked if obj.is_visible]

                if visible:
                    detection_count += len(visible)
                    payload = build_detection_payload(visible, class_mapping)
                    if connected:
                        try:
                            sio.emit("detection_update", payload)
                        except Exception as e:
                            logger.error("Failed to send detection: %s", e)
                else:
                    if connected:
                        try:
                            sio.emit("detection_update", {"type": "detection_update", "detections": []})
                        except Exception as e:
                            logger.error("Failed to send empty update: %s", e)

            # FPS counter
            fps_frames += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 2.0:
                fps = fps_frames / elapsed
                logger.info(
                    "FPS: %.1f | Frames: %d | Detections: %d | Inference: %.1fms",
                    fps,
                    frame_count,
                    detection_count,
                    engine.avg_inference_ms,
                )
                fps_frames = 0
                fps_timer = time.perf_counter()

            # Throttle to target FPS
            elapsed = time.perf_counter() - loop_start
            sleep_time = FRAME_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        logger.info("Cleaning up...")
        engine.stop()
        cap.release()
        if sio.connected:
            sio.disconnect()
        logger.info(
            "Done. Processed %d frames, %d total detections.", frame_count, detection_count
        )


if __name__ == "__main__":
    main()
