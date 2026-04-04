#!/usr/bin/env python3
"""
Live stream demo with 3-minute delay buffer + Wan 2.7 AI video editing.

Usage:
    export DASHSCOPE_API_KEY=sk-xxx
    python3 demo_live.py                        # camera + delay buffer + YOLO detection
    python3 demo_live.py --delay 180 --port 8765  # 3 min delay, stream on :8765
    python3 demo_live.py --dry-run               # skip Wan API calls, test buffer only
"""

import sys
import os
import argparse
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import cv2
import numpy as np

from lib.detector import DetectionEngine
from lib.tracker import ObjectTracker
from lib.stream_buffer import StreamBuffer
from lib.ad_pipeline import LiveAdPipeline
from lib.wan_client import WanClient
from lib.delayed_output import DelayedStreamOutput
from lib.config import load_class_mapping, ASSETS_DIR, COCO_REPLACEABLE_CLASSES


def draw_live_hud(frame, delay, buffer_duration, active_jobs, fps, detection_names):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (420, 130), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"LIVE DELAY: {delay:.0f}s | Buffer: {buffer_duration:.0f}s",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 200, 255),
        1,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.0f} | Active Wan jobs: {active_jobs}",
        (10, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    if detection_names:
        det_text = ", ".join(detection_names[:3])
        cv2.putText(
            frame,
            f"Detected: {det_text}",
            (10, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
    else:
        cv2.putText(
            frame,
            "Detecting...",
            (10, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )

    cv2.putText(
        frame,
        "Q=quit | T=trigger ad manually",
        (10, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (150, 150, 150),
        1,
    )


def main():
    parser = argparse.ArgumentParser(description="AdStream live delay demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--delay", type=float, default=480.0, help="Stream delay in seconds"
    )
    parser.add_argument("--port", type=int, default=8765, help="MJPEG output port")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--product", type=str, default="cup")
    parser.add_argument("--branded", type=str, default="Coca-Cola")
    parser.add_argument("--item", type=str, default="can")
    parser.add_argument("--segment-duration", type=float, default=5.0)
    parser.add_argument("--region", type=str, default="singapore")
    parser.add_argument(
        "--dry-run", action="store_true", help="Skip Wan API, test buffer only"
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    print("=" * 55)
    print("  AdStream — Live Delay + Wan 2.7 Video Editing")
    print("=" * 55)
    print(f"  Delay: {args.delay}s | Segment: {args.segment_duration}s")
    print(f"  Product: {args.product} → {args.branded} {args.item}")
    print(f"  Dry run: {args.dry_run}")
    print()

    class_mapping = load_class_mapping()
    target_classes = (
        [int(k) for k in class_mapping.keys()] if class_mapping else [41, 39]
    )
    target_names = [COCO_REPLACEABLE_CLASSES.get(c, str(c)) for c in target_classes]
    print(f"  Targeting: {target_names}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(
        f"  Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {actual_fps:.0f}fps"
    )

    buffer_duration = args.delay + 60.0
    stream_buffer = StreamBuffer(buffer_duration=buffer_duration, fps=actual_fps)
    print(f"  Ring buffer: {buffer_duration:.0f}s capacity")

    wan_client = WanClient(region=args.region, resolution="720P")
    if not args.dry_run and not wan_client.api_key:
        print("WARNING: DASHSCOPE_API_KEY not set. Wan 2.7 calls will fail.")
        print("  Set it with: export DASHSCOPE_API_KEY=sk-xxx")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = LiveAdPipeline(
        wan_client=wan_client,
        stream_buffer=stream_buffer,
        product=args.product,
        branded=args.branded,
        item=args.item,
        segment_duration=args.segment_duration,
        temp_dir=output_dir,
    )
    print(f'  Prompt: "{pipeline.prompt}"')

    delayed_output = DelayedStreamOutput(
        stream_buffer=stream_buffer,
        delay=args.delay,
        output_fps=actual_fps,
    )
    delayed_output.start(mjpeg_port=args.port)

    detector = DetectionEngine(
        model_name="yolov8n.pt",
        confidence_threshold=args.conf,
        target_classes=target_classes,
    )
    detector.start()
    tracker = ObjectTracker(smoothing_method="kalman", min_visible_frames=3)

    print()
    print(f"  Delayed stream: http://localhost:{args.port}/stream")
    print(f"  (Will start showing frames after {args.delay:.0f}s buffer fills)")
    print()
    print("  Waiting for model + buffer to fill...")

    fps_counter = 0
    fps_timer = time.time()
    fps_display = 0.0
    last_result = None
    last_trigger_time = 0.0
    detection_names = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            stream_buffer.push_frame(frame, timestamp=now)

            detector.submit_frame(frame)
            result = detector.get_result()
            if result is not None:
                last_result = result
                detection_names = [
                    f"{d.class_name}({d.confidence:.0%})" for d in result.detections
                ]

            if last_result and last_result.detections:
                tracker.update(last_result.detections)

            fps_counter += 1
            if now - fps_timer >= 1.0:
                fps_display = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            display = frame.copy()
            active_jobs = len(pipeline.get_active_jobs())
            draw_live_hud(
                display,
                args.delay,
                stream_buffer.duration,
                active_jobs,
                fps_display,
                detection_names,
            )

            delayed_frame = delayed_output.get_frame()
            if delayed_frame is not None:
                dh, dw = delayed_frame.shape[:2]
                preview_w = 320
                preview_h = int(dh * preview_w / dw)
                preview = cv2.resize(delayed_frame, (preview_w, preview_h))

                fh, fw = display.shape[:2]
                x_off = fw - preview_w - 10
                y_off = fh - preview_h - 10
                display[y_off : y_off + preview_h, x_off : x_off + preview_w] = preview

                cv2.rectangle(
                    display,
                    (x_off - 1, y_off - 20),
                    (x_off + preview_w + 1, y_off),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    display,
                    f"DELAYED OUTPUT (-{args.delay:.0f}s)",
                    (x_off + 5, y_off - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 200, 255),
                    1,
                )

            cv2.imshow("AdStream LIVE", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                job = pipeline.trigger_ad_replacement(detection_ts=now)
                if job:
                    print(f"  [MANUAL] Ad job triggered: {job.job_id}")

            pipeline.cleanup_old_jobs()

    except KeyboardInterrupt:
        print("\nShutting down...")

    detector.stop()
    delayed_output.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
