#!/usr/bin/env python3
"""
Standalone demo — run from terminal, no OBS required.

Usage:
    python3 demo.py                    # webcam + detection overlay
    python3 demo.py --ad coca_cola.png # webcam + object replacement with your ad image
    python3 demo.py --stream           # also starts MJPEG stream at localhost:8765
    python3 demo.py --camera 1         # use camera index 1
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from lib.detector import DetectionEngine
from lib.tracker import ObjectTracker
from lib.replacer import ObjectReplacer, ReplacementAsset, ReplacementManager
from lib.timing import AdTimingEngine
from lib.config import COCO_REPLACEABLE_CLASSES, load_class_mapping, ASSETS_DIR


def generate_placeholder_ad(width=200, height=300):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:, :, 0] = 0
    img[:, :, 1] = 0
    img[:, :, 2] = 220
    img[:, :, 3] = 200

    cv2.rectangle(img, (4, 4), (width - 4, height - 4), (255, 255, 255, 255), 2)
    cv2.putText(
        img,
        "YOUR",
        (width // 2 - 40, height // 2 - 20),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (255, 255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        "AD",
        (width // 2 - 25, height // 2 + 20),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (255, 255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        "HERE",
        (width // 2 - 40, height // 2 + 55),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (255, 255, 255, 255),
        2,
    )

    path = os.path.join(os.path.dirname(__file__), "assets", "_placeholder_ad.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    return path


def draw_hud(frame, detections, timing_result, inference_ms, fps):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (340, 110), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"FPS: {fps:.0f}  Inference: {inference_ms:.0f}ms",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    cv2.putText(
        frame,
        f"Detections: {len(detections)}",
        (10, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    timing_status = "SHOWING AD" if timing_result.get("show_ad") else "waiting"
    opacity = timing_result.get("opacity", 0)
    reason = timing_result.get("reason", "")
    color = (0, 0, 255) if timing_result.get("show_ad") else (200, 200, 200)
    cv2.putText(
        frame,
        f"Ad: {timing_status} (opacity={opacity:.2f})",
        (10, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )
    if reason:
        cv2.putText(
            frame,
            f"Reason: {reason}",
            (10, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )

    cv2.putText(
        frame,
        "Press Q to quit | F to force ad | R to reset",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (150, 150, 150),
        1,
    )


def draw_detection_box(frame, obj, show_label=True):
    x1, y1, x2, y2 = map(int, obj.bbox)
    color = (0, 255, 0) if obj.is_visible else (100, 100, 100)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if show_label:
        label = f"{obj.class_name} {obj.confidence:.0%} #{obj.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )


def main():
    parser = argparse.ArgumentParser(description="AdStream standalone demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--ad", type=str, default=None, help="Path to ad image (PNG/JPG)"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Start MJPEG stream on localhost:8765"
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model")
    parser.add_argument(
        "--seg", action="store_true", help="Use YOLO segmentation model"
    )
    parser.add_argument(
        "--sam", action="store_true", help="Use FastSAM for precise mask fitting"
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    print("=" * 50)
    print("  AdStream — Standalone Demo")
    print("=" * 50)

    ad_path = args.ad
    cocacola_default = os.path.join(os.path.dirname(__file__), "assets", "cocacola.png")
    if ad_path is None:
        if os.path.exists(cocacola_default):
            ad_path = cocacola_default
        else:
            print("No --ad image provided, using placeholder.")
            ad_path = generate_placeholder_ad()
    elif not os.path.exists(ad_path):
        print(f"Ad image not found: {ad_path}")
        sys.exit(1)
    print(f"Ad image: {ad_path}")

    class_mapping = load_class_mapping()
    if class_mapping:
        target_classes = [int(k) for k in class_mapping.keys()]
        print(
            f"Targeting from class_mapping.json: {[COCO_REPLACEABLE_CLASSES.get(c, c) for c in target_classes]}"
        )
    else:
        target_classes = [41, 39]
        print(f"No class_mapping.json found, defaulting to: cup, bottle")

    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_w}x{actual_h}")

    print(f"Loading {args.model}{'(seg)' if args.seg else ''}...")
    detector = DetectionEngine(
        model_name=args.model,
        use_segmentation=args.seg,
        confidence_threshold=args.conf,
        target_classes=target_classes,
    )
    detector.start()

    tracker = ObjectTracker(smoothing_method="kalman", min_visible_frames=2)
    replacer = ObjectReplacer(blend_mode="alpha", color_adapt=True, edge_feather_px=5)
    timing = AdTimingEngine(
        min_ad_interval=10.0, min_ad_duration=3.0, max_ad_duration=15.0
    )

    sam_refiner = None
    if args.sam:
        from lib.detector import FastSAMRefiner

        print("Loading FastSAM-s for mask refinement...")
        sam_refiner = FastSAMRefiner(model_name="FastSAM-s.pt")

    replacement_mgr = ReplacementManager()
    if class_mapping:
        replacement_mgr.load_from_mapping(class_mapping, ASSETS_DIR)
    else:
        for cls_id in target_classes:
            replacement_mgr.set_asset(cls_id, ad_path)

    stream_output = None
    if args.stream:
        from lib.virtual_cam import VideoStreamOutput

        stream_output = VideoStreamOutput(width=actual_w, height=actual_h, fps=15)
        if stream_output.start():
            print(f"MJPEG stream: {stream_output.stream_url}")
        else:
            stream_output = None

    print()
    print("Waiting for model to load (first inference is slow)...")
    print("Press Q to quit | F to force ad | R to reset tracker")
    print()

    fps_counter = 0
    fps_timer = time.time()
    fps_display = 0.0
    inference_ms = 0.0
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detector.submit_frame(frame)
        result = detector.get_result()
        if result is not None:
            if sam_refiner and result.detections:
                result.detections = sam_refiner.refine(frame, result.detections)
            last_result = result
            inference_ms = result.inference_time_ms

        timing_result = timing.update(frame=frame, audio_level_db=None)

        display_frame = frame.copy()

        if last_result is not None:
            tracked = tracker.update(last_result.detections)

            for obj in tracked:
                draw_detection_box(display_frame, obj)

                if obj.is_visible:
                    asset = replacement_mgr.get_asset(obj.class_id)
                    if asset is not None:
                        display_frame = replacer.composite_frame(
                            display_frame,
                            asset,
                            obj.bbox,
                            mask=obj.mask,
                        )

        fps_counter += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_display = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer = now

        tracked_count = len(tracker.update([])) if last_result is None else 0
        draw_hud(
            display_frame,
            last_result.detections if last_result else [],
            timing_result,
            inference_ms,
            fps_display,
        )

        cv2.imshow("AdStream Demo", display_frame)

        if stream_output and stream_output.is_running:
            stream_output.send_frame(display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            timing.force_show()
            print("[Manual] Ad triggered")
        elif key == ord("r"):
            tracker.reset()
            print("[Manual] Tracker reset")

    print("Shutting down...")
    detector.stop()
    if stream_output:
        stream_output.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
