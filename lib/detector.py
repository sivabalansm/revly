"""
YOLOv8 Detection & Segmentation Engine

Runs inference in a background thread with queue-based result passing.
Supports both detection (bounding boxes) and segmentation (pixel masks).
All OBS API calls are forbidden in this module — pure ML only.
"""

import threading
import queue
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("adstream.detector")


@dataclass
class Detection:
    """Single object detection result."""

    bbox: np.ndarray  # [x1, y1, x2, y2] in pixel coords
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = None  # (H, W) binary mask if segmentation mode


@dataclass
class FrameResult:
    """Result for a single frame's inference."""

    detections: list = field(default_factory=list)
    frame_shape: tuple = (0, 0, 3)
    inference_time_ms: float = 0.0
    timestamp: float = 0.0


class DetectionEngine:
    """
    Threaded YOLOv8 inference engine.

    Usage:
        engine = DetectionEngine(model_name="yolov8n.pt", use_segmentation=True)
        engine.start()
        engine.submit_frame(frame)
        result = engine.get_result()  # non-blocking
        engine.stop()
    """

    # COCO class IDs for common replaceable objects
    DEFAULT_TARGET_CLASSES = {
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        56: "chair",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        73: "book",
    }

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        use_segmentation: bool = False,
        target_classes: Optional[list] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_detections: int = 20,
        input_size: int = 640,
        device: str = "auto",
        queue_size: int = 2,
    ):
        self.model_name = model_name
        self.use_segmentation = use_segmentation
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.input_size = input_size
        self.device = device
        self.queue_size = queue_size

        # Use segmentation model variant if requested
        if use_segmentation and not model_name.endswith("-seg.pt"):
            seg_name = model_name.replace(".pt", "-seg.pt")
            self.model_name = seg_name

        # Target classes to detect (None = all COCO classes)
        if target_classes is not None:
            self.target_classes = target_classes
        else:
            self.target_classes = list(self.DEFAULT_TARGET_CLASSES.keys())

        # Model and threading
        self._model = None
        self._input_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._output_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._model_loaded = threading.Event()
        self._lock = threading.Lock()

        # Stats
        self.frames_processed = 0
        self.avg_inference_ms = 0.0

    def start(self):
        """Start the inference thread. Model loads on first frame."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Detection engine started (model=%s, seg=%s)",
            self.model_name,
            self.use_segmentation,
        )

    def stop(self):
        """Stop the inference thread and release model."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._model = None
        logger.info("Detection engine stopped")

    def submit_frame(self, frame: np.ndarray):
        """
        Submit a frame for inference (non-blocking).
        Drops frame if queue is full (latest-wins policy).
        """
        try:
            # Clear old frames — we only care about the latest
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except queue.Empty:
                    break
            self._input_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Drop frame

    def get_result(self) -> Optional[FrameResult]:
        """
        Get latest inference result (non-blocking).
        Returns None if no result available.
        """
        result = None
        # Drain queue to get the latest result
        while not self._output_queue.empty():
            try:
                result = self._output_queue.get_nowait()
            except queue.Empty:
                break
        return result

    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._model_loaded.is_set()

    def update_config(
        self,
        confidence_threshold: Optional[float] = None,
        target_classes: Optional[list] = None,
        use_segmentation: Optional[bool] = None,
    ):
        """Thread-safe config update."""
        with self._lock:
            if confidence_threshold is not None:
                self.confidence_threshold = confidence_threshold
            if target_classes is not None:
                self.target_classes = target_classes
            if (
                use_segmentation is not None
                and use_segmentation != self.use_segmentation
            ):
                self.use_segmentation = use_segmentation
                # Reload model on next frame
                self._model = None
                self._model_loaded.clear()

    # ── Private ──────────────────────────────────────────────

    def _load_model(self):
        """Load YOLO model. Called from inference thread."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            return False

        try:
            model_path = self.model_name
            if self.use_segmentation and not model_path.endswith("-seg.pt"):
                model_path = model_path.replace(".pt", "-seg.pt")

            self._model = YOLO(model_path)

            # Determine device
            if self.device == "auto":
                try:
                    import torch

                    if torch.cuda.is_available():
                        self._model.to("cuda")
                        logger.info("Using CUDA GPU for inference")
                    elif (
                        hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ):
                        self._model.to("mps")
                        logger.info("Using Apple MPS for inference")
                    else:
                        logger.info("Using CPU for inference")
                except Exception:
                    logger.info("Using CPU for inference (torch device check failed)")
            elif self.device != "cpu":
                self._model.to(self.device)

            self._model_loaded.set()
            logger.info("Model loaded: %s", model_path)
            return True

        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_name, e)
            return False

    def _inference_loop(self):
        """Main inference loop — runs in background thread."""
        while self._running:
            # Wait for a frame
            try:
                frame = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Lazy model loading
            if self._model is None:
                if not self._load_model():
                    time.sleep(1.0)
                    continue

            # Run inference
            try:
                result = self._run_inference(frame)
                # Put result (non-blocking, drop old results)
                while not self._output_queue.empty():
                    try:
                        self._output_queue.get_nowait()
                    except queue.Empty:
                        break
                self._output_queue.put_nowait(result)
            except Exception as e:
                logger.error("Inference error: %s", e)

    def _run_inference(self, frame: np.ndarray) -> FrameResult:
        """Run YOLOv8 inference on a single frame."""
        t0 = time.perf_counter()

        # Thread-safe config read
        with self._lock:
            conf = self.confidence_threshold
            classes = self.target_classes
            iou = self.iou_threshold
            max_det = self.max_detections
            imgsz = self.input_size

        # Run prediction
        results = self._model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=imgsz,
            classes=classes if classes else None,
            verbose=False,
            stream=False,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Parse results
        frame_result = FrameResult(
            frame_shape=frame.shape,
            inference_time_ms=elapsed_ms,
            timestamp=time.time(),
        )

        if results and len(results) > 0:
            r = results[0]

            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)

                # Get masks if segmentation mode
                masks = None
                if self.use_segmentation and r.masks is not None:
                    masks = r.masks.data.cpu().numpy()

                for i in range(len(boxes)):
                    mask = None
                    if masks is not None and i < len(masks):
                        # Resize mask to frame dimensions
                        mask_raw = masks[i]
                        if mask_raw.shape != frame.shape[:2]:
                            import cv2

                            mask = cv2.resize(
                                mask_raw.astype(np.float32),
                                (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            mask = (mask > 0.5).astype(np.uint8)
                        else:
                            mask = (mask_raw > 0.5).astype(np.uint8)

                    det = Detection(
                        bbox=boxes[i],
                        confidence=float(confs[i]),
                        class_id=int(cls_ids[i]),
                        class_name=r.names.get(int(cls_ids[i]), "unknown"),
                        mask=mask,
                    )
                    frame_result.detections.append(det)

        # Update stats
        self.frames_processed += 1
        alpha = 0.1
        self.avg_inference_ms = alpha * elapsed_ms + (1 - alpha) * self.avg_inference_ms

        return frame_result


class FastSAMRefiner:
    """
    Refines YOLO bounding box detections into precise segmentation masks
    using FastSAM. YOLO detects fast → FastSAM segments within the bbox.

    ~58ms per frame on CPU (Apple Silicon) — viable for real-time.
    """

    def __init__(self, model_name: str = "FastSAM-s.pt", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return True
        try:
            from ultralytics import FastSAM

            self._model = FastSAM(self.model_name)
            self._loaded = True
            logger.info("FastSAM loaded: %s", self.model_name)
            return True
        except ImportError:
            logger.error("ultralytics not installed")
            return False
        except Exception as e:
            logger.error("Failed to load FastSAM: %s", e)
            return False

    def refine(self, frame: np.ndarray, detections: list) -> list:
        """
        Takes YOLO detections with bboxes, returns them with refined SAM masks.
        Modifies Detection.mask in-place.
        """
        if not self._loaded:
            if not self.load():
                return detections

        if not detections:
            return detections

        import cv2

        bboxes = np.array([d.bbox for d in detections])

        try:
            results = self._model(
                frame,
                bboxes=bboxes,
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9,
                verbose=False,
            )

            if results and len(results) > 0 and results[0].masks is not None:
                masks_data = results[0].masks.data.cpu().numpy()
                for i, det in enumerate(detections):
                    if i < len(masks_data):
                        mask_raw = masks_data[i]
                        if mask_raw.shape != frame.shape[:2]:
                            mask = cv2.resize(
                                mask_raw.astype(np.float32),
                                (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            det.mask = (mask > 0.5).astype(np.uint8)
                        else:
                            det.mask = (mask_raw > 0.5).astype(np.uint8)

        except Exception as e:
            logger.error("FastSAM refinement failed: %s", e)

        return detections
