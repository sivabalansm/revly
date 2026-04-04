"""
AdStream — OBS Python Script Plugin
Smart ad insertion with ML-powered object detection and replacement.

Install: OBS -> Tools -> Scripts -> Add this file
"""

import sys
import os
import logging
import time
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="[AdStream] %(levelname)s: %(message)s",
)
logger = logging.getLogger("adstream")

try:
    import obspython as obs

    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False
    logger.warning("obspython not available — running standalone")

from lib.config import DEFAULT_CONFIG, ASSETS_DIR, CONFIG_DIR, load_class_mapping
from lib.detector import DetectionEngine
from lib.tracker import ObjectTracker
from lib.replacer import ReplacementManager
from lib.timing import AdTimingEngine
from lib.overlay import OBSOverlayManager
from lib.frame_capture import FrameCaptureManager
from lib.virtual_cam import VideoStreamOutput, ReplaceModeProcessor


class AdStreamPlugin:
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.detector: DetectionEngine = None
        self.tracker: ObjectTracker = None
        self.replacement_mgr: ReplacementManager = None
        self.timing_engine: AdTimingEngine = None
        self.overlay_mgr: OBSOverlayManager = None
        self.frame_capture: FrameCaptureManager = None
        self.video_output: VideoStreamOutput = None
        self.replace_processor: ReplaceModeProcessor = None

        self._initialized = False
        self._processing = False
        self._last_detection_result = None
        self._last_frame_time = 0.0
        self._stats = {
            "frames_processed": 0,
            "detections_total": 0,
            "ads_shown": 0,
            "avg_inference_ms": 0.0,
        }

    def initialize(self):
        if self._initialized:
            return

        logger.info("Initializing AdStream plugin...")

        self.detector = DetectionEngine(
            model_name=self.config["model_name"],
            use_segmentation=self.config["use_segmentation"],
            confidence_threshold=self.config["confidence_threshold"],
            target_classes=self.config["target_classes"],
        )

        self.tracker = ObjectTracker(
            smoothing_method=self.config["smoothing_method"],
            ema_alpha=self.config["ema_alpha"],
        )

        self.replacement_mgr = ReplacementManager()
        class_mapping = load_class_mapping()
        if class_mapping:
            self.replacement_mgr.load_from_mapping(class_mapping, ASSETS_DIR)

        self.timing_engine = AdTimingEngine(
            min_ad_interval=self.config["min_ad_interval"],
            min_ad_duration=self.config["min_ad_duration"],
            max_ad_duration=self.config["max_ad_duration"],
            fade_in_duration=self.config["fade_in_duration"],
            fade_out_duration=self.config["fade_out_duration"],
        )

        self.overlay_mgr = OBSOverlayManager()

        self.frame_capture = FrameCaptureManager(
            mode="camera",
            camera_index=self.config["camera_index"],
            width=self.config["capture_width"],
            height=self.config["capture_height"],
        )

        if self.config["virtual_cam_enabled"]:
            self.video_output = VideoStreamOutput(
                width=self.config["virtual_cam_width"],
                height=self.config["virtual_cam_height"],
                fps=self.config["virtual_cam_fps"],
            )

        self._initialized = True
        logger.info("AdStream initialized (mode=%s)", self.config["mode"])

    def start_processing(self):
        if self._processing:
            return

        self.initialize()

        if not self.frame_capture.start():
            logger.error("Failed to start frame capture")
            return

        self.detector.start()

        if self.config["mode"] == "replace" and self.video_output:
            if self.video_output.start():
                self.replace_processor = ReplaceModeProcessor(
                    detector=self.detector,
                    tracker=self.tracker,
                    replacement_manager=self.replacement_mgr,
                    frame_capture=self.frame_capture,
                    video_output=self.video_output,
                )
                self.replace_processor.start()
                logger.info("Replace mode stream: %s", self.video_output.stream_url)

        self._processing = True
        logger.info("AdStream processing started")

    def stop_processing(self):
        if not self._processing:
            return

        if self.replace_processor:
            self.replace_processor.stop()
            self.replace_processor = None

        if self.video_output:
            self.video_output.stop()

        self.detector.stop()
        self.frame_capture.stop()
        self.overlay_mgr.remove_all()

        self._processing = False
        logger.info("AdStream processing stopped")

    def process_tick(self):
        """Called from OBS main thread every frame."""
        if not self._processing:
            return

        if self.config["mode"] == "replace":
            return

        frame = self.frame_capture.get_frame()
        if frame is None:
            return

        now = time.time()
        interval_sec = self.config["inference_interval_ms"] / 1000.0
        if now - self._last_frame_time >= interval_sec:
            self.detector.submit_frame(frame)
            self._last_frame_time = now

        result = self.detector.get_result()
        if result is not None:
            self._last_detection_result = result
            self._stats["frames_processed"] += 1
            self._stats["avg_inference_ms"] = result.inference_time_ms

        timing_result = self.timing_engine.update(frame=frame, audio_level_db=None)

        if self._last_detection_result is not None:
            tracked_objects = self.tracker.update(
                self._last_detection_result.detections
            )

            for obj in tracked_objects:
                if not obj.is_visible:
                    continue

                asset = self.replacement_mgr.get_asset(obj.class_id)
                if asset is None:
                    continue

                ad_opacity = (
                    timing_result.get("opacity", 1.0)
                    if timing_result.get("show_ad", True)
                    else 0.0
                )

                overlay_path = self.replacement_mgr.replacer.save_overlay_image(
                    asset,
                    obj.bbox,
                    obj.track_id,
                )

                if overlay_path:
                    x1, y1, x2, y2 = obj.bbox
                    self.overlay_mgr.update_overlay(
                        track_id=obj.track_id,
                        image_path=overlay_path,
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1),
                        target_opacity=ad_opacity,
                    )

                    self._stats["detections_total"] += 1

        self.overlay_mgr.tick()

    def update_config(self, key: str, value):
        self.config[key] = value

        if self.detector and key in (
            "confidence_threshold",
            "target_classes",
            "use_segmentation",
        ):
            self.detector.update_config(**{key: value})

        if key == "mode" and self._processing:
            self.stop_processing()
            self.start_processing()

    def shutdown(self):
        self.stop_processing()
        self._initialized = False
        logger.info("AdStream shutdown complete")


plugin = AdStreamPlugin()


# ── OBS Script Exports ───────────────────────────────────────────


def script_description():
    return (
        "<h2>AdStream — Smart Ad Insertion</h2>"
        "<p>ML-powered object detection and replacement for seamless ad integration.</p>"
        "<p><b>Modes:</b></p>"
        "<ul>"
        "<li><b>Overlay:</b> Positions ad images over detected objects using OBS sources</li>"
        "<li><b>Replace:</b> Pixel-level object replacement via MJPEG stream (add Media Source → http://localhost:8765/stream)</li>"
        "</ul>"
        "<p>Detects optimal ad timing via audio silence, scene changes, and low motion.</p>"
    )


def script_properties():
    props = obs.obs_properties_create()

    mode_list = obs.obs_properties_add_list(
        props,
        "mode",
        "Processing Mode",
        obs.OBS_COMBO_TYPE_LIST,
        obs.OBS_COMBO_FORMAT_STRING,
    )
    obs.obs_property_list_add_string(mode_list, "Overlay (OBS Sources)", "overlay")
    obs.obs_property_list_add_string(mode_list, "Replace (MJPEG Stream)", "replace")

    obs.obs_properties_add_int(props, "camera_index", "Camera Index", 0, 10, 1)

    model_list = obs.obs_properties_add_list(
        props,
        "model_name",
        "YOLO Model",
        obs.OBS_COMBO_TYPE_LIST,
        obs.OBS_COMBO_FORMAT_STRING,
    )
    obs.obs_property_list_add_string(model_list, "YOLOv8 Nano (fastest)", "yolov8n.pt")
    obs.obs_property_list_add_string(model_list, "YOLOv8 Small", "yolov8s.pt")
    obs.obs_property_list_add_string(model_list, "YOLOv8 Medium", "yolov8m.pt")

    obs.obs_properties_add_bool(
        props, "use_segmentation", "Enable Segmentation (pixel masks)"
    )

    obs.obs_properties_add_float_slider(
        props,
        "confidence_threshold",
        "Confidence Threshold",
        0.1,
        1.0,
        0.05,
    )

    obs.obs_properties_add_int_slider(
        props,
        "inference_interval_ms",
        "Inference Interval (ms)",
        16,
        1000,
        10,
    )

    smooth_list = obs.obs_properties_add_list(
        props,
        "smoothing_method",
        "Tracking Smoothing",
        obs.OBS_COMBO_TYPE_LIST,
        obs.OBS_COMBO_FORMAT_STRING,
    )
    obs.obs_property_list_add_string(smooth_list, "Kalman Filter", "kalman")
    obs.obs_property_list_add_string(smooth_list, "Exponential Moving Average", "ema")

    blend_list = obs.obs_properties_add_list(
        props,
        "blend_mode",
        "Blend Mode",
        obs.OBS_COMBO_TYPE_LIST,
        obs.OBS_COMBO_FORMAT_STRING,
    )
    obs.obs_property_list_add_string(blend_list, "Alpha Blend", "alpha")
    obs.obs_property_list_add_string(blend_list, "Poisson (Seamless)", "poisson")

    obs.obs_properties_add_bool(props, "color_adapt", "Adapt Colors to Scene")

    obs.obs_properties_add_float_slider(
        props,
        "min_ad_interval",
        "Min Ad Interval (seconds)",
        10.0,
        300.0,
        5.0,
    )
    obs.obs_properties_add_float_slider(
        props,
        "min_ad_duration",
        "Min Ad Duration (seconds)",
        1.0,
        30.0,
        0.5,
    )
    obs.obs_properties_add_float_slider(
        props,
        "silence_threshold_db",
        "Silence Threshold (dB)",
        -60.0,
        -10.0,
        1.0,
    )

    obs.obs_properties_add_path(
        props,
        "ad_asset_path",
        "Ad Image",
        obs.OBS_PATH_FILE,
        "Images (*.png *.jpg *.jpeg *.bmp);;All (*)",
        ASSETS_DIR,
    )

    obs.obs_properties_add_bool(
        props, "virtual_cam_enabled", "Enable Virtual Camera (Replace Mode)"
    )

    obs.obs_properties_add_button(
        props, "btn_start", "Start Processing", on_start_clicked
    )
    obs.obs_properties_add_button(props, "btn_stop", "Stop Processing", on_stop_clicked)
    obs.obs_properties_add_button(
        props, "btn_force_ad", "Force Show Ad", on_force_ad_clicked
    )
    obs.obs_properties_add_button(
        props, "btn_install_deps", "Install Dependencies", on_install_deps_clicked
    )

    return props


def script_defaults(settings):
    for key, value in DEFAULT_CONFIG.items():
        if isinstance(value, str):
            obs.obs_data_set_default_string(settings, key, value)
        elif isinstance(value, bool):
            obs.obs_data_set_default_bool(settings, key, value)
        elif isinstance(value, int):
            obs.obs_data_set_default_int(settings, key, value)
        elif isinstance(value, float):
            obs.obs_data_set_default_double(settings, key, value)


def script_update(settings):
    cfg = plugin.config

    str_keys = ["mode", "model_name", "smoothing_method", "blend_mode"]
    bool_keys = ["use_segmentation", "color_adapt", "virtual_cam_enabled"]
    int_keys = ["camera_index", "inference_interval_ms"]
    float_keys = [
        "confidence_threshold",
        "ema_alpha",
        "min_ad_interval",
        "min_ad_duration",
        "max_ad_duration",
        "silence_threshold_db",
        "fade_in_duration",
        "fade_out_duration",
    ]

    for key in str_keys:
        val = obs.obs_data_get_string(settings, key)
        if val:
            plugin.update_config(key, val)

    for key in bool_keys:
        plugin.update_config(key, obs.obs_data_get_bool(settings, key))

    for key in int_keys:
        val = obs.obs_data_get_int(settings, key)
        if val > 0:
            plugin.update_config(key, val)

    for key in float_keys:
        val = obs.obs_data_get_double(settings, key)
        if val != 0:
            plugin.update_config(key, val)

    ad_path = obs.obs_data_get_string(settings, "ad_asset_path")
    if ad_path and os.path.exists(ad_path):
        plugin.initialize()
        if plugin.replacement_mgr:
            for class_id in plugin.config.get("target_classes", []):
                plugin.replacement_mgr.set_asset(class_id, ad_path)


def script_load(settings):
    logger.info("AdStream plugin loaded")


def script_tick(seconds):
    plugin.process_tick()


def script_unload():
    plugin.shutdown()
    logger.info("AdStream plugin unloaded")


# ── Button Callbacks ─────────────────────────────────────────────


def on_start_clicked(props, prop):
    plugin.start_processing()
    return True


def on_stop_clicked(props, prop):
    plugin.stop_processing()
    return True


def on_force_ad_clicked(props, prop):
    if plugin.timing_engine:
        plugin.timing_engine.force_show()
    return True


def on_install_deps_clicked(props, prop):
    import subprocess

    requirements = os.path.join(SCRIPT_DIR, "requirements.txt")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                requirements,
            ]
        )
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install dependencies: %s", e)
    return True
