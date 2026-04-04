"""Default configuration and constants for the AdStream plugin."""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger("adstream.config")

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PLUGIN_DIR, "assets")
CONFIG_DIR = os.path.join(PLUGIN_DIR, "config")

COCO_REPLACEABLE_CLASSES = {
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

DEFAULT_CONFIG = {
    "mode": "overlay",
    "model_name": "yolov8n.pt",
    "use_segmentation": False,
    "confidence_threshold": 0.5,
    "target_classes": [41, 39, 67],
    "camera_index": 0,
    "capture_width": 1920,
    "capture_height": 1080,
    "inference_interval_ms": 100,
    "smoothing_method": "kalman",
    "ema_alpha": 0.35,
    "blend_mode": "alpha",
    "color_adapt": True,
    "edge_feather_px": 5,
    "min_ad_interval": 60.0,
    "min_ad_duration": 5.0,
    "max_ad_duration": 30.0,
    "silence_threshold_db": -40.0,
    "min_silence_duration": 1.5,
    "scene_change_threshold": 0.4,
    "fade_in_duration": 0.8,
    "fade_out_duration": 0.5,
    "virtual_cam_enabled": False,
    "virtual_cam_width": 1920,
    "virtual_cam_height": 1080,
    "virtual_cam_fps": 30,
}


def load_class_mapping(path: Optional[str] = None) -> dict:
    if path is None:
        path = os.path.join(CONFIG_DIR, "class_mapping.json")

    if not os.path.exists(path):
        logger.info("No class mapping found at %s, using empty mapping", path)
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load class mapping: %s", e)
        return {}


def save_class_mapping(mapping: dict, path: Optional[str] = None):
    if path is None:
        path = os.path.join(CONFIG_DIR, "class_mapping.json")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        with open(path, "w") as f:
            json.dump(mapping, f, indent=2)
    except OSError as e:
        logger.error("Failed to save class mapping: %s", e)


def get_available_classes() -> dict[int, str]:
    return COCO_REPLACEABLE_CLASSES.copy()
