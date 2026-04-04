"""
OBS Overlay Source Manager — creates, positions, fades image sources via obspython.
All OBS API calls happen on the main thread via timer callbacks.
"""

import os
import logging
import time
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("adstream.overlay")

try:
    import obspython as obs

    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False
    logger.warning("obspython not available — running in standalone mode")


@dataclass
class OverlaySource:
    track_id: int
    source_name: str
    scene_item: object = None
    image_path: str = ""
    target_opacity: float = 1.0
    current_opacity: float = 0.0
    position: tuple = (0, 0)
    scale: tuple = (1.0, 1.0)
    rotation: float = 0.0
    created: bool = False
    last_update: float = 0.0


class OBSOverlayManager:
    """
    Manages OBS image sources that act as ad overlays.

    Creates image sources on demand, positions them at detected object locations,
    and handles fade in/out transitions. Must be called from OBS main thread.
    """

    SOURCE_PREFIX = "adstream_overlay_"

    def __init__(
        self,
        scene_name: str = "",
        fade_speed: float = 0.05,
        max_overlays: int = 10,
    ):
        self.scene_name = scene_name
        self.fade_speed = fade_speed
        self.max_overlays = max_overlays
        self._overlays: dict[int, OverlaySource] = {}
        self._scene_width = 1920
        self._scene_height = 1080

    def set_scene_resolution(self, width: int, height: int):
        self._scene_width = width
        self._scene_height = height

    def update_overlay(
        self,
        track_id: int,
        image_path: str,
        x: float,
        y: float,
        width: float,
        height: float,
        target_opacity: float = 1.0,
    ):
        """Create or update an overlay for a tracked object."""
        if not OBS_AVAILABLE:
            self._update_standalone(
                track_id, image_path, x, y, width, height, target_opacity
            )
            return

        source_name = f"{self.SOURCE_PREFIX}{track_id}"

        if track_id not in self._overlays:
            if len(self._overlays) >= self.max_overlays:
                self._remove_oldest_overlay()

            overlay = OverlaySource(
                track_id=track_id,
                source_name=source_name,
            )
            self._overlays[track_id] = overlay
            self._create_obs_source(overlay, image_path)

        overlay = self._overlays[track_id]
        overlay.image_path = image_path
        overlay.target_opacity = target_opacity
        overlay.position = (x, y)
        overlay.scale = (width, height)
        overlay.last_update = time.time()

        self._update_obs_source_image(overlay, image_path)
        self._position_obs_source(overlay, x, y, width, height)

    def tick(self):
        """Called every frame from script_tick — handles fade transitions."""
        stale = []
        for track_id, overlay in self._overlays.items():
            opacity_diff = overlay.target_opacity - overlay.current_opacity
            if abs(opacity_diff) > 0.01:
                step = self.fade_speed if opacity_diff > 0 else -self.fade_speed
                overlay.current_opacity = max(
                    0.0, min(1.0, overlay.current_opacity + step)
                )
                self._set_obs_source_opacity(overlay)

            if time.time() - overlay.last_update > 2.0 and overlay.target_opacity > 0:
                overlay.target_opacity = 0.0

            if overlay.current_opacity <= 0 and overlay.target_opacity <= 0:
                stale_seconds = time.time() - overlay.last_update
                if stale_seconds > 5.0:
                    stale.append(track_id)

        for track_id in stale:
            self.remove_overlay(track_id)

    def remove_overlay(self, track_id: int):
        if track_id not in self._overlays:
            return
        overlay = self._overlays.pop(track_id)
        self._destroy_obs_source(overlay)

    def remove_all(self):
        for track_id in list(self._overlays.keys()):
            self.remove_overlay(track_id)

    def set_global_opacity(self, opacity: float):
        for overlay in self._overlays.values():
            overlay.target_opacity = opacity

    def _update_standalone(self, track_id, image_path, x, y, w, h, opacity):
        """Standalone tracking (no OBS) for testing."""
        if track_id not in self._overlays:
            self._overlays[track_id] = OverlaySource(
                track_id=track_id,
                source_name=f"{self.SOURCE_PREFIX}{track_id}",
            )
        ov = self._overlays[track_id]
        ov.image_path = image_path
        ov.position = (x, y)
        ov.scale = (w, h)
        ov.target_opacity = opacity
        ov.current_opacity = opacity
        ov.last_update = time.time()

    def _remove_oldest_overlay(self):
        if not self._overlays:
            return
        oldest_id = min(self._overlays, key=lambda k: self._overlays[k].last_update)
        self.remove_overlay(oldest_id)

    # ── OBS API Wrappers (main thread only) ──────────────────

    def _create_obs_source(self, overlay: OverlaySource, image_path: str):
        if not OBS_AVAILABLE:
            return

        settings = obs.obs_data_create()
        obs.obs_data_set_string(settings, "file", image_path)

        source = obs.obs_source_create_private(
            "image_source", overlay.source_name, settings
        )
        obs.obs_data_release(settings)

        scene_source = (
            obs.obs_frontend_get_current_scene()
            if not self.scene_name
            else obs.obs_get_source_by_name(self.scene_name)
        )
        if scene_source:
            scene = obs.obs_scene_from_source(scene_source)
            if scene:
                scene_item = obs.obs_scene_add(scene, source)
                overlay.scene_item = scene_item
                obs.obs_sceneitem_set_visible(scene_item, True)
            obs.obs_source_release(scene_source)

        obs.obs_source_release(source)
        overlay.created = True
        logger.debug("Created OBS source: %s", overlay.source_name)

    def _destroy_obs_source(self, overlay: OverlaySource):
        if not OBS_AVAILABLE or not overlay.created:
            return

        if overlay.scene_item:
            obs.obs_sceneitem_remove(overlay.scene_item)
            overlay.scene_item = None

        source = obs.obs_get_source_by_name(overlay.source_name)
        if source:
            obs.obs_source_remove(source)
            obs.obs_source_release(source)

        overlay.created = False
        logger.debug("Destroyed OBS source: %s", overlay.source_name)

    def _update_obs_source_image(self, overlay: OverlaySource, image_path: str):
        if not OBS_AVAILABLE or not overlay.created:
            return

        source = obs.obs_get_source_by_name(overlay.source_name)
        if source:
            settings = obs.obs_source_get_settings(source)
            obs.obs_data_set_string(settings, "file", image_path)
            obs.obs_source_update(source, settings)
            obs.obs_data_release(settings)
            obs.obs_source_release(source)

    def _position_obs_source(
        self, overlay: OverlaySource, x: float, y: float, width: float, height: float
    ):
        if not OBS_AVAILABLE or overlay.scene_item is None:
            return

        pos = obs.vec2()
        pos.x = x
        pos.y = y
        obs.obs_sceneitem_set_pos(overlay.scene_item, pos)

        source = obs.obs_get_source_by_name(overlay.source_name)
        if source:
            source_width = obs.obs_source_get_width(source)
            source_height = obs.obs_source_get_height(source)
            obs.obs_source_release(source)

            if source_width > 0 and source_height > 0:
                scale = obs.vec2()
                scale.x = width / source_width
                scale.y = height / source_height
                obs.obs_sceneitem_set_scale(overlay.scene_item, scale)

    def _set_obs_source_opacity(self, overlay: OverlaySource):
        if not OBS_AVAILABLE or not overlay.created:
            return

        source = obs.obs_get_source_by_name(overlay.source_name)
        if not source:
            return

        opacity_value = int(overlay.current_opacity * 100)

        filter_name = f"{overlay.source_name}_opacity"
        existing_filter = obs.obs_source_get_filter_by_name(source, filter_name)

        if existing_filter:
            settings = obs.obs_source_get_settings(existing_filter)
            obs.obs_data_set_int(settings, "opacity", opacity_value)
            obs.obs_source_update(existing_filter, settings)
            obs.obs_data_release(settings)
            obs.obs_source_release(existing_filter)
        else:
            settings = obs.obs_data_create()
            obs.obs_data_set_int(settings, "opacity", opacity_value)
            color_filter = obs.obs_source_create_private(
                "color_filter", filter_name, settings
            )
            obs.obs_source_filter_add(source, color_filter)
            obs.obs_source_release(color_filter)
            obs.obs_data_release(settings)

        obs.obs_source_release(source)
