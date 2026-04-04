"""
Object Replacement Renderer

Handles perspective warp, color adaptation, alpha compositing,
and Poisson blending for seamless object replacement.
"""

import os
import logging
import tempfile
from typing import Optional

import numpy as np
import cv2

logger = logging.getLogger("adstream.replacer")


class ReplacementAsset:
    """A loaded ad/replacement image with metadata."""

    def __init__(self, image_path: str, name: str = ""):
        self.image_path = image_path
        self.name = name or os.path.basename(image_path)
        self._image: Optional[np.ndarray] = None
        self._image_rgba: Optional[np.ndarray] = None

    @property
    def image(self) -> Optional[np.ndarray]:
        """Load image lazily (BGR)."""
        if self._image is None:
            self._load()
        return self._image

    @property
    def image_rgba(self) -> Optional[np.ndarray]:
        """Load image with alpha channel (BGRA)."""
        if self._image_rgba is None:
            self._load()
        return self._image_rgba

    def _load(self):
        """Load image from disk."""
        if not os.path.exists(self.image_path):
            logger.error("Asset not found: %s", self.image_path)
            return

        # Load with alpha channel if present
        img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error("Failed to load: %s", self.image_path)
            return

        if img.ndim == 2:
            # Grayscale -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.shape[2] == 4:
            self._image_rgba = img.copy()
            self._image = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            self._image = img.copy()
            # Create RGBA with full opacity
            alpha = np.ones((*img.shape[:2], 1), dtype=np.uint8) * 255
            self._image_rgba = np.concatenate([img, alpha], axis=2)


class ObjectReplacer:
    """
    Renders replacement images at detected object locations.

    Two modes:
    1. Overlay mode: Generates a warped overlay image + mask for OBS source positioning
    2. Replace mode: Full pixel-level compositing into the frame
    """

    def __init__(
        self,
        blend_mode: str = "alpha",  # "alpha", "poisson", "seamless"
        color_adapt: bool = True,
        edge_feather_px: int = 5,
        temp_dir: Optional[str] = None,
    ):
        self.blend_mode = blend_mode
        self.color_adapt = color_adapt
        self.edge_feather_px = edge_feather_px
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="adstream_")
        os.makedirs(self.temp_dir, exist_ok=True)

    def render_overlay(
        self,
        asset: ReplacementAsset,
        bbox: np.ndarray,
        frame_shape: tuple,
        mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Render a replacement image warped to the detection bbox.

        Returns:
            (warped_image, warped_mask) both at frame_shape resolution.
            warped_image is BGRA (with alpha).
        """
        if asset.image is None:
            return np.zeros((*frame_shape[:2], 4), dtype=np.uint8), np.zeros(
                frame_shape[:2], dtype=np.uint8
            )

        x1, y1, x2, y2 = map(int, bbox)
        target_w = max(1, x2 - x1)
        target_h = max(1, y2 - y1)

        # Resize replacement to target dimensions
        replacement = cv2.resize(
            asset.image_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA
        )

        # Create full-frame output
        warped = np.zeros((*frame_shape[:2], 4), dtype=np.uint8)
        warped_mask = np.zeros(frame_shape[:2], dtype=np.uint8)

        # Clamp to frame boundaries
        fx1 = max(0, x1)
        fy1 = max(0, y1)
        fx2 = min(frame_shape[1], x2)
        fy2 = min(frame_shape[0], y2)

        # Source crop offsets
        sx1 = fx1 - x1
        sy1 = fy1 - y1
        sx2 = sx1 + (fx2 - fx1)
        sy2 = sy1 + (fy2 - fy1)

        if sx2 <= sx1 or sy2 <= sy1:
            return warped, warped_mask

        warped[fy1:fy2, fx1:fx2] = replacement[sy1:sy2, sx1:sx2]
        warped_mask[fy1:fy2, fx1:fx2] = replacement[sy1:sy2, sx1:sx2, 3]

        # If segmentation mask provided, apply it to refine edges
        if mask is not None:
            warped_mask = cv2.bitwise_and(warped_mask, mask * 255)

        # Feather edges
        if self.edge_feather_px > 0:
            k = self.edge_feather_px * 2 + 1
            warped_mask = cv2.GaussianBlur(warped_mask, (k, k), 0)

        return warped, warped_mask

    def composite_frame(
        self,
        frame: np.ndarray,
        asset: ReplacementAsset,
        bbox: np.ndarray,
        mask: Optional[np.ndarray] = None,
        context_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Full pixel-level replacement in frame (Replace Mode).

        Args:
            frame: Input frame (BGR)
            asset: Replacement image asset
            bbox: [x1, y1, x2, y2]
            mask: Optional segmentation mask
            context_frame: Frame for color adaptation context (defaults to frame)

        Returns:
            Frame with object replaced (BGR)
        """
        if asset.image_rgba is None:
            return frame

        result = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)
        target_w = max(1, x2 - x1)
        target_h = max(1, y2 - y1)

        resized_rgba = cv2.resize(
            asset.image_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA
        )
        replacement = resized_rgba[:, :, :3]
        png_alpha = resized_rgba[:, :, 3].astype(np.float32) / 255.0

        if mask is not None:
            replacement, png_alpha = self._fit_to_mask(
                replacement, png_alpha, mask, bbox
            )

        if self.color_adapt:
            ctx = context_frame if context_frame is not None else frame
            replacement = self._adapt_colors(replacement, ctx, bbox)

        if mask is not None:
            mh, mw = mask.shape[:2]
            cy1, cy2 = max(0, y1), min(mh, y2)
            cx1, cx2 = max(0, x1), min(mw, x2)
            if cy2 > cy1 and cx2 > cx1:
                seg_mask = mask[cy1:cy2, cx1:cx2].astype(np.float32)
                if seg_mask.shape[:2] != png_alpha.shape[:2]:
                    seg_mask = cv2.resize(seg_mask, (target_w, target_h))
                if seg_mask.max() > 1:
                    seg_mask = seg_mask / 255.0
                combined_alpha = png_alpha * seg_mask
            else:
                combined_alpha = png_alpha
        else:
            combined_alpha = png_alpha

        if self.blend_mode == "poisson" or self.blend_mode == "seamless":
            alpha_mask_uint8 = (combined_alpha * 255).astype(np.uint8)
            result = self._seamless_blend(result, replacement, bbox, alpha_mask_uint8)
        else:
            result = self._alpha_blend_with_alpha(
                result, replacement, bbox, combined_alpha
            )

        return result

    def save_overlay_image(
        self,
        asset: ReplacementAsset,
        bbox: np.ndarray,
        track_id: int,
    ) -> Optional[str]:
        """
        Save a cropped+warped replacement image to temp file.
        Returns the file path for use as an OBS image source.
        """
        if asset.image is None:
            return None

        x1, y1, x2, y2 = map(int, bbox)
        target_w = max(1, x2 - x1)
        target_h = max(1, y2 - y1)

        # Resize with alpha
        overlay = cv2.resize(
            asset.image_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA
        )

        # Save as PNG (preserves alpha)
        path = os.path.join(self.temp_dir, f"overlay_{track_id}.png")
        cv2.imwrite(path, overlay)

        return path

    def _fit_to_mask(
        self,
        replacement: np.ndarray,
        png_alpha: np.ndarray,
        mask: np.ndarray,
        bbox: np.ndarray,
    ) -> tuple:
        x1, y1, x2, y2 = map(int, bbox)
        target_w = max(1, x2 - x1)
        target_h = max(1, y2 - y1)

        mask_h, mask_w = mask.shape[:2]
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(mask_w, x2), min(mask_h, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            return replacement, png_alpha

        roi_mask = mask[cy1:cy2, cx1:cx2]
        if roi_mask.size == 0:
            return replacement, png_alpha
        if roi_mask.shape[:2] != (target_h, target_w):
            roi_mask = cv2.resize(roi_mask.astype(np.uint8), (target_w, target_h))

        roi_mask_uint8 = (
            (roi_mask * 255).astype(np.uint8)
            if roi_mask.max() <= 1
            else roi_mask.astype(np.uint8)
        )

        contours, _ = cv2.findContours(
            roi_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return replacement, png_alpha

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return replacement, png_alpha

        mask_rect = cv2.boundingRect(largest)
        mx, my, mw, mh = mask_rect

        src_h, src_w = replacement.shape[:2]
        ad_opaque = (png_alpha > 0.1).astype(np.uint8) * 255
        ad_contours, _ = cv2.findContours(
            ad_opaque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if ad_contours:
            ad_largest = max(ad_contours, key=cv2.contourArea)
            ax, ay, aw, ah = cv2.boundingRect(ad_largest)
        else:
            ax, ay, aw, ah = 0, 0, src_w, src_h

        src_pts = np.float32(
            [
                [ax, ay],
                [ax + aw, ay],
                [ax + aw, ay + ah],
                [ax, ay + ah],
            ]
        )

        dst_pts = np.float32(
            [
                [mx, my],
                [mx + mw, my],
                [mx + mw, my + mh],
                [mx, my + mh],
            ]
        )

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_bgr = cv2.warpPerspective(
            replacement,
            M,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_alpha = cv2.warpPerspective(
            png_alpha,
            M,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return warped_bgr, warped_alpha

    # ── Color Adaptation ─────────────────────────────────────

    def _adapt_colors(
        self,
        replacement: np.ndarray,
        frame: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray:
        """
        Adapt replacement colors to match scene lighting.
        Uses Reinhard color transfer in LAB space.
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Get context region around the object
        pad = max(10, int(min(x2 - x1, y2 - y1) * 0.3))
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(frame.shape[1], x2 + pad)
        cy2 = min(frame.shape[0], y2 + pad)

        context = frame[cy1:cy2, cx1:cx2]
        if context.size == 0:
            return replacement

        # Convert to LAB
        try:
            rep_lab = cv2.cvtColor(replacement, cv2.COLOR_BGR2LAB).astype(np.float64)
            ctx_lab = cv2.cvtColor(context, cv2.COLOR_BGR2LAB).astype(np.float64)
        except Exception:
            return replacement

        # Transfer L channel statistics (luminance) — most impactful
        for ch in range(3):
            src_mean = rep_lab[:, :, ch].mean()
            src_std = rep_lab[:, :, ch].std() + 1e-8
            tgt_mean = ctx_lab[:, :, ch].mean()
            tgt_std = ctx_lab[:, :, ch].std() + 1e-8

            # Only adapt L channel fully, a/b channels partially
            weight = 1.0 if ch == 0 else 0.3
            rep_lab[:, :, ch] = (
                (rep_lab[:, :, ch] - src_mean) * (tgt_std / src_std) * weight
                + src_mean * (1 - weight)
                + tgt_mean * weight
            )

        rep_lab = np.clip(rep_lab, 0, 255).astype(np.uint8)
        adapted = cv2.cvtColor(rep_lab, cv2.COLOR_LAB2BGR)

        return adapted

    # ── Blending Methods ─────────────────────────────────────

    def _alpha_blend(
        self,
        frame: np.ndarray,
        replacement: np.ndarray,
        bbox: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Alpha blend replacement into frame at bbox location."""
        result = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp
        h, w = frame.shape[:2]
        fx1, fy1 = max(0, x1), max(0, y1)
        fx2, fy2 = min(w, x2), min(h, y2)
        sx1, sy1 = fx1 - x1, fy1 - y1
        sx2, sy2 = sx1 + (fx2 - fx1), sy1 + (fy2 - fy1)

        if sx2 <= sx1 or sy2 <= sy1:
            return result

        region = replacement[sy1:sy2, sx1:sx2]

        if mask is not None:
            # Use segmentation mask
            m = mask[fy1:fy2, fx1:fx2].astype(np.float32)
            if m.max() > 1:
                m = m / 255.0
        else:
            # Full opacity within bbox
            m = np.ones((fy2 - fy1, fx2 - fx1), dtype=np.float32)

        # Feather edges
        if self.edge_feather_px > 0:
            k = self.edge_feather_px * 2 + 1
            m = cv2.GaussianBlur(m, (k, k), 0)

        m3 = m[:, :, np.newaxis]
        blended = region.astype(np.float32) * m3 + result[fy1:fy2, fx1:fx2].astype(
            np.float32
        ) * (1 - m3)
        result[fy1:fy2, fx1:fx2] = np.clip(blended, 0, 255).astype(np.uint8)

        return result

    def _alpha_blend_with_alpha(
        self,
        frame: np.ndarray,
        replacement: np.ndarray,
        bbox: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        result = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)

        h, w = frame.shape[:2]
        fx1, fy1 = max(0, x1), max(0, y1)
        fx2, fy2 = min(w, x2), min(h, y2)
        sx1, sy1 = fx1 - x1, fy1 - y1
        sx2, sy2 = sx1 + (fx2 - fx1), sy1 + (fy2 - fy1)

        if sx2 <= sx1 or sy2 <= sy1:
            return result

        region = replacement[sy1:sy2, sx1:sx2]
        m = alpha[sy1:sy2, sx1:sx2].copy()

        if self.edge_feather_px > 0:
            k = self.edge_feather_px * 2 + 1
            m = cv2.GaussianBlur(m, (k, k), 0)

        m3 = m[:, :, np.newaxis]
        blended = region.astype(np.float32) * m3 + result[fy1:fy2, fx1:fx2].astype(
            np.float32
        ) * (1 - m3)
        result[fy1:fy2, fx1:fx2] = np.clip(blended, 0, 255).astype(np.uint8)

        return result

    def _seamless_blend(
        self,
        frame: np.ndarray,
        replacement: np.ndarray,
        bbox: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Poisson/seamless blend replacement into frame."""
        x1, y1, x2, y2 = map(int, bbox)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Create mask for seamlessClone
        if mask is not None:
            blend_mask = mask[y1:y2, x1:x2]
            if blend_mask.max() > 1:
                blend_mask = (blend_mask > 127).astype(np.uint8) * 255
        else:
            blend_mask = np.ones(replacement.shape[:2], dtype=np.uint8) * 255
            # Erode slightly to avoid edge artifacts
            kernel = np.ones((3, 3), np.uint8)
            blend_mask = cv2.erode(blend_mask, kernel, iterations=1)

        # Ensure blend_mask matches replacement dimensions
        if blend_mask.shape[:2] != replacement.shape[:2]:
            blend_mask = cv2.resize(
                blend_mask, (replacement.shape[1], replacement.shape[0])
            )

        try:
            result = cv2.seamlessClone(
                replacement,
                frame,
                blend_mask,
                (center_x, center_y),
                cv2.NORMAL_CLONE,
            )
            return result
        except cv2.error as e:
            logger.warning("seamlessClone failed: %s, falling back to alpha blend", e)
            return self._alpha_blend(frame, replacement, bbox, mask)


class ReplacementManager:
    """
    Manages the mapping of detected object classes to replacement assets.
    """

    def __init__(self):
        self._assets: dict[int, ReplacementAsset] = {}  # class_id -> asset
        self._default_asset: Optional[ReplacementAsset] = None
        self._replacer = ObjectReplacer()

    @property
    def replacer(self) -> ObjectReplacer:
        return self._replacer

    def set_asset(self, class_id: int, image_path: str, name: str = ""):
        """Map a COCO class ID to a replacement image."""
        self._assets[class_id] = ReplacementAsset(image_path, name)

    def set_default_asset(self, image_path: str, name: str = ""):
        """Set a fallback asset for any detected object."""
        self._default_asset = ReplacementAsset(image_path, name)

    def get_asset(self, class_id: int) -> Optional[ReplacementAsset]:
        """Get the replacement asset for a class, or default."""
        return self._assets.get(class_id, self._default_asset)

    def load_from_mapping(self, mapping: dict, assets_dir: str = ""):
        """
        Load class-to-asset mapping from a dict.
        Format: { "41": "coca_cola.png", "44": "pepsi.png" }
        """
        for class_id_str, filename in mapping.items():
            class_id = int(class_id_str)
            path = os.path.join(assets_dir, filename) if assets_dir else filename
            if os.path.exists(path):
                self.set_asset(class_id, path)
                logger.info("Mapped class %d -> %s", class_id, path)
            else:
                logger.warning("Asset not found: %s", path)

    def clear(self):
        """Remove all asset mappings."""
        self._assets.clear()
        self._default_asset = None
