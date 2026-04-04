"""
Ad Timing Engine

Detects optimal moments to show ads during a stream:
- Audio silence detection (natural pauses)
- Scene change detection (frame differencing)
- Low motion periods (streamer away, loading screens)
- Cooldown management between ad insertions
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional
from collections import deque

import numpy as np
import cv2

logger = logging.getLogger("adstream.timing")


@dataclass
class TimingSignal:
    """A signal indicating an opportunity to show an ad."""

    signal_type: str  # "silence", "scene_change", "low_motion", "manual"
    confidence: float  # 0-1, how good the opportunity is
    timestamp: float
    duration: float = 0.0  # How long the condition has persisted


class SilenceDetector:
    """
    Detects silence / low audio levels in the stream.
    Works by analyzing audio RMS levels over time.
    """

    def __init__(
        self,
        silence_threshold_db: float = -40.0,
        min_silence_duration: float = 1.5,
        window_size: int = 30,
    ):
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration = min_silence_duration
        self.window_size = window_size

        self._levels: deque = deque(maxlen=window_size)
        self._silence_start: Optional[float] = None
        self._is_silent = False

    def update(self, audio_level_db: float) -> Optional[TimingSignal]:
        """
        Feed an audio level measurement.
        Returns a TimingSignal if silence is detected.
        """
        now = time.time()
        self._levels.append((now, audio_level_db))

        if audio_level_db < self.silence_threshold_db:
            if not self._is_silent:
                self._silence_start = now
                self._is_silent = True

            duration = now - self._silence_start
            if duration >= self.min_silence_duration:
                confidence = min(1.0, duration / (self.min_silence_duration * 3))
                return TimingSignal(
                    signal_type="silence",
                    confidence=confidence,
                    timestamp=now,
                    duration=duration,
                )
        else:
            self._is_silent = False
            self._silence_start = None

        return None

    def update_from_frame_audio(
        self, audio_samples: np.ndarray
    ) -> Optional[TimingSignal]:
        """
        Compute audio level from raw samples and check for silence.

        Args:
            audio_samples: numpy array of audio samples (int16 or float32)
        """
        if audio_samples is None or len(audio_samples) == 0:
            return self.update(-60.0)  # Treat no audio as silence

        # Compute RMS
        samples = audio_samples.astype(np.float64)
        rms = np.sqrt(np.mean(samples**2))

        # Convert to dB
        if rms > 0:
            db = 20 * np.log10(rms / 32768.0)  # Assuming int16 range
        else:
            db = -96.0

        return self.update(db)


class SceneChangeDetector:
    """
    Detects significant visual changes between frames.
    Uses frame differencing with histogram comparison.
    """

    def __init__(
        self,
        change_threshold: float = 0.4,
        min_stable_frames: int = 5,
        downscale_factor: int = 4,
    ):
        self.change_threshold = change_threshold
        self.min_stable_frames = min_stable_frames
        self.downscale_factor = downscale_factor

        self._prev_frame: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._stable_count = 0
        self._change_detected = False

    def update(self, frame: np.ndarray) -> Optional[TimingSignal]:
        """
        Feed a video frame and detect scene changes.
        Returns TimingSignal if a scene change is detected.
        """
        now = time.time()

        # Downscale for performance
        h, w = frame.shape[:2]
        small = cv2.resize(
            frame,
            (w // self.downscale_factor, h // self.downscale_factor),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Compute histogram
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if self._prev_hist is None:
            self._prev_hist = hist
            self._prev_frame = gray
            return None

        # Compare histograms (Bhattacharyya distance)
        dist = cv2.compareHist(
            self._prev_hist,
            hist,
            cv2.HISTCMP_BHATTACHARYYA,
        )

        # Also compute structural difference
        if self._prev_frame is not None and self._prev_frame.shape == gray.shape:
            diff = cv2.absdiff(self._prev_frame, gray)
            mean_diff = diff.mean() / 255.0
        else:
            mean_diff = 0.0

        # Combined score
        change_score = 0.6 * dist + 0.4 * mean_diff

        self._prev_hist = hist
        self._prev_frame = gray

        if change_score > self.change_threshold:
            self._change_detected = True
            self._stable_count = 0
            return TimingSignal(
                signal_type="scene_change",
                confidence=min(1.0, change_score / self.change_threshold),
                timestamp=now,
            )
        else:
            if self._change_detected:
                self._stable_count += 1
                if self._stable_count >= self.min_stable_frames:
                    self._change_detected = False

        return None


class MotionDetector:
    """
    Detects low-motion periods (streamer away, loading screen, etc.).
    """

    def __init__(
        self,
        low_motion_threshold: float = 0.02,
        min_low_motion_duration: float = 3.0,
        downscale_factor: int = 4,
    ):
        self.low_motion_threshold = low_motion_threshold
        self.min_low_motion_duration = min_low_motion_duration
        self.downscale_factor = downscale_factor

        self._prev_gray: Optional[np.ndarray] = None
        self._low_motion_start: Optional[float] = None
        self._motion_history: deque = deque(maxlen=30)

    def update(self, frame: np.ndarray) -> Optional[TimingSignal]:
        """Check for low motion in the frame."""
        now = time.time()

        h, w = frame.shape[:2]
        small = cv2.resize(
            frame,
            (w // self.downscale_factor, h // self.downscale_factor),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        if self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return None

        # Frame difference
        diff = cv2.absdiff(self._prev_gray, gray)
        motion_score = diff.mean() / 255.0
        self._motion_history.append(motion_score)
        self._prev_gray = gray

        # Average motion over window
        avg_motion = np.mean(self._motion_history)

        if avg_motion < self.low_motion_threshold:
            if self._low_motion_start is None:
                self._low_motion_start = now

            duration = now - self._low_motion_start
            if duration >= self.min_low_motion_duration:
                confidence = min(1.0, duration / (self.min_low_motion_duration * 2))
                return TimingSignal(
                    signal_type="low_motion",
                    confidence=confidence,
                    timestamp=now,
                    duration=duration,
                )
        else:
            self._low_motion_start = None

        return None


class AdTimingEngine:
    """
    Coordinates all timing signals to determine when to show/hide ads.
    """

    def __init__(
        self,
        min_ad_interval: float = 60.0,
        min_ad_duration: float = 5.0,
        max_ad_duration: float = 30.0,
        confidence_threshold: float = 0.5,
        fade_in_duration: float = 0.8,
        fade_out_duration: float = 0.5,
    ):
        self.min_ad_interval = min_ad_interval
        self.min_ad_duration = min_ad_duration
        self.max_ad_duration = max_ad_duration
        self.confidence_threshold = confidence_threshold
        self.fade_in_duration = fade_in_duration
        self.fade_out_duration = fade_out_duration

        # Sub-detectors
        self.silence_detector = SilenceDetector()
        self.scene_change_detector = SceneChangeDetector()
        self.motion_detector = MotionDetector()

        # State
        self._last_ad_time: float = 0.0
        self._ad_active = False
        self._ad_start_time: float = 0.0
        self._current_opacity: float = 0.0  # 0-1 for fade transitions
        self._pending_signals: list[TimingSignal] = []

    @property
    def is_ad_active(self) -> bool:
        return self._ad_active

    @property
    def current_opacity(self) -> float:
        """Current ad opacity (0-1), accounting for fade transitions."""
        if not self._ad_active:
            return max(0.0, self._current_opacity)
        return min(1.0, self._current_opacity)

    def update(
        self,
        frame: Optional[np.ndarray] = None,
        audio_level_db: Optional[float] = None,
    ) -> dict:
        """
        Update timing engine with current frame/audio data.

        Returns:
            dict with keys:
                - show_ad: bool
                - opacity: float (0-1)
                - signal: Optional[TimingSignal]
                - reason: str
        """
        now = time.time()
        signals = []

        # Collect signals from sub-detectors
        if audio_level_db is not None:
            sig = self.silence_detector.update(audio_level_db)
            if sig:
                signals.append(sig)

        if frame is not None:
            sig = self.scene_change_detector.update(frame)
            if sig:
                signals.append(sig)

            sig = self.motion_detector.update(frame)
            if sig:
                signals.append(sig)

        # Process signals
        best_signal = None
        if signals:
            # Pick the highest confidence signal
            best_signal = max(signals, key=lambda s: s.confidence)

        # Decide whether to show/hide ad
        result = {
            "show_ad": False,
            "opacity": 0.0,
            "signal": best_signal,
            "reason": "",
        }

        if self._ad_active:
            # Ad is currently showing
            ad_duration = now - self._ad_start_time

            # Update fade-in
            if ad_duration < self.fade_in_duration:
                self._current_opacity = ad_duration / self.fade_in_duration
            else:
                self._current_opacity = 1.0

            # Check if we should stop the ad
            if ad_duration >= self.max_ad_duration:
                self._start_fade_out(now)
                result["reason"] = "max duration reached"
            elif (
                ad_duration >= self.min_ad_duration
                and best_signal
                and best_signal.signal_type == "scene_change"
            ):
                # Natural cut point — stop ad
                self._start_fade_out(now)
                result["reason"] = "scene change during ad"
            else:
                result["show_ad"] = True
                result["opacity"] = self._current_opacity
                result["reason"] = "ad playing"

        elif self._current_opacity > 0:
            # Fading out
            fade_elapsed = now - self._ad_start_time  # repurposed as fade-out start
            self._current_opacity = max(
                0.0, self._current_opacity - (1.0 / (self.fade_out_duration * 30))
            )
            result["show_ad"] = True
            result["opacity"] = self._current_opacity
            result["reason"] = "fading out"
            if self._current_opacity <= 0:
                result["show_ad"] = False

        else:
            # No ad active — check if we should start one
            time_since_last = now - self._last_ad_time

            if time_since_last >= self.min_ad_interval and best_signal:
                if best_signal.confidence >= self.confidence_threshold:
                    self._start_ad(now)
                    result["show_ad"] = True
                    result["opacity"] = 0.01  # Starting fade-in
                    result["reason"] = f"triggered by {best_signal.signal_type}"

        return result

    def force_show(self):
        """Manually trigger an ad (e.g., from hotkey)."""
        self._start_ad(time.time())

    def force_hide(self):
        """Manually hide the current ad."""
        self._start_fade_out(time.time())

    def _start_ad(self, now: float):
        """Begin showing an ad."""
        self._ad_active = True
        self._ad_start_time = now
        self._current_opacity = 0.01
        logger.info("Ad triggered at %.1f", now)

    def _start_fade_out(self, now: float):
        """Begin fading out the ad."""
        self._ad_active = False
        self._last_ad_time = now
        # _current_opacity will decay in update()
        logger.info("Ad ending at %.1f", now)
