"""Flicker noise / pink (1/f) noise — Ch. 8."""

import cv2
import numpy as np

from .base import Noise


class FlickerNoise(Noise):
    """Slow electronic drift with more energy at low temporal frequencies.

    Noise model: 1/f (pink) noise manifesting as global brightness
    fluctuations that drift slowly over time.

    Detection: tracks a running mean of per-frame global brightness.  A
    significant deviation of the current frame's brightness from the running
    mean signals a flicker event.

    Removal: scale the current frame so that its global brightness matches
    the running mean, correcting the temporal drift.
    """

    def __init__(
        self,
        warmup_frames: int = 5,
        flicker_threshold: float = 5.0,
        learning_rate: float = 0.1,
    ) -> None:
        self._warmup = warmup_frames
        self._flicker_threshold = flicker_threshold
        self._lr = learning_rate
        self._frame_count: int = 0
        self._running_mean: float | None = None

    @property
    def name(self) -> str:
        return "Flicker Noise / Pink Noise"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _global_brightness(self, frame: np.ndarray) -> float:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        return float(gray.mean())

    def _update_running_mean(self, brightness: float) -> None:
        if self._running_mean is None:
            self._running_mean = brightness
        else:
            self._running_mean = (
                (1.0 - self._lr) * self._running_mean + self._lr * brightness
            )
        self._frame_count += 1

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        brightness = self._global_brightness(frame)
        self._update_running_mean(brightness)
        if self._frame_count < self._warmup:
            return False
        return abs(brightness - self._running_mean) > self._flicker_threshold

    def remove(self, frame: np.ndarray) -> np.ndarray:
        if self._running_mean is None:
            return frame
        brightness = self._global_brightness(frame)
        if brightness < 1.0:
            return frame
        scale = self._running_mean / brightness
        result = frame.astype(np.float32) * scale
        return np.clip(result, 0, 255).astype(np.uint8)
