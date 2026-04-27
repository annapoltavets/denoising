"""Fixed-pattern noise — Ch. 1 (correlated noise, nonuniformity)."""

import cv2
import numpy as np

from .base import Noise


class FixedPatternNoise(Noise):
    """Stable, sensor-tied additive pattern that repeats across frames.

    Noise model: structured correlated noise; each pixel has a systematic
    bias that is constant over time.

    Detection: accumulates a running temporal mean.  Once enough frames have
    been seen, the deviation of the current frame's spatial structure from the
    temporal mean is evaluated.  A high structural similarity between the
    current frame and the estimated pattern indicates fixed-pattern noise.

    Removal: subtract the estimated fixed pattern from the current frame.
    """

    def __init__(
        self,
        warmup_frames: int = 10,
        pattern_threshold: float = 3.0,
        learning_rate: float = 0.05,
    ) -> None:
        self._warmup = warmup_frames
        self._pattern_threshold = pattern_threshold
        self._lr = learning_rate
        self._frame_count: int = 0
        self._mean_frame: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "Fixed-Pattern Noise"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_float_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        return gray.astype(np.float32)

    def _update_mean(self, gray_f: np.ndarray) -> None:
        if self._mean_frame is None:
            self._mean_frame = gray_f.copy()
        else:
            self._mean_frame = (
                (1.0 - self._lr) * self._mean_frame + self._lr * gray_f
            )
        self._frame_count += 1

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        gray_f = self._to_float_gray(frame)
        self._update_mean(gray_f)
        if self._frame_count < self._warmup:
            return False
        # Pattern estimate: mean minus its own mean (zero-DC component)
        pattern = self._mean_frame - self._mean_frame.mean()
        pattern_std = float(np.std(pattern))
        return pattern_std > self._pattern_threshold

    def remove(self, frame: np.ndarray) -> np.ndarray:
        if self._mean_frame is None:
            return frame
        pattern = self._mean_frame - self._mean_frame.mean()
        if frame.ndim == 3:
            result = frame.astype(np.float32)
            for c in range(3):
                result[:, :, c] -= pattern
            return np.clip(result, 0, 255).astype(np.uint8)
        result = frame.astype(np.float32) - pattern
        return np.clip(result, 0, 255).astype(np.uint8)
