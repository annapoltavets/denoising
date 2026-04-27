"""Photo-response nonuniformity (PRNU) — Ch. 1."""

import cv2
import numpy as np

from .base import Noise


class PRNUNoise(Noise):
    """Stable per-pixel multiplicative gain variation from manufacturing spread.

    Noise model: each pixel *i* has a small gain offset *K_i* so that
    *Y_i = K_i · X_i + N_i*.  *K_i* is fixed for a given sensor.

    Detection: accumulates a running per-pixel mean (gain map) from multiple
    frames.  The coefficient of variation of the gain map indicates how much
    pixels differ from one another; a value above *cv_threshold* flags PRNU.

    Removal: flat-field correction — divide each frame by the normalised gain
    map to equalise per-pixel response.
    """

    def __init__(
        self,
        warmup_frames: int = 15,
        cv_threshold: float = 0.02,
        learning_rate: float = 0.05,
    ) -> None:
        self._warmup = warmup_frames
        self._cv_threshold = cv_threshold
        self._lr = learning_rate
        self._frame_count: int = 0
        self._gain_map: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "Photo-Response Nonuniformity (PRNU)"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_gray_float(self, frame: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        return gray.astype(np.float32) + 1.0

    def _update_gain_map(self, gray_f: np.ndarray) -> None:
        if self._gain_map is None:
            self._gain_map = gray_f.copy()
        else:
            self._gain_map = (1.0 - self._lr) * self._gain_map + self._lr * gray_f
        self._frame_count += 1

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        gray_f = self._to_gray_float(frame)
        self._update_gain_map(gray_f)
        if self._frame_count < self._warmup:
            return False
        gm = self._gain_map
        cv = float(np.std(gm) / (np.mean(gm) + 1e-10))
        return cv > self._cv_threshold

    def remove(self, frame: np.ndarray) -> np.ndarray:
        if self._gain_map is None:
            return frame
        gm = self._gain_map  # shape (H, W), float32
        normalised_gain = gm / (gm.mean() + 1e-10)  # (H, W)
        frame_f = frame.astype(np.float32)
        if frame.ndim == 3:
            result = frame_f / (normalised_gain[:, :, np.newaxis] + 1e-10)
        else:
            result = frame_f / (normalised_gain + 1e-10)
        return np.clip(result, 0, 255).astype(np.uint8)
