"""Clipping / saturation noise — Ch. 1 ("Doubly Censored Heteroskedastic Gaussian")."""

import cv2
import numpy as np

from .base import Noise


class ClippingSaturationNoise(Noise):
    """Highlights or shadows hitting the sensor's dynamic-range limits.

    Noise model: doubly censored Gaussian — the true signal is clipped at 0
    (under-exposure) or 255 (over-exposure).

    Detection: measures the fraction of pixels at the minimum and maximum
    intensity levels.  A combined saturation fraction above *clip_threshold*
    flags the frame.

    Removal: inpaints clipped regions using the surrounding pixel context
    (OpenCV Telea inpainting), so that saturated areas are replaced with
    plausible reconstructed values.
    """

    def __init__(
        self,
        clip_threshold: float = 0.01,
        inpaint_radius: int = 3,
    ) -> None:
        self._clip_threshold = clip_threshold
        self._inpaint_radius = inpaint_radius

    @property
    def name(self) -> str:
        return "Clipping / Saturation Noise"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clip_mask(self, gray: np.ndarray) -> np.ndarray:
        return ((gray == 0) | (gray == 255)).astype(np.uint8)

    def _clip_fraction(self, gray: np.ndarray) -> float:
        mask = self._clip_mask(gray)
        return float(mask.sum()) / max(gray.size, 1)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        return self._clip_fraction(gray) > self._clip_threshold

    def remove(self, frame: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        mask = self._clip_mask(gray)
        return cv2.inpaint(frame, mask, self._inpaint_radius, cv2.INPAINT_TELEA)
