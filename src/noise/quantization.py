"""Quantization noise — Ch. 1 (acquisition devices and noise sources)."""

import cv2
import numpy as np

from .base import Noise


class QuantizationNoise(Noise):
    """Errors introduced by analog-to-digital conversion at limited bit depth.

    Noise model: approximately uniform over one quantisation step; variance
    ≈ Δ²/12 where Δ is the step size.

    Detection: analyses the image histogram for regularly spaced peaks
    (quantisation levels).  A low effective bit-depth inferred from the
    histogram's occupied levels signals quantisation artefacts.

    Removal: mild Gaussian smoothing to break up the staircase structure,
    without blurring real edges significantly.
    """

    def __init__(
        self,
        effective_bits_threshold: int = 5,
        gauss_sigma: float = 0.8,
    ) -> None:
        self._bits_threshold = effective_bits_threshold
        self._gauss_sigma = gauss_sigma

    @property
    def name(self) -> str:
        return "Quantization Noise"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_bits(self, gray: np.ndarray) -> float:
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        occupied = int(np.count_nonzero(hist))
        return float(np.log2(max(occupied, 1)))

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        return self._effective_bits(gray) < self._bits_threshold

    def remove(self, frame: np.ndarray) -> np.ndarray:
        ksize = 0
        return cv2.GaussianBlur(frame, (ksize, ksize), self._gauss_sigma)
