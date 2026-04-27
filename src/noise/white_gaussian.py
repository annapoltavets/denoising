"""White Gaussian sensor noise — AWGN (Ch. 1, Ch. 3, Ch. 4)."""

import cv2
import numpy as np

from .base import Noise


class WhiteGaussianNoise(Noise):
    """Additive White Gaussian Noise (AWGN).

    Detection: estimates the noise standard-deviation from the high-frequency
    Laplacian residual (Liu et al. method). When the estimate exceeds
    *sigma_threshold* DN the frame is considered noisy.

    Removal: OpenCV fast Non-Local Means denoising.
    """

    def __init__(self, sigma_threshold: float = 5.0, h: int = 10) -> None:
        self._sigma_threshold = sigma_threshold
        self._h = h

    @property
    def name(self) -> str:
        return "White Gaussian Sensor Noise"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_sigma(self, gray: np.ndarray) -> float:
        """Estimate noise σ via Laplacian-based method (Liu et al. 2006)."""
        kernel = np.array(
            [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32
        )
        filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sigma = np.sqrt(np.pi / 2.0) * np.mean(np.abs(filtered)) / 6.0
        return float(sigma)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        return self._estimate_sigma(gray) > self._sigma_threshold

    def remove(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(
                frame, None, self._h, self._h, 7, 21
            )
        return cv2.fastNlMeansDenoising(frame, None, self._h, 7, 21)
