"""Real camera RAW / output noise — Ch. 1, Ch. 4, Ch. 7, Ch. 8, Ch. 9."""

import cv2
import numpy as np

from .base import Noise


class RealCameraRawNoise(Noise):
    """Practical camera noise: signal-dependent, possibly correlated mixture.

    Noise model: camera-specific mixture — Poisson photon noise + Gaussian
    read noise + potential correlations introduced by demosaicing, white
    balance, tone mapping, and compression.

    Detection: combined heuristic — estimates both a Poisson-style
    variance-proportional component *and* an additive Gaussian component via
    block statistics.  The frame is flagged when both components are
    simultaneously present and the overall noise level is non-trivial.

    Removal: Non-Local Means denoising (Buades et al.) which is well suited
    to mixed noise because it leverages patch self-similarity without assuming
    a specific parametric model.
    """

    def __init__(
        self,
        noise_threshold: float = 5.0,
        block_size: int = 16,
        h_luminance: int = 10,
        h_color: int = 10,
    ) -> None:
        self._noise_threshold = noise_threshold
        self._block_size = block_size
        self._h_luminance = h_luminance
        self._h_color = h_color

    @property
    def name(self) -> str:
        return "Real Camera RAW / Output Noise"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_noise_sigma(self, gray: np.ndarray) -> float:
        """Laplacian-based noise sigma estimator (Liu et al.)."""
        kernel = np.array(
            [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32
        )
        filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        return float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(filtered)) / 6.0)

    def _has_poisson_component(self, gray: np.ndarray) -> bool:
        bs = self._block_size
        h, w = gray.shape
        means, variances = [], []
        for i in range(0, h - bs, bs):
            for j in range(0, w - bs, bs):
                block = gray[i : i + bs, j : j + bs].astype(np.float64)
                means.append(block.mean())
                variances.append(block.var())
        if len(means) < 4:
            return False
        means_arr = np.array(means)
        variances_arr = np.array(variances)
        A = np.column_stack([means_arr, np.ones_like(means_arr)])
        coeffs, _, _, _ = np.linalg.lstsq(A, variances_arr, rcond=None)
        return bool(coeffs[0] > 0 and coeffs[1] > 0)

    def _noise_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        kernel = np.array(
            [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32
        )
        residual = np.abs(
            cv2.filter2D(gray.astype(np.float32), -1, kernel)
        )
        threshold = self._noise_threshold * 6.0 / np.sqrt(np.pi / 2.0)
        mask = (residual > threshold).astype(np.uint8) * 255
        return mask

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        sigma = self._estimate_noise_sigma(gray)
        if sigma < self._noise_threshold:
            return False
        return self._has_poisson_component(gray)

    def remove(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(
                frame, None, self._h_luminance, self._h_color, 7, 21
            )
        return cv2.fastNlMeansDenoising(frame, None, self._h_luminance, 7, 21)
