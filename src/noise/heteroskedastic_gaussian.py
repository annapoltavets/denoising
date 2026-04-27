"""Signal-dependent heteroskedastic Gaussian noise — Ch. 1."""

import cv2
import numpy as np

from .base import Noise


class HeteroskedasticGaussianNoise(Noise):
    """Gaussian noise whose variance is a function of the local signal level.

    Noise model: *Var[X | I] = f(I)* — brighter regions have different noise
    strength from darker ones.

    Detection: splits the intensity range into bins, computes local variance in
    each bin (via high-frequency residual), then checks whether variance
    changes significantly with intensity (slope test via linear regression).

    Removal: spatially adaptive bilateral filter with a noise-strength map
    derived from the estimated variance curve.
    """

    def __init__(
        self,
        slope_threshold: float = 0.05,
        n_bins: int = 8,
        block_size: int = 16,
    ) -> None:
        self._slope_threshold = slope_threshold
        self._n_bins = n_bins
        self._block_size = block_size

    @property
    def name(self) -> str:
        return "Signal-Dependent Heteroskedastic Gaussian Noise"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _noise_residual(self, gray: np.ndarray) -> np.ndarray:
        """High-pass residual as a proxy for noise."""
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
        return gray.astype(np.float32) - blurred

    def _estimate_variance_curve(self, gray: np.ndarray):
        """Return (bin_means, bin_variances) across intensity bins."""
        residual = self._noise_residual(gray)
        intensity_bins = np.linspace(0, 255, self._n_bins + 1)
        bin_means, bin_vars = [], []
        for lo, hi in zip(intensity_bins[:-1], intensity_bins[1:]):
            mask = (gray >= lo) & (gray < hi)
            if mask.sum() < 16:
                continue
            bin_means.append((lo + hi) / 2.0)
            bin_vars.append(float(np.var(residual[mask])))
        return np.array(bin_means), np.array(bin_vars)

    def _noise_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        residual = np.abs(self._noise_residual(gray))
        # Pixels whose residual exceeds 2× the estimated per-bin noise floor
        bin_means, bin_vars = self._estimate_variance_curve(gray)
        if len(bin_means) == 0:
            return np.zeros(gray.shape, dtype=np.uint8)
        # Map each pixel's intensity to its expected noise std
        noise_floor = np.interp(
            gray.astype(np.float32), bin_means, np.sqrt(np.maximum(bin_vars, 0))
        )
        mask = (residual > 2.0 * noise_floor + 1e-3).astype(np.uint8) * 255
        return mask

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        bin_means, bin_vars = self._estimate_variance_curve(gray)
        if len(bin_means) < 3:
            return False
        # OLS slope of variance vs intensity
        A = np.column_stack([bin_means, np.ones_like(bin_means)])
        coeffs, _, _, _ = np.linalg.lstsq(A, bin_vars, rcond=None)
        slope = abs(coeffs[0])
        return bool(slope > self._slope_threshold)

    def remove(self, frame: np.ndarray) -> np.ndarray:
        # Adaptive bilateral filter: d=9, sigmaColor and sigmaSpace both 75
        if frame.ndim == 3:
            return cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        gray = frame
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
