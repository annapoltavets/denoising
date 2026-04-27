"""Poisson–Gaussian mixed noise — Ch. 1, Ch. 8."""

import cv2
import numpy as np

from .base import Noise


class PoissonGaussianNoise(Noise):
    """Mixed Poisson–Gaussian noise.

    Noise model: *Var[X] = α · E[X] + σ²* where *α* is the Poisson gain
    and *σ²* is the additive Gaussian variance.

    Detection: ordinary-least-squares fit of *variance ~ mean* across
    non-overlapping blocks; both a positive Poisson coefficient *α* and a
    positive intercept *σ²* must be found.

    Removal: Generalised Anscombe VST that jointly stabilises the mixed
    model, followed by Gaussian denoising and the inverse transform.
    """

    def __init__(
        self,
        r2_threshold: float = 0.6,
        block_size: int = 16,
        gauss_sigma: float = 1.0,
    ) -> None:
        self._r2_threshold = r2_threshold
        self._block_size = block_size
        self._gauss_sigma = gauss_sigma
        self._alpha: float = 1.0
        self._sigma2: float = 0.0

    @property
    def name(self) -> str:
        return "Poisson–Gaussian Mixed Noise"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _block_stats(self, gray: np.ndarray):
        bs = self._block_size
        h, w = gray.shape
        means, variances = [], []
        for i in range(0, h - bs, bs):
            for j in range(0, w - bs, bs):
                block = gray[i : i + bs, j : j + bs].astype(np.float64)
                means.append(block.mean())
                variances.append(block.var())
        return np.array(means), np.array(variances)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        means, variances = self._block_stats(gray)
        if len(means) < 4:
            return False
        # OLS: var = alpha * mean + sigma2
        A = np.column_stack([means, np.ones_like(means)])
        coeffs, _, _, _ = np.linalg.lstsq(A, variances, rcond=None)
        alpha, sigma2 = coeffs
        predicted = alpha * means + sigma2
        ss_res = np.sum((variances - predicted) ** 2)
        ss_tot = np.sum((variances - variances.mean()) ** 2) + 1e-10
        r2 = 1.0 - ss_res / ss_tot
        if r2 > self._r2_threshold and alpha > 0 and sigma2 > 0:
            self._alpha = float(alpha)
            self._sigma2 = float(sigma2)
            return True
        return False

    def remove(self, frame: np.ndarray) -> np.ndarray:
        alpha = self._alpha
        sigma2 = self._sigma2
        f = frame.astype(np.float32)
        # Generalised Anscombe VST for Poisson-Gaussian
        inner = np.maximum(alpha * f + (3.0 / 8.0) * alpha ** 2 + sigma2, 0.0)
        transformed = (2.0 / alpha) * np.sqrt(inner)
        ksize = 0
        denoised = cv2.GaussianBlur(
            transformed, (ksize, ksize), self._gauss_sigma
        )
        # Approximate inverse
        restored = ((alpha / 2.0) * denoised) ** 2 / alpha - (3.0 / 8.0) * alpha - sigma2 / alpha
        return np.clip(restored, 0, 255).astype(np.uint8)
