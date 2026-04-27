"""Scaled Poisson noise — Ch. 1 "Scaled Poisson Distribution Family"."""

import cv2
import numpy as np

from .base import Noise


class ScaledPoissonNoise(Noise):
    """Poisson noise scaled by a sensor gain factor *α*.

    Noise model: *Var[X] = α · E[X]*.

    Detection: same block-statistics regression as :class:`ShotNoise` but
    additionally checks that the estimated scale *α* deviates from 1 (pure
    Poisson) to confirm gain scaling.

    Removal: Generalised Anscombe VST for scaled Poisson, then Gaussian
    denoising, then inverse transform.
    """

    def __init__(
        self,
        r2_threshold: float = 0.7,
        alpha_min: float = 0.1,
        block_size: int = 16,
        gauss_sigma: float = 1.0,
    ) -> None:
        self._r2_threshold = r2_threshold
        self._alpha_min = alpha_min
        self._block_size = block_size
        self._gauss_sigma = gauss_sigma
        self._alpha: float = 1.0  # estimated gain

    @property
    def name(self) -> str:
        return "Scaled Poisson Noise"

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
        alpha = np.dot(means, variances) / (np.dot(means, means) + 1e-10)
        residuals = variances - alpha * means
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((variances - variances.mean()) ** 2) + 1e-10
        r2 = 1.0 - ss_res / ss_tot
        if r2 > self._r2_threshold and alpha >= self._alpha_min:
            self._alpha = float(alpha)
            return True
        return False

    def remove(self, frame: np.ndarray) -> np.ndarray:
        alpha = self._alpha
        f = frame.astype(np.float32)
        # Generalised Anscombe: (2/α) * sqrt(α*x + 3*α²/8)
        inner = np.maximum(alpha * f + 0.375 * alpha ** 2, 0.0)
        transformed = (2.0 / alpha) * np.sqrt(inner)
        ksize = 0
        denoised = cv2.GaussianBlur(
            transformed, (ksize, ksize), self._gauss_sigma
        )
        # Inverse: x = ((α/2)*y)^2/α - 3*α/8  simplified
        restored = ((alpha / 2.0) * denoised) ** 2 / alpha - 0.375 * alpha
        return np.clip(restored, 0, 255).astype(np.uint8)
