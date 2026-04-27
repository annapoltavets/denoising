"""Shot (photon / Poisson) noise — Ch. 1, Ch. 8."""

import cv2
import numpy as np

from .base import Noise


class ShotNoise(Noise):
    """Poisson (photon-counting) noise.

    Detection: fits *variance = α · mean* across non-overlapping blocks.
    A high R² indicates that variance is proportional to mean — the defining
    characteristic of Poisson noise.

    Removal: Anscombe variance-stabilising transform (VST) followed by
    Gaussian denoising and the algebraic inverse Anscombe.
    """

    def __init__(
        self,
        r2_threshold: float = 0.7,
        block_size: int = 16,
        gauss_sigma: float = 1.0,
    ) -> None:
        self._r2_threshold = r2_threshold
        self._block_size = block_size
        self._gauss_sigma = gauss_sigma

    @property
    def name(self) -> str:
        return "Shot Noise / Photon Noise"

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

    def _noise_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        bs = self._block_size
        h, w = gray.shape
        means, variances = self._block_stats(gray)
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(means) < 4:
            return mask
        alpha = np.dot(means, variances) / (np.dot(means, means) + 1e-10)
        # Mark blocks whose variance significantly exceeds α * mean
        idx = 0
        for i in range(0, h - bs, bs):
            for j in range(0, w - bs, bs):
                expected = alpha * means[idx]
                if variances[idx] > expected * 1.5 + 1.0:
                    mask[i : i + bs, j : j + bs] = 255
                idx += 1
        return mask

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
        # Fit var = α * mean (through origin, OLS)
        alpha = np.dot(means, variances) / (np.dot(means, means) + 1e-10)
        residuals = variances - alpha * means
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((variances - variances.mean()) ** 2) + 1e-10
        r2 = 1.0 - ss_res / ss_tot
        return bool(r2 > self._r2_threshold and alpha > 0)

    def remove(self, frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.float32)
        # Anscombe transform: stabilise Poisson variance to ~1
        transformed = 2.0 * np.sqrt(np.maximum(f + 0.375, 0.0))
        # Gaussian denoising in the stabilised domain
        ksize = 0  # auto from sigma
        denoised = cv2.GaussianBlur(
            transformed, (ksize, ksize), self._gauss_sigma
        )
        # Algebraic inverse Anscombe
        restored = (denoised / 2.0) ** 2 - 0.375
        return np.clip(restored, 0, 255).astype(np.uint8)
