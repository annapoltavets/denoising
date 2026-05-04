"""Colored / spatially correlated noise — Ch. 1."""

import cv2
import numpy as np

from .base import Noise


class ColoredCorrelatedNoise(Noise):
    """Noise with a non-flat power spectral density (PSD).

    Noise model: stationary correlated noise; spatial covariance is
    non-diagonal, i.e., neighbouring pixel errors are correlated and the PSD
    is not uniform across frequencies.

    Detection: computes the 2-D PSD of the high-frequency residual and
    measures its flatness using the spectral flatness measure (geometric mean /
    arithmetic mean of the PSD).  A flatness below *flatness_threshold*
    indicates significant spectral coloring.

    Removal: frequency-domain Wiener filter using the estimated PSD as the
    noise model; this suppresses frequency bands where the noise dominates.
    """

    def __init__(self, flatness_threshold: float = 0.3) -> None:
        self._flatness_threshold = flatness_threshold

    @property
    def name(self) -> str:
        return "Colored / Spatially Correlated Noise"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _high_freq_residual(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.5)
        return gray.astype(np.float32) - blurred

    def _spectral_flatness(self, residual: np.ndarray) -> float:
        F = np.fft.fft2(residual)
        psd = np.abs(F) ** 2 + 1e-10
        log_geo = np.mean(np.log(psd))
        log_arith = np.log(np.mean(psd) + 1e-10)
        return float(np.exp(log_geo - log_arith))

    def _wiener_channel(self, channel: np.ndarray) -> np.ndarray:
        f = channel.astype(np.float32)
        F = np.fft.fft2(f)
        psd = np.abs(F) ** 2
        noise_psd = np.median(psd)
        wiener_gain = psd / (psd + noise_psd + 1e-10)
        restored = np.fft.ifft2(F * wiener_gain).real
        return np.clip(restored, 0, 255).astype(np.float32)

    def _noise_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        residual = np.abs(self._high_freq_residual(gray))
        threshold = float(np.std(residual)) * 2.0
        mask = (residual > threshold).astype(np.uint8) * 255
        return mask

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        residual = self._high_freq_residual(gray)
        flatness = self._spectral_flatness(residual)
        return flatness < self._flatness_threshold

    def remove(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            channels = [self._wiener_channel(frame[:, :, c]) for c in range(3)]
            return np.stack(channels, axis=-1).astype(np.uint8)
        return self._wiener_channel(frame).astype(np.uint8)
