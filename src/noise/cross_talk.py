"""Cross-talk (correlated / colored) noise — Ch. 1, Ch. 8."""

import cv2
import numpy as np

from .base import Noise


class CrossTalkNoise(Noise):
    """Neighboring-pixel cross-contamination producing spatially correlated noise.

    Noise model: colored (non-white) additive noise with non-flat power
    spectral density caused by charge coupling between sensing elements.

    Detection: computes the normalised autocorrelation of the high-frequency
    residual at lag-1 (horizontal and vertical).  A significant nonzero lag-1
    correlation indicates spatial dependence consistent with cross-talk.

    Removal: 2-D Wiener-style filter (via frequency-domain division by the
    estimated PSD) to whiten and suppress the correlated component.
    """

    def __init__(self, corr_threshold: float = 0.15) -> None:
        self._corr_threshold = corr_threshold

    @property
    def name(self) -> str:
        return "Cross-Talk Noise"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _noise_residual(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.5)
        return gray.astype(np.float32) - blurred

    def _lag1_correlation(self, arr: np.ndarray) -> float:
        flat = arr.ravel()
        if flat.size < 2:
            return 0.0
        centered = flat - flat.mean()
        var = np.var(centered) + 1e-10
        corr_h = float(np.mean(arr[:, :-1] * arr[:, 1:]) / var)
        corr_v = float(np.mean(arr[:-1, :] * arr[1:, :]) / var)
        return max(abs(corr_h), abs(corr_v))

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        residual = self._noise_residual(gray)
        return self._lag1_correlation(residual) > self._corr_threshold

    def remove(self, frame: np.ndarray) -> np.ndarray:
        """Frequency-domain Wiener filter to suppress correlated noise."""

        def _wiener_channel(channel: np.ndarray) -> np.ndarray:
            f = channel.astype(np.float32)
            F = np.fft.fft2(f)
            psd = np.abs(F) ** 2
            # Estimate noise PSD as the median (robust to signal)
            noise_psd = np.median(psd)
            wiener = psd / (psd + noise_psd + 1e-10)
            restored = np.fft.ifft2(F * wiener).real
            return np.clip(restored, 0, 255).astype(np.float32)

        if frame.ndim == 3:
            channels = [_wiener_channel(frame[:, :, c]) for c in range(3)]
            return np.stack(channels, axis=-1).astype(np.uint8)
        return _wiener_channel(frame).astype(np.uint8)
