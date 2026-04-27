"""Multiplicative noise — Ch. 1 (signal-dependent heteroskedastic models)."""

import cv2
import numpy as np

from .base import Noise


class MultiplicativeNoise(Noise):
    """Noise whose amplitude scales with the signal.

    Noise model: *Y = X · (1 + N)* where *N* is zero-mean Gaussian.

    Detection: in the log domain multiplicative noise becomes additive.  The
    detector checks whether the noise in the log domain is more uniform
    (lower coefficient of variation) than in the linear domain — the hallmark
    of multiplicative noise.

    Removal: log transform → Gaussian denoising → exp.
    """

    def __init__(
        self,
        cv_ratio_threshold: float = 0.6,
        gauss_sigma: float = 1.5,
    ) -> None:
        self._cv_ratio_threshold = cv_ratio_threshold
        self._gauss_sigma = gauss_sigma

    @property
    def name(self) -> str:
        return "Multiplicative Noise"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _high_freq_residual(self, img: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        return img - blurred

    def _coeff_of_variation(self, arr: np.ndarray) -> float:
        mean = np.mean(np.abs(arr)) + 1e-10
        std = np.std(arr)
        return float(std / mean)

    def _noise_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        linear = gray.astype(np.float32) + 1.0
        log_domain = np.log(linear)
        residual = np.abs(self._high_freq_residual(log_domain))
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
        linear = gray.astype(np.float32) + 1.0  # avoid log(0)
        log_domain = np.log(linear)

        cv_linear = self._coeff_of_variation(self._high_freq_residual(linear))
        cv_log = self._coeff_of_variation(self._high_freq_residual(log_domain))

        # In multiplicative noise cv_log < cv_linear
        if cv_linear < 1e-6:
            return False
        return bool((cv_log / cv_linear) < self._cv_ratio_threshold)

    def remove(self, frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.float32) + 1.0
        log_f = np.log(f)
        ksize = 0
        denoised_log = cv2.GaussianBlur(log_f, (ksize, ksize), self._gauss_sigma)
        restored = np.exp(denoised_log) - 1.0
        return np.clip(restored, 0, 255).astype(np.uint8)
