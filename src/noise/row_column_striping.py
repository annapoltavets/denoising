"""Row / column striping noise — Ch. 1 (correlated noise, PSD examples)."""

import cv2
import numpy as np

from .base import Noise


class RowColumnStripingNoise(Noise):
    """Horizontal or vertical banding caused by readout electronics.

    Noise model: colored noise with PSD concentrated at row or column
    frequencies.

    Detection: computes the standard deviation of per-row means and per-column
    means.  Excessive variance in row or column means (compared to a
    shot-noise baseline) indicates striping.

    Removal: subtract the per-row mean deviation and/or per-column mean
    deviation from the frame.
    """

    def __init__(self, stripe_threshold: float = 2.0) -> None:
        self._stripe_threshold = stripe_threshold

    @property
    def name(self) -> str:
        return "Row / Column Striping Noise"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stripe_score(self, gray: np.ndarray):
        row_means = gray.mean(axis=1)
        col_means = gray.mean(axis=0)
        row_score = float(np.std(row_means))
        col_score = float(np.std(col_means))
        return row_score, col_score

    def _noise_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        f = gray.astype(np.float32)
        global_mean = f.mean()
        row_means = f.mean(axis=1, keepdims=True)
        col_means = f.mean(axis=0, keepdims=True)
        # Deviation map: combine row and column bias
        row_dev = np.abs(row_means - global_mean)
        col_dev = np.abs(col_means - global_mean)
        dev_map = (
            np.broadcast_to(row_dev, f.shape) + np.broadcast_to(col_dev, f.shape)
        )
        threshold = float(np.std(f)) * 0.5
        mask = (dev_map > threshold).astype(np.uint8) * 255
        return mask

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        row_score, col_score = self._stripe_score(gray)
        return max(row_score, col_score) > self._stripe_threshold

    def remove(self, frame: np.ndarray) -> np.ndarray:
        def _correct(channel: np.ndarray) -> np.ndarray:
            f = channel.astype(np.float32)
            global_mean = f.mean()
            # Row correction
            row_means = f.mean(axis=1, keepdims=True)
            f -= row_means - global_mean
            # Column correction
            col_means = f.mean(axis=0, keepdims=True)
            f -= col_means - global_mean
            return f

        if frame.ndim == 3:
            channels = [_correct(frame[:, :, c]) for c in range(3)]
            return np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
        return np.clip(_correct(frame), 0, 255).astype(np.uint8)
