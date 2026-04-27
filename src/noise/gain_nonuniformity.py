"""Gain nonuniformity noise — Ch. 1, Ch. 8."""

import cv2
import numpy as np

from .base import Noise


class GainNonuniformityNoise(Noise):
    """Some pixels, columns, or image regions amplify more than others.

    Noise model: structured multiplicative / heteroskedastic — regional mean
    brightness differs due to analog gain variation in the readout chain.

    Detection: divides the frame into a coarse grid, computes the mean
    intensity of each tile, and evaluates the coefficient of variation of
    those tile means.  Large regional differences indicate gain nonuniformity.

    Removal: normalise each tile so that its mean matches the global mean,
    effectively flat-field-correcting at a coarse spatial scale.
    """

    def __init__(
        self,
        grid_size: int = 4,
        cv_threshold: float = 0.05,
    ) -> None:
        self._grid = grid_size
        self._cv_threshold = cv_threshold

    @property
    def name(self) -> str:
        return "Gain Nonuniformity"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tile_means(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        th, tw = h // self._grid, w // self._grid
        means = []
        for i in range(self._grid):
            for j in range(self._grid):
                tile = gray[i * th : (i + 1) * th, j * tw : (j + 1) * tw]
                means.append(float(tile.mean()))
        return np.array(means)

    def _noise_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        f = gray.astype(np.float32)
        h, w = f.shape
        th, tw = h // self._grid, w // self._grid
        global_mean = f.mean() + 1e-10
        mask = np.zeros((h, w), dtype=np.uint8)
        tile_means = self._tile_means(gray)
        tile_std = float(np.std(tile_means))
        threshold = tile_std * 1.0
        idx = 0
        for i in range(self._grid):
            for j in range(self._grid):
                r0, r1 = i * th, (i + 1) * th
                c0, c1 = j * tw, (j + 1) * tw
                if abs(tile_means[idx] - global_mean) > threshold:
                    mask[r0:r1, c0:c1] = 255
                idx += 1
        return mask

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        means = self._tile_means(gray)
        cv = float(np.std(means) / (np.mean(means) + 1e-10))
        return cv > self._cv_threshold

    def remove(self, frame: np.ndarray) -> np.ndarray:
        def _correct(channel: np.ndarray) -> np.ndarray:
            h, w = channel.shape
            th, tw = h // self._grid, w // self._grid
            f = channel.astype(np.float32)
            global_mean = f.mean() + 1e-10
            for i in range(self._grid):
                for j in range(self._grid):
                    r0, r1 = i * th, (i + 1) * th
                    c0, c1 = j * tw, (j + 1) * tw
                    tile = f[r0:r1, c0:c1]
                    tile_mean = tile.mean() + 1e-10
                    f[r0:r1, c0:c1] = tile * (global_mean / tile_mean)
            return f

        if frame.ndim == 3:
            channels = [_correct(frame[:, :, c]) for c in range(3)]
            return np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
        return np.clip(_correct(frame), 0, 255).astype(np.uint8)
