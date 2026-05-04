"""Lens flare as noise / unwanted optical artifact — Ch. 8."""

import cv2
import numpy as np

from .base import Noise


class LensFlareNoise(Noise):
    """Veiling glare, bright streaks, or halos from internal lens reflections.

    Noise model: structured artifact — a bright, spatially concentrated
    superposition on the scene that does not correspond to real scene content.

    Detection: identifies abnormally bright blobs or streaks: pixels above a
    high-brightness threshold are clustered; if a coherent bright region
    exceeding *area_threshold* pixels is found the frame is flagged.

    Removal: the detected flare mask is inpainted using the Telea algorithm
    so that the affected region is reconstructed from its surroundings.
    """

    def __init__(
        self,
        bright_threshold: int = 240,
        area_threshold: int = 200,
        dilate_ksize: int = 5,
        inpaint_radius: int = 5,
    ) -> None:
        self._bright_threshold = bright_threshold
        self._area_threshold = area_threshold
        self._dilate_ksize = dilate_ksize
        self._inpaint_radius = inpaint_radius

    @property
    def name(self) -> str:
        return "Lens Flare / Unwanted Optical Artifact"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flare_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )
        _, binary = cv2.threshold(gray, self._bright_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((self._dilate_ksize, self._dilate_ksize), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        return dilated.astype(np.uint8)

    def _noise_mask(self, frame: np.ndarray) -> np.ndarray:
        return self._flare_mask(frame)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        mask = self._flare_mask(frame)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for label_idx in range(1, n_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= self._area_threshold:
                return True
        return False

    def remove(self, frame: np.ndarray) -> np.ndarray:
        mask = self._flare_mask(frame)
        return cv2.inpaint(frame, mask, self._inpaint_radius, cv2.INPAINT_TELEA)
