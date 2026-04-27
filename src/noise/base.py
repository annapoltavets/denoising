from abc import ABC, abstractmethod

import cv2
import numpy as np


class Noise(ABC):
    """Abstract base class for every noise type.

    Subclasses must implement :meth:`name`, :meth:`detect`, :meth:`remove`,
    and :meth:`_noise_mask`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this noise type."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> bool:
        """Return *True* if this noise type is present in *frame*.

        Parameters
        ----------
        frame:
            BGR ``uint8`` image ``(H, W, 3)`` or grayscale ``(H, W)``.
        """

    @abstractmethod
    def remove(self, frame: np.ndarray) -> np.ndarray:
        """Return a denoised copy of *frame*.

        Parameters
        ----------
        frame:
            BGR ``uint8`` image ``(H, W, 3)`` or grayscale ``(H, W)``.

        Returns
        -------
        np.ndarray
            Denoised image with the same dtype and shape as *frame*.
        """

    @abstractmethod
    def _noise_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return a binary mask ``(H, W)`` of ``uint8`` where noise pixels are 255.

        Parameters
        ----------
        frame:
            BGR ``uint8`` image ``(H, W, 3)`` or grayscale ``(H, W)``.

        Returns
        -------
        np.ndarray
            Binary mask of shape ``(H, W)`` with dtype ``uint8``:
            255 for noisy pixels, 0 otherwise.
        """

    def color_red(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of *frame* with noise pixels painted red.

        Noise pixels are identified via :meth:`_noise_mask`.  The result is
        always a BGR ``uint8`` image so that red ``[0, 0, 255]`` is well
        defined regardless of the input colour space.

        Parameters
        ----------
        frame:
            BGR ``uint8`` image ``(H, W, 3)`` or grayscale ``(H, W)``.

        Returns
        -------
        np.ndarray
            BGR ``uint8`` image ``(H, W, 3)`` with noisy pixels set to
            pure red ``[0, 0, 255]``.
        """
        # Ensure we work in BGR
        if frame.ndim == 2:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            bgr = frame.copy()

        mask = self._noise_mask(frame)  # (H, W) uint8, values 0 or 255
        bgr[mask == 255] = (0, 0, 255)  # paint red in BGR
        return bgr
