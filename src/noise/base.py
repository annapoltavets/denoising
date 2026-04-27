from abc import ABC, abstractmethod

import numpy as np


class Noise(ABC):
    """Abstract base class for every noise type.

    Subclasses must implement :meth:`name`, :meth:`detect`, and :meth:`remove`.
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
