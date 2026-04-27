"""Video noise processor.

Reads a video file frame-by-frame, applies every registered noise detector,
removes noise when detected, and writes the result to an output file.
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from .noise import Noise, all_noises

logger = logging.getLogger(__name__)


class Processor:
    """Detect and remove multiple noise types from a video.

    Parameters
    ----------
    noises:
        List of :class:`~src.noise.base.Noise` instances to apply.  When
        *None* (default) all registered noise types are used via
        :func:`~src.noise.all_noises`.

    Examples
    --------
    Process a single video with the default noise pipeline::

        from src.processor import Processor

        proc = Processor()
        proc.process_video("input.mp4", "output.mp4")

    Use a custom subset of noise types::

        from src.processor import Processor
        from src.noise import WhiteGaussianNoise, RowColumnStripingNoise

        proc = Processor(noises=[WhiteGaussianNoise(), RowColumnStripingNoise()])
        proc.process_video("input.mp4", "output.mp4")
    """

    def __init__(self, noises: Optional[List[Noise]] = None) -> None:
        self.noises: List[Noise] = noises if noises is not None else all_noises()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply the noise pipeline to a single BGR frame.

        Parameters
        ----------
        frame:
            ``uint8`` BGR image ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Denoised BGR ``uint8`` image with the same shape as *frame*.
        """
        result = frame.copy()
        for noise in self.noises:
            if noise.detect(result):
                logger.debug("Detected '%s' — removing.", noise.name)
                result = noise.remove(result)
        return result

    def process_video(self, input_path: str, output_path: str) -> None:
        """Read *input_path*, denoise every frame, and write to *output_path*.

        Parameters
        ----------
        input_path:
            Path to the source video file (any format supported by OpenCV).
        output_path:
            Path for the denoised output video (MP4 / MPEG-4 Part 2 container).

        Raises
        ------
        IOError
            If *input_path* cannot be opened.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path!r}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed = self.process_frame(frame)
                out.write(processed)
                frame_idx += 1
                logger.info("Processed frame %d.", frame_idx)
        finally:
            cap.release()
            out.release()

        logger.info(
            "Saved denoised video to %r (%d frame(s)).", output_path, frame_idx
        )
