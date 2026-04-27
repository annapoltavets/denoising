"""Temporal block-outlier noise — multi-frame statistical detector."""

from __future__ import annotations

from collections import deque
from typing import Sequence

import cv2
import numpy as np

from .base import Noise


class TemporalBlockOutlierNoise(Noise):
    """Noise detected as per-block outliers across a sequence of frames.

    The frame is divided into non-overlapping spatial blocks.  For each block
    position the distribution of per-block mean intensities is built from a
    rolling buffer of recent frames.  A block in the *current* frame is
    classified as noisy when its mean lies more than ``zscore_threshold``
    standard deviations away from the buffer mean — i.e., it is a statistical
    outlier in the temporal distribution of that block.

    This detects transient localised artefacts (dust, particle hits, brief
    over-exposure patches, …) that affect individual spatial blocks for only
    a small number of frames.

    Detection interface
    -------------------
    * :meth:`detect` — standard single-frame interface (accumulates an
      internal frame buffer; returns *False* until ``warmup_frames`` have been
      seen).
    * :meth:`detect_sequence` — batch interface that accepts an explicit list
      of frames; computes block distributions across the whole sequence and
      reports whether the **last** frame contains outlier blocks.

    Removal
    -------
    Outlier block pixels are replaced with the temporal median of that block
    across the rolling buffer, restoring the expected local appearance.

    Parameters
    ----------
    block_size:
        Side length (pixels) of each square spatial block.
    zscore_threshold:
        How many standard deviations from the buffer mean a block must be to
        be considered an outlier.
    min_outlier_fraction:
        Minimum fraction of blocks that must be outliers for the whole frame
        to be flagged as noisy.
    buffer_size:
        Maximum number of past frames kept in the rolling buffer.
    warmup_frames:
        Minimum number of frames required before detection is activated
        (single-frame :meth:`detect` path only).
    """

    def __init__(
        self,
        block_size: int = 8,
        zscore_threshold: float = 3.0,
        min_outlier_fraction: float = 0.01,
        buffer_size: int = 20,
        warmup_frames: int = 5,
    ) -> None:
        self._block_size = block_size
        self._zscore_threshold = zscore_threshold
        self._min_outlier_fraction = min_outlier_fraction
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._warmup = warmup_frames

    # ------------------------------------------------------------------
    # Name
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Temporal Block-Outlier Noise"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        return (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        )

    def _block_means(self, gray: np.ndarray) -> np.ndarray:
        """Return a 2-D array of per-block mean values for *gray*.

        Shape: ``(n_row_blocks, n_col_blocks)``
        """
        h, w = gray.shape
        bs = self._block_size
        nr = h // bs
        nc = w // bs
        f = gray[:nr * bs, :nc * bs].astype(np.float32)
        # Reshape to (nr, bs, nc, bs) then average over the two block axes
        blocks = f.reshape(nr, bs, nc, bs)
        return blocks.mean(axis=(1, 3))  # (nr, nc)

    def _outlier_block_mask(
        self,
        buffer_gray: list[np.ndarray],
        target_gray: np.ndarray,
    ) -> np.ndarray:
        """Return a full-resolution binary mask (0/255) of outlier blocks.

        Parameters
        ----------
        buffer_gray:
            List of grayscale float32 frames forming the reference buffer.
        target_gray:
            The grayscale frame to test.

        Returns
        -------
        np.ndarray
            ``uint8`` mask of shape ``(H, W)``; 255 where the block is an
            outlier, 0 elsewhere.
        """
        h, w = target_gray.shape
        bs = self._block_size
        nr = h // bs
        nc = w // bs

        if nr == 0 or nc == 0 or len(buffer_gray) < 2:
            return np.zeros((h, w), dtype=np.uint8)

        # Stack block means across the buffer: shape (n_frames, nr, nc)
        stacked = np.stack(
            [self._block_means(g) for g in buffer_gray], axis=0
        )  # (T, nr, nc)

        buf_mean = stacked.mean(axis=0)   # (nr, nc)
        buf_std = stacked.std(axis=0)     # (nr, nc)

        target_means = self._block_means(target_gray)  # (nr, nc)

        # z-score of the target block relative to the buffer distribution
        zscores = np.abs(target_means - buf_mean) / (buf_std + 1e-6)  # (nr, nc)
        outlier_blocks = zscores > self._zscore_threshold  # (nr, nc) bool

        # Expand block mask back to pixel resolution
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:nr * bs, :nc * bs] = np.kron(
            outlier_blocks.astype(np.uint8), np.ones((bs, bs), dtype=np.uint8)
        ) * 255
        return mask

    def _buffer_as_gray_list(self) -> list[np.ndarray]:
        return [self._to_gray(f).astype(np.float32) for f in self._buffer]

    # ------------------------------------------------------------------
    # _noise_mask (required by base)
    # ------------------------------------------------------------------

    def _noise_mask(self, frame: np.ndarray) -> np.ndarray:
        """Binary block-outlier mask for *frame* given the current buffer.

        Returns an all-zero mask when the buffer has fewer than 2 frames.
        """
        buf = self._buffer_as_gray_list()
        target_gray = self._to_gray(frame).astype(np.float32)
        return self._outlier_block_mask(buf, target_gray)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> bool:
        """Update the frame buffer and return *True* if the frame is noisy.

        The frame is added to the rolling buffer **before** detection so that
        its block statistics contribute to future calls.  Detection is
        suppressed until ``warmup_frames`` have been accumulated.
        """
        self._buffer.append(frame)
        if len(self._buffer) < max(self._warmup, 2):
            return False

        buf = self._buffer_as_gray_list()
        target_gray = self._to_gray(frame).astype(np.float32)
        mask = self._outlier_block_mask(buf, target_gray)

        h, w = target_gray.shape
        bs = self._block_size
        nr, nc = h // bs, w // bs
        total_blocks = nr * nc
        if total_blocks == 0:
            return False

        # Count non-zero blocks in the mask (each outlier block fills bs*bs pixels)
        n_outlier = int((mask == 255).sum()) // (bs * bs)
        return (n_outlier / total_blocks) >= self._min_outlier_fraction

    def detect_sequence(
        self,
        frames: Sequence[np.ndarray],
    ) -> bool:
        """Detect block-outlier noise in the **last** frame of *frames*.

        The entire provided sequence is used as the reference distribution.
        The last frame is tested against the block statistics computed from
        the preceding frames.  This method does **not** modify the internal
        rolling buffer.

        Parameters
        ----------
        frames:
            Ordered sequence of BGR ``uint8`` or grayscale ``uint8`` frames,
            all with the same spatial dimensions.  At least 3 frames are
            required; otherwise *False* is returned.

        Returns
        -------
        bool
            *True* if the last frame contains outlier blocks relative to the
            rest of the sequence.
        """
        if len(frames) < 3:
            return False

        gray_frames = [
            self._to_gray(f).astype(np.float32) for f in frames
        ]
        # Reference: all frames except the last
        reference = gray_frames[:-1]
        target = gray_frames[-1]

        h, w = target.shape
        bs = self._block_size
        nr, nc = h // bs, w // bs
        total_blocks = nr * nc
        if total_blocks == 0:
            return False

        mask = self._outlier_block_mask(reference, target)
        n_outlier = int((mask == 255).sum()) // (bs * bs)
        return (n_outlier / total_blocks) >= self._min_outlier_fraction

    def remove(self, frame: np.ndarray) -> np.ndarray:
        """Replace outlier blocks with their temporal median across the buffer.

        If the buffer is empty the original frame is returned unchanged.
        """
        if len(self._buffer) < 2:
            return frame

        buf = self._buffer_as_gray_list()
        target_gray = self._to_gray(frame).astype(np.float32)
        mask = self._outlier_block_mask(buf, target_gray)

        if not mask.any():
            return frame

        h, w = target_gray.shape
        bs = self._block_size
        nr, nc = h // bs, w // bs

        # Temporal median stack for each channel (or single grayscale channel)
        if frame.ndim == 3:
            channels_out = []
            for c in range(frame.shape[2]):
                buf_channel = np.stack(
                    [
                        (
                            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                            if f.ndim == 3
                            else f
                        ).astype(np.float32)
                        for f in self._buffer
                    ],
                    axis=0,
                )  # (T, H, W) — we use per-channel later; grayscale proxy for now
                # Use BGR channel from buffer
                buf_c = np.stack(
                    [f[:, :, c].astype(np.float32) for f in self._buffer],
                    axis=0,
                )  # (T, H, W)
                temporal_median = np.median(buf_c, axis=0)  # (H, W)
                channel = frame[:, :, c].astype(np.float32)
                channel[:nr * bs, :nc * bs][
                    mask[:nr * bs, :nc * bs] == 255
                ] = temporal_median[:nr * bs, :nc * bs][
                    mask[:nr * bs, :nc * bs] == 255
                ]
                channels_out.append(channel)
            result = np.stack(channels_out, axis=-1)
        else:
            buf_stack = np.stack(
                [
                    (
                        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                        if f.ndim == 3
                        else f
                    ).astype(np.float32)
                    for f in self._buffer
                ],
                axis=0,
            )  # (T, H, W)
            temporal_median = np.median(buf_stack, axis=0)  # (H, W)
            result = frame.astype(np.float32)
            result[:nr * bs, :nc * bs][
                mask[:nr * bs, :nc * bs] == 255
            ] = temporal_median[:nr * bs, :nc * bs][
                mask[:nr * bs, :nc * bs] == 255
            ]

        return np.clip(result, 0, 255).astype(np.uint8)
