"""Noise-type implementations.

Exports
-------
All concrete :class:`~src.noise.base.Noise` subclasses plus a convenience
factory :func:`all_noises` that returns a ready-to-use list ordered from
most common / fast-to-detect to more specialised types.
"""

from .base import Noise
from .white_gaussian import WhiteGaussianNoise
from .shot_noise import ShotNoise
from .scaled_poisson import ScaledPoissonNoise
from .poisson_gaussian import PoissonGaussianNoise
from .heteroskedastic_gaussian import HeteroskedasticGaussianNoise
from .multiplicative import MultiplicativeNoise
from .thermal import ThermalNoise
from .cross_talk import CrossTalkNoise
from .fixed_pattern import FixedPatternNoise
from .row_column_striping import RowColumnStripingNoise
from .quantization import QuantizationNoise
from .clipping_saturation import ClippingSaturationNoise
from .prnu import PRNUNoise
from .gain_nonuniformity import GainNonuniformityNoise
from .flicker import FlickerNoise
from .lens_flare import LensFlareNoise
from .colored_correlated import ColoredCorrelatedNoise
from .real_camera_raw import RealCameraRawNoise

__all__ = [
    "Noise",
    "WhiteGaussianNoise",
    "ShotNoise",
    "ScaledPoissonNoise",
    "PoissonGaussianNoise",
    "HeteroskedasticGaussianNoise",
    "MultiplicativeNoise",
    "ThermalNoise",
    "CrossTalkNoise",
    "FixedPatternNoise",
    "RowColumnStripingNoise",
    "QuantizationNoise",
    "ClippingSaturationNoise",
    "PRNUNoise",
    "GainNonuniformityNoise",
    "FlickerNoise",
    "LensFlareNoise",
    "ColoredCorrelatedNoise",
    "RealCameraRawNoise",
    "all_noises",
]


def all_noises() -> list["Noise"]:
    """Return one instance of every registered noise type.

    The order matters: faster / simpler detectors run first so that the
    Processor can bail out quickly on clean frames before reaching heavier
    detectors.
    """
    return [
        # Single-frame, statistical detectors (fast)
        RowColumnStripingNoise(),
        ClippingSaturationNoise(),
        QuantizationNoise(),
        WhiteGaussianNoise(),
        ShotNoise(),
        ScaledPoissonNoise(),
        PoissonGaussianNoise(),
        HeteroskedasticGaussianNoise(),
        MultiplicativeNoise(),
        ThermalNoise(),
        CrossTalkNoise(),
        ColoredCorrelatedNoise(),
        GainNonuniformityNoise(),
        LensFlareNoise(),
        # Temporal detectors (need frame history; maintain internal state)
        FixedPatternNoise(),
        PRNUNoise(),
        FlickerNoise(),
        # Catch-all for mixed real-camera noise (most expensive; runs last)
        RealCameraRawNoise(),
    ]
