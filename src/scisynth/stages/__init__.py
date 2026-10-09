"""Stages: transformations applied to an observation after sampling.

Its public names are importable from the package itself, for example
``from scisynth.stages import GaussianNoise, Quantize``.
"""

from .base import Stage
from .defects.missing import RandomDropout
from .defects.noise import GaussianNoise, PoissonNoise
from .defects.points import PositionJitter, RandomInsertion
from .defects.quantization import Clip, Quantize, Saturate
from .defects.systematic import CoordinateShift, Drift, Gain, ValueOffset
from .transforms.resample import Downsample

__all__ = [
    "Stage",
    "GaussianNoise",
    "PoissonNoise",
    "Quantize",
    "Clip",
    "Saturate",
    "ValueOffset",
    "Gain",
    "Drift",
    "CoordinateShift",
    "RandomDropout",
    "PositionJitter",
    "RandomInsertion",
    "Downsample",
]
