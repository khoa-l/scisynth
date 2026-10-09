"""Samplers: how a latent is queried to produce a first observation.

Its public names are importable from the package itself, for example
``from scisynth.samplers import GridSampler, PointSampler``.
"""

from .base import IncompatibleLatentError, Sampler
from .grid import GridSampler
from .points import PointSampler

__all__ = [
    "Sampler",
    "GridSampler",
    "PointSampler",
    "IncompatibleLatentError",
]
