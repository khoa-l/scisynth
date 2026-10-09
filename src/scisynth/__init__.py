"""Synthetic scientific data and synthetic observation pipelines."""

from importlib.metadata import PackageNotFoundError, version

from .core import Axis, Domain, Observation, resolve, spawn_rngs
from .latent import (
    AnalyticField,
    FieldLatent,
    GaussianField,
    InterpolatedField,
    Latent,
    LatentGenerator,
    Multichannel,
    Product,
    Sum,
)
from .observer import Observer, StageDiff, StageList, Trace
from .projection import Projection
from .samplers import GridSampler, IncompatibleLatentError, PointSampler, Sampler
from .stages import (
    Clip,
    CoordinateShift,
    Downsample,
    Drift,
    Gain,
    GaussianNoise,
    PoissonNoise,
    PositionJitter,
    Quantize,
    RandomDropout,
    RandomInsertion,
    Saturate,
    Stage,
    ValueOffset,
)

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "AnalyticField",
    "Axis",
    "Clip",
    "CoordinateShift",
    "Domain",
    "Downsample",
    "Drift",
    "FieldLatent",
    "Gain",
    "GaussianField",
    "GaussianNoise",
    "GridSampler",
    "IncompatibleLatentError",
    "InterpolatedField",
    "Latent",
    "LatentGenerator",
    "Multichannel",
    "Observation",
    "Observer",
    "PointSampler",
    "PoissonNoise",
    "PositionJitter",
    "Product",
    "Projection",
    "Quantize",
    "RandomDropout",
    "RandomInsertion",
    "Sampler",
    "Saturate",
    "Stage",
    "StageDiff",
    "StageList",
    "Sum",
    "Trace",
    "ValueOffset",
    "__version__",
    "resolve",
    "spawn_rngs",
]
