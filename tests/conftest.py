from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from scisynth import (
    AnalyticField,
    Clip,
    CoordinateShift,
    Domain,
    Downsample,
    Drift,
    Gain,
    GaussianField,
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
from scisynth.latent.base import Kind

STAGES: list[Stage] = [
    GaussianNoise(0.3),
    PoissonNoise(2.0),
    RandomDropout(0.2),
    Quantize(levels=8, vmin=0.0, vmax=1.0),
    Clip(-1.0, None),
    Saturate(3.0),
    ValueOffset(0.5),
    Gain(2.0),
    Drift(0.1, axis="x"),
    CoordinateShift((0.1, 0.2)),
    Downsample([2, 4]),
    PositionJitter(0.05),
    RandomInsertion(5),
]


@pytest.fixture(params=STAGES, ids=lambda stage: type(stage).__name__)
def stage(request: pytest.FixtureRequest) -> Stage:
    """Each implemented stage in turn."""
    stage: Stage = request.param
    return stage


@dataclass
class FakeCatalog:
    """A latent of a kind that field samplers must reject."""

    domain: Domain
    kind: ClassVar[Kind] = "catalog"


@pytest.fixture
def domain() -> Domain:
    return Domain.from_extents([(0.0, 1.0), (0.0, 2.0)])


@pytest.fixture
def field(domain: Domain) -> AnalyticField:
    return AnalyticField(
        lambda x, y: 1.0 + np.sin(2 * np.pi * x) * np.cos(np.pi * y), domain
    )


@pytest.fixture
def line() -> AnalyticField:
    return AnalyticField(lambda x: x**2, Domain.from_extents([(0.0, 1.0)]))


@pytest.fixture
def grf_latent(domain: Domain) -> object:
    return GaussianField(domain, length_scale=0.2, resolution=32).realize(0)


@pytest.fixture
def catalog(domain: Domain) -> FakeCatalog:
    return FakeCatalog(domain)
