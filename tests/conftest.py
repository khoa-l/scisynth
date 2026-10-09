from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from scisynth import AnalyticField, Domain, GaussianField
from scisynth.latent.base import Kind


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
