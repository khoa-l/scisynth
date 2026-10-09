"""Stubs can be imported and inspected, but raise ``NotImplementedError`` when used.

The list matches the stub table in ``docs/decisions.md``; plot stubs are in
``viz/test_registry.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from scisynth import (
    AnalyticField,
    Domain,
    GridSampler,
    Observation,
    Observer,
    Sampler,
    Stage,
)
from scisynth.adapters.polars_data import from_dataframe
from scisynth.adapters.scipy_stats import DistributionCatalog, DistributionField
from scisynth.adapters.sklearn_data import SklearnDataCatalog
from scisynth.adapters.sklearn_gp import GPField
from scisynth.adapters.xarray_data import from_xarray
from scisynth.io.config import dump_config, load_config
from scisynth.io.observation_io import load_observation, save_observation
from scisynth.latent.base import Kind, LatentGenerator, LatentPlotMixin
from scisynth.latent.catalogs import CatalogLatent, PointProcessLatent
from scisynth.latent.compose import Warp
from scisynth.latent.dataset import DatasetLatent
from scisynth.latent.gridded import GriddedLatent
from scisynth.latent.labels import LabelLatent
from scisynth.latent.timeseries import OrnsteinUhlenbeck, TimeSeriesLatent
from scisynth.samplers.binning import BinSampler
from scisynth.samplers.catalog import CatalogSampler
from scisynth.samplers.points import JitteredSampler
from scisynth.samplers.subset import SubsetSampler
from scisynth.stages.defects.missing import BlockGaps, RegionMask
from scisynth.stages.defects.noise import MultiplicativeNoise, UniformNoise
from scisynth.stages.transforms.crop import Crop
from scisynth.stages.transforms.resample import Regrid

STAGES = [
    UniformNoise,
    MultiplicativeNoise,
    BlockGaps,
    RegionMask,
    Regrid,
    Crop,
]
SAMPLERS = [CatalogSampler, SubsetSampler, BinSampler, JitteredSampler]
GENERATORS = [
    OrnsteinUhlenbeck,
    PointProcessLatent,
    DistributionField,
    DistributionCatalog,
    SklearnDataCatalog,
    GPField,
]


REALIZED_LATENTS = [CatalogLatent, DatasetLatent, GriddedLatent, LabelLatent]
FUNCTIONS = [
    lambda: from_dataframe(None),
    lambda: from_xarray(None),
    lambda: dump_config(Observer(GridSampler(2)), "x.yaml"),
    lambda: load_config("x.yaml"),
    lambda: save_observation(dummy_obs(), "x.npz"),
    lambda: load_observation("x.npz"),
]


def dummy_obs() -> Observation:
    return Observation(np.zeros((3, 1)), {"x": np.arange(3.0)}, np.ones(3, bool))


@pytest.mark.parametrize("cls", STAGES)
def test_stage_stub(cls: type[Stage]) -> None:
    assert issubclass(cls, Stage)
    with pytest.raises(NotImplementedError):
        cls()(dummy_obs(), np.random.default_rng(0))


@pytest.mark.parametrize("cls", SAMPLERS)
def test_sampler_stub(cls: type[Sampler]) -> None:
    @dataclass
    class Fake(LatentPlotMixin):
        domain: Domain
        kind: ClassVar[Kind] = next(iter(cls.accepts))

    with pytest.raises(NotImplementedError):
        cls()(Fake(Domain.from_extents([(0, 1)])), np.random.default_rng(0))


@pytest.mark.parametrize("cls", GENERATORS)
def test_generator_stub(cls: type[LatentGenerator]) -> None:
    with pytest.raises(NotImplementedError):
        cls().realize(0)


@pytest.mark.parametrize("cls", [*REALIZED_LATENTS, TimeSeriesLatent])
def test_realized_latent_stub(cls: type) -> None:
    with pytest.raises(NotImplementedError):
        cls()


def test_warp_stub(field: AnalyticField) -> None:
    with pytest.raises(NotImplementedError):
        Warp(field, None)


@pytest.mark.parametrize("call", FUNCTIONS)
def test_function_stub(call: Callable[[], object]) -> None:
    with pytest.raises(NotImplementedError):
        call()
