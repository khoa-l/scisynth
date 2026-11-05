"""Stubs can be imported and inspected, but raise ``NotImplementedError`` when used.

The list matches the stub table in ``docs/decisions.md``; plot stubs are in
``viz/test_registry.py``.
"""

from collections.abc import Callable

import numpy as np
import pytest

from scisynth import GridSampler, Observation, Observer, Stage
from scisynth.adapters.polars_data import from_dataframe
from scisynth.adapters.scipy_stats import DistributionCatalog, DistributionField
from scisynth.adapters.sklearn_data import SklearnDataCatalog
from scisynth.adapters.sklearn_gp import GPField
from scisynth.adapters.xarray_data import from_xarray
from scisynth.io.config import dump_config, load_config
from scisynth.io.observation_io import load_observation, save_observation
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

CLASSES = [
    UniformNoise,
    MultiplicativeNoise,
    BlockGaps,
    RegionMask,
    Regrid,
    Crop,
    CatalogSampler,
    SubsetSampler,
    BinSampler,
    JitteredSampler,
    OrnsteinUhlenbeck,
    PointProcessLatent,
    DistributionField,
    DistributionCatalog,
    SklearnDataCatalog,
    GPField,
    CatalogLatent,
    DatasetLatent,
    GriddedLatent,
    LabelLatent,
    TimeSeriesLatent,
    Warp,
]
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


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__name__)
def test_a_planned_class_raises_when_created(cls: type) -> None:
    with pytest.raises(NotImplementedError, match=cls.__name__):
        cls()


def test_a_config_naming_a_planned_class_cannot_be_loaded() -> None:
    with pytest.raises(NotImplementedError, match="UniformNoise"):
        Stage.from_dict({"type": "UniformNoise", "params": {}})


@pytest.mark.parametrize("call", FUNCTIONS)
def test_function_stub(call: Callable[[], object]) -> None:
    with pytest.raises(NotImplementedError):
        call()
