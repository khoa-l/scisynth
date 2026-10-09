import numpy as np
import pytest

from scisynth import AnalyticField, GridSampler, PointSampler
from scisynth.testing.invariants import check_observation


def test_grid_sampler_layout(field: AnalyticField) -> None:
    obs = GridSampler((5, 9))(field, np.random.default_rng(0))
    check_observation(obs)
    assert obs.values.shape == (5, 9, 1)
    assert set(obs.coords) == {"x", "y"}
    assert obs.coords["x"][0] == 0.0 and obs.coords["x"][-1] == 1.0
    assert obs.coords["y"][-1] == 2.0
    assert obs.mask.all()
    assert np.array_equal(obs.truth, obs.values)  # type: ignore[arg-type]
    assert obs.meta == [
        {
            "name": "GridSampler",
            "type": "GridSampler",
            "params": {"shape": [5, 9], "max_cells": 10_000_000},
        }
    ]


def test_grid_sampler_scalar_shape_repeats(field: AnalyticField) -> None:
    assert GridSampler(6)(field, np.random.default_rng(0)).values.shape == (6, 6, 1)


def test_grid_sampler_wrong_ndim(field: AnalyticField) -> None:
    with pytest.raises(ValueError, match="Expected 2 positive sizes"):
        GridSampler((3, 3, 3))(field, np.random.default_rng(0))


def test_grid_values_match_function(field: AnalyticField) -> None:
    obs = GridSampler((4, 4))(field, np.random.default_rng(0))
    x, y = np.meshgrid(obs.coords["x"], obs.coords["y"], indexing="ij")
    assert np.allclose(
        obs.values[..., 0], 1.0 + np.sin(2 * np.pi * x) * np.cos(np.pi * y)
    )


def test_point_sampler_layout_and_bounds(field: AnalyticField) -> None:
    obs = PointSampler(50)(field, np.random.default_rng(0))
    check_observation(obs)
    assert obs.values.shape == (50, 1)
    assert obs.coords["x"].shape == (50,)
    assert obs.coords["x"].min() >= 0 and obs.coords["y"].max() <= 2.0
    assert np.allclose(
        obs.values[:, 0],
        1.0 + np.sin(2 * np.pi * obs.coords["x"]) * np.cos(np.pi * obs.coords["y"]),
    )


def test_grid_sampler_refuses_an_oversized_grid_before_allocating(
    field: AnalyticField,
) -> None:
    with pytest.raises(ValueError, match=r"shape \(5000, 5000\).*25,000,000 cells"):
        GridSampler((5000, 5000))(field, np.random.default_rng(0))
    # the limit is adjustable
    small = GridSampler((4, 4), max_cells=15)
    with pytest.raises(ValueError, match="over the limit of 15"):
        small(field, np.random.default_rng(0))
    assert GridSampler((4, 4), max_cells=16)(field, np.random.default_rng(0)).mask.all()
