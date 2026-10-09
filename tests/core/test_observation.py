from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scisynth import (
    AnalyticField,
    GaussianNoise,
    GridSampler,
    Observation,
    Observer,
    PointSampler,
    RandomDropout,
)
from scisynth.testing.invariants import (
    InvariantError,
    check_observation,
    observations_equal,
)


def grid_obs(shape: tuple[int, ...] = (3, 4), d: int = 2) -> Observation:
    values = np.arange(np.prod(shape) * d, dtype=float).reshape(*shape, d)
    coords: dict[str, NDArray[Any]] = {
        f"a{i}": np.arange(n, dtype=float) for i, n in enumerate(shape)
    }
    return Observation(values, coords, np.ones(shape, bool))


def test_channel_axis_is_always_last() -> None:
    obs = grid_obs((3, 4), d=2)
    assert obs.shape == (3, 4, 2)
    assert obs.spatial_shape == (3, 4)
    assert obs.n_channels == 2
    assert obs.valid.shape == (3, 4, 1)
    assert grid_obs((3, 4), d=1).n_channels == 1
    check_observation(obs)


def test_scalar_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="channel axis"):
        Observation(np.asarray(1.0), {}, np.asarray(True))


def test_layout_ignores_the_channel_axis() -> None:
    assert grid_obs((3, 4), d=2).layout == "grid"
    # as many channels as points does not turn points into a grid
    pts = Observation(
        np.zeros((3, 3)),
        {"x": np.arange(3.0), "y": np.arange(3.0)},
        np.ones(3, bool),
    )
    assert pts.layout == "points"
    check_observation(pts)


def test_invariants_reject_a_mask_with_a_channel_axis() -> None:
    obs = grid_obs()
    bad = obs.replace(mask=np.ones(obs.shape, bool))
    with pytest.raises(InvariantError, match="mask shape"):
        check_observation(bad)


def test_ids_default_to_storage_order_and_have_the_spatial_shape() -> None:
    obs = grid_obs((3, 4), d=2)
    assert obs.ids.shape == (3, 4) and obs.ids.dtype == np.int64
    assert obs.ids.ravel().tolist() == list(range(12))
    check_observation(obs)


def test_samplers_assign_ids_that_stages_keep(field: AnalyticField) -> None:
    grid = GridSampler((3, 4))(field, np.random.default_rng(0))
    assert grid.ids.tolist() == np.arange(12).reshape(3, 4).tolist()  # C order
    assert PointSampler(7)(field, np.random.default_rng(0)).ids.tolist() == list(
        range(7)
    )

    trace = Observer(PointSampler(30), [GaussianNoise(0.1), RandomDropout(0.3)]).trace(
        field, 1
    )
    assert all(np.array_equal(o.ids, trace[0].ids) for o in trace)  # a drop is a mask


def test_ids_with_the_wrong_shape_are_rejected_and_bad_ids_are_caught() -> None:
    obs = grid_obs((3, 4), d=1)
    with pytest.raises(ValueError, match="ids shape"):
        obs.replace(ids=np.arange(5))
    duplicated = obs.replace(ids=np.zeros((3, 4), dtype=int))
    with pytest.raises(InvariantError, match="unique"):
        check_observation(duplicated)


def test_valid_locations_need_finite_coordinates() -> None:
    obs = Observation(
        np.zeros((3, 1)), {"x": np.array([0.0, np.nan, 2.0])}, np.ones(3, dtype=bool)
    )
    with pytest.raises(InvariantError, match="finite coordinates"):
        check_observation(obs)
    masked = obs.replace(
        mask=np.array([True, False, True]), values=np.array([[0.0], [np.nan], [0.0]])
    )
    check_observation(masked)  # no position is fine where there is no data


def test_observations_with_different_ids_are_not_equal() -> None:
    a = grid_obs((2, 2), d=1)
    assert observations_equal(a, a.replace())
    assert not observations_equal(a, a.replace(ids=a.ids + 1))


def test_has_position_marks_locations_with_finite_coordinates() -> None:
    from scisynth.core._coords import has_position

    x = np.array([0.0, np.nan, 2.0, np.inf])
    y = np.array([0.0, 1.0, np.nan, 3.0])
    points = Observation(np.zeros((4, 1)), {"x": x, "y": y}, np.ones(4, dtype=bool))
    assert has_position(points).tolist() == [True, False, False, False]
    grid = grid_obs((3, 4), d=1)
    assert has_position(grid).shape == (3, 4) and has_position(grid).all()
    nan_row = grid.replace(coords={**grid.coords, "a0": np.array([0.0, np.nan, 2.0])})
    assert has_position(nan_row).tolist() == [[True] * 4, [False] * 4, [True] * 4]
