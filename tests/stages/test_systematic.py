import numpy as np
import pytest

from scisynth import (
    AnalyticField,
    CoordinateShift,
    Drift,
    Gain,
    GridSampler,
    Observation,
    PointSampler,
    RandomDropout,
    ValueOffset,
)
from scisynth.testing.invariants import check_observation


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def grid(field: AnalyticField, shape: tuple[int, int] = (5, 4)) -> Observation:
    return GridSampler(shape)(field, rng())


def points(field: AnalyticField, n: int = 12) -> Observation:
    return PointSampler(n)(field, rng())


def masked(obs: Observation) -> Observation:
    return RandomDropout(0.4)(obs, rng(1))


# ------------------------------------------------------- ValueOffset / Gain


def test_value_offset_adds_and_gain_multiplies(field: AnalyticField) -> None:
    obs = grid(field)
    assert np.allclose(ValueOffset(0.5)(obs, rng()).values, obs.values + 0.5)
    assert np.allclose(Gain(1.5)(obs, rng()).values, obs.values * 1.5)


# ---------------------------------------------------------------------- Drift


def test_drift_along_a_named_grid_axis_depends_only_on_that_axis(
    field: AnalyticField,
) -> None:
    obs = grid(field, (5, 4))
    out = Drift(rate=2.0, axis="x")(obs, rng())
    check_observation(out)
    x = obs.coords["x"]
    expected = 2.0 * (x - x.min())[:, None] * np.ones((1, 4))
    assert np.allclose((out.values - obs.values)[..., 0], expected)
    assert (out.values - obs.values)[0].max() == 0  # zero where the axis starts
    y_out = Drift(rate=-1.5, axis="y")(obs, rng())
    y = obs.coords["y"]
    assert np.allclose(
        (y_out.values - obs.values)[..., 0], -1.5 * (y - y.min())[None, :]
    )


def test_drift_along_an_axis_for_point_observations(field: AnalyticField) -> None:
    obs = points(field)
    out = Drift(3.0, axis="y")(obs, rng())
    y = obs.coords["y"]
    assert np.allclose((out.values - obs.values)[:, 0], 3.0 * (y - y.min()))


def test_drift_in_sample_order(field: AnalyticField) -> None:
    obs = grid(field, (3, 4))
    out = Drift(rate=0.5)(obs, rng())
    assert np.allclose(
        (out.values - obs.values)[..., 0], 0.5 * np.arange(12).reshape(3, 4)
    )  # per sample, C order
    pts = points(field)
    assert np.allclose(
        (Drift(1.0)(pts, rng()).values - pts.values)[:, 0], np.arange(len(pts.mask))
    )


def test_drift_ignores_locations_without_a_position(field: AnalyticField) -> None:
    obs = points(field)
    keep = np.arange(obs.size) % 2 == 0
    coords = {k: np.where(keep, c, np.nan) for k, c in obs.coords.items()}
    holey = obs.replace(
        coords=coords, mask=keep, values=np.where(keep[:, None], obs.values, np.nan)
    )
    with np.errstate(all="raise"):
        out = Drift(2.0, axis="x")(holey, rng())
    x = coords["x"][keep]
    assert np.allclose((out.values - holey.values)[keep, 0], 2.0 * (x - x.min()))
    assert np.array_equal(np.isnan(out.values), np.isnan(holey.values))


def test_drift_rejects_a_valid_location_without_a_position(
    field: AnalyticField,
) -> None:
    obs = points(field)
    x = obs.coords["x"].copy()
    x[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Drift(1.0, axis="x")(obs.replace(coords={**obs.coords, "x": x}), rng())


def test_drift_axis_by_name_or_index(field: AnalyticField) -> None:
    obs = grid(field)
    by_name = Drift(1.0, axis="y")(obs, rng()).values
    for index in (1, -1):
        assert np.array_equal(Drift(1.0, axis=index)(obs, rng()).values, by_name)
    with pytest.raises(ValueError, match=r"'z'.*\['x', 'y'\]"):
        Drift(1.0, axis="z")(obs, rng())
    with pytest.raises(ValueError, match="out of range"):
        Drift(1.0, axis=2)(obs, rng())
    with pytest.raises(ValueError, match="out of range"):
        Drift(1.0, axis=-3)(obs, rng())


# ------------------------------------------------------------ CoordinateShift


def test_coordinate_shift_one_number_or_one_per_axis(field: AnalyticField) -> None:
    for obs in (masked(grid(field)), points(field)):
        out = CoordinateShift(0.25)(obs, rng())
        assert all(np.allclose(out.coords[k], obs.coords[k] + 0.25) for k in "xy")
        out = CoordinateShift((0.1, -0.2))(obs, rng())
        assert np.allclose(out.coords["x"], obs.coords["x"] + 0.1)
        assert np.allclose(out.coords["y"], obs.coords["y"] - 0.2)


def test_coordinate_shift_from_a_callable_gives_one_shift_per_axis(
    field: AnalyticField,
) -> None:
    obs = grid(field)
    per_axis = CoordinateShift(lambda r: r.normal(0, 0.1, 2))(obs, rng(5))
    s = per_axis.meta[-1]["params"]["shift"]
    assert len(s) == 2
    assert np.allclose(per_axis.coords["x"], obs.coords["x"] + s[0])


def test_coordinate_shift_rejects_wrong_length(field: AnalyticField) -> None:
    with pytest.raises(ValueError, match="one entry per axis"):
        CoordinateShift((1.0, 2.0, 3.0))(grid(field), rng())
