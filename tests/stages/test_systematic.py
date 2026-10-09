import numpy as np
import pytest
from scipy import stats

from scisynth import (
    AnalyticField,
    CoordinateShift,
    Drift,
    Gain,
    GridSampler,
    Observation,
    Observer,
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


def test_value_offset_adds_to_valid_entries_only(field: AnalyticField) -> None:
    obs = masked(grid(field))
    out = ValueOffset(0.5)(obs, rng())
    check_observation(out)
    assert np.allclose(out.values[obs.mask], obs.values[obs.mask] + 0.5)
    assert np.isnan(out.values[~obs.mask]).all()
    assert np.array_equal(out.mask, obs.mask)
    assert np.array_equal(out.truth, obs.truth)  # type: ignore[arg-type]


def test_gain_multiplies_valid_entries_only(field: AnalyticField) -> None:
    obs = masked(grid(field))
    out = Gain(1.5)(obs, rng())
    check_observation(out)
    assert np.allclose(out.values[obs.mask], obs.values[obs.mask] * 1.5)
    assert np.isnan(out.values[~obs.mask]).all()
    assert np.array_equal(out.truth, obs.truth)  # type: ignore[arg-type]


def test_defaults_are_identities(field: AnalyticField) -> None:
    obs = masked(points(field))
    for stage in (ValueOffset(), Gain(), Drift(), CoordinateShift()):
        out = stage(obs, rng())
        assert np.array_equal(out.values, obs.values, equal_nan=True)
        assert all(np.array_equal(out.coords[k], obs.coords[k]) for k in obs.coords)


def test_inputs_are_not_mutated(field: AnalyticField) -> None:
    obs = grid(field)
    values, coords = obs.values.copy(), {k: v.copy() for k, v in obs.coords.items()}
    for stage in (
        ValueOffset(1.0),
        Gain(2.0),
        Drift(1.0, axis="x"),
        CoordinateShift(1.0),
    ):
        stage(obs, rng())
    assert np.array_equal(obs.values, values)
    assert all(np.array_equal(obs.coords[k], coords[k]) for k in coords)


def test_value_offset_and_gain_accept_arrays(field: AnalyticField) -> None:
    obs = grid(field, (3, 4))
    per_column = np.arange(4.0)
    assert np.allclose(
        ValueOffset(per_column)(obs, rng()).values, obs.values + per_column
    )
    assert np.allclose(Gain(per_column)(obs, rng()).values, obs.values * per_column)


def test_distribution_gives_one_shared_draw_that_is_logged(
    field: AnalyticField,
) -> None:
    obs = grid(field)
    out = ValueOffset(stats.norm(0, 1))(obs, rng(3))
    diff = out.values - obs.values
    assert np.allclose(diff, diff.flat[0])  # the same shift everywhere
    assert out.meta[-1]["params"]["value"] == pytest.approx(diff.flat[0])
    again = ValueOffset(stats.norm(0, 1))(obs, rng(3))
    assert np.array_equal(out.values, again.values)
    other = ValueOffset(stats.norm(0, 1))(obs, rng(4))
    assert not np.array_equal(out.values, other.values)

    gout = Gain(stats.uniform(0.5, 1.0))(obs, rng(3))
    ratio = gout.values[obs.values != 0] / obs.values[obs.values != 0]
    assert np.allclose(ratio, ratio[0]) and 0.5 <= ratio[0] <= 1.5


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


def test_drift_axis_by_index_matches_name(field: AnalyticField) -> None:
    obs = grid(field)
    assert np.array_equal(
        Drift(1.0, axis=1)(obs, rng()).values, Drift(1.0, axis="y")(obs, rng()).values
    )
    assert np.array_equal(
        Drift(1.0, axis=-1)(obs, rng()).values, Drift(1.0, axis="y")(obs, rng()).values
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


def test_drift_one_dimensional(line: AnalyticField) -> None:
    obs = GridSampler(6)(line, rng())
    out = Drift(2.0, axis="x")(obs, rng())
    assert np.allclose((out.values - obs.values)[:, 0], 2.0 * obs.coords["x"])
    assert np.allclose(
        (Drift(2.0)(obs, rng()).values - obs.values)[:, 0], 2.0 * np.arange(6)
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


def test_drift_skips_masked_entries(field: AnalyticField) -> None:
    obs = masked(grid(field, (6, 6)))
    out = Drift(1.0, axis="x")(obs, rng())
    assert np.isnan(out.values[~obs.mask]).all()
    assert np.array_equal(out.mask, obs.mask)
    check_observation(out)


def test_drift_rejects_unknown_axis(field: AnalyticField) -> None:
    obs = grid(field)
    with pytest.raises(ValueError, match=r"'z'.*\['x', 'y'\]"):
        Drift(1.0, axis="z")(obs, rng())
    with pytest.raises(ValueError, match="out of range"):
        Drift(1.0, axis=2)(obs, rng())
    with pytest.raises(ValueError, match="out of range"):
        Drift(1.0, axis=-3)(obs, rng())


def test_drift_rate_distribution_is_one_shared_draw(field: AnalyticField) -> None:
    obs = grid(field)
    out = Drift(stats.norm(0, 1), axis="x")(obs, rng(2))
    rate = out.meta[-1]["params"]["rate"]
    x = obs.coords["x"]
    assert np.allclose((out.values - obs.values)[..., 0], rate * (x - x.min())[:, None])


# ------------------------------------------------------------ CoordinateShift


def test_coordinate_shift_shifts_every_axis_by_a_number(field: AnalyticField) -> None:
    obs = grid(field)
    out = CoordinateShift(0.25)(obs, rng())
    check_observation(out)
    assert np.allclose(out.coords["x"], obs.coords["x"] + 0.25)
    assert np.allclose(out.coords["y"], obs.coords["y"] + 0.25)
    assert list(out.coords) == ["x", "y"]


def test_coordinate_shift_per_axis_leaves_values_mask_and_truth_alone(
    field: AnalyticField,
) -> None:
    obs = masked(grid(field))
    out = CoordinateShift((0.1, -0.2))(obs, rng())
    assert np.allclose(out.coords["x"], obs.coords["x"] + 0.1)
    assert np.allclose(out.coords["y"], obs.coords["y"] - 0.2)
    assert np.array_equal(out.values, obs.values, equal_nan=True)
    assert np.array_equal(out.mask, obs.mask)
    assert np.array_equal(out.truth, obs.truth)  # type: ignore[arg-type]
    assert np.allclose(obs.coords["x"], grid(field).coords["x"])  # input intact


def test_coordinate_shift_on_points_and_array_shift(field: AnalyticField) -> None:
    obs = points(field)
    out = CoordinateShift(np.array([1.0, 2.0]))(obs, rng())
    assert np.allclose(out.coords["x"], obs.coords["x"] + 1.0)
    assert np.allclose(out.coords["y"], obs.coords["y"] + 2.0)
    check_observation(out)


def test_coordinate_shift_shift_from_distribution_or_callable(
    field: AnalyticField,
) -> None:
    obs = grid(field)
    out = CoordinateShift(stats.norm(0, 0.1))(obs, rng(5))
    shift = out.meta[-1]["params"]["shift"]
    assert np.allclose(out.coords["y"], obs.coords["y"] + shift)
    per_axis = CoordinateShift(lambda r: r.normal(0, 0.1, 2))(obs, rng(5))
    s = per_axis.meta[-1]["params"]["shift"]
    assert len(s) == 2
    assert np.allclose(per_axis.coords["x"], obs.coords["x"] + s[0])


def test_coordinate_shift_rejects_wrong_length(field: AnalyticField) -> None:
    with pytest.raises(ValueError, match="one entry per axis"):
        CoordinateShift((1.0, 2.0, 3.0))(grid(field), rng())


# ------------------------------------------------------------- pipeline-level


def test_stages_compose_in_an_observer(field: AnalyticField) -> None:
    o = Observer(
        GridSampler((8, 8)),
        [Gain(1.1), ValueOffset(0.2), Drift(0.5, axis="x"), CoordinateShift(0.05)],
    )
    obs = o.run(field, 0)
    check_observation(obs)
    clean = GridSampler((8, 8))(field, rng())
    x = clean.coords["x"]
    expected = clean.values * 1.1 + 0.2 + 0.5 * (x - x.min())[:, None, None]
    assert np.allclose(obs.values, expected)
