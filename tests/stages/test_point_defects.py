import numpy as np
import pytest
from scipy import stats

from scisynth import (
    AnalyticField,
    GridSampler,
    Observer,
    PointSampler,
    PositionJitter,
    RandomDropout,
    RandomInsertion,
)
from scisynth.testing.invariants import (
    assert_append_stable,
    assert_reproducible,
    check_observation,
)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_jitter_moves_points_and_leaves_the_rest(field: AnalyticField) -> None:
    obs = PointSampler(50)(field, rng())
    out = PositionJitter(0.1)(obs, rng(1))
    check_observation(out)
    assert np.array_equal(out.values, obs.values)
    assert np.array_equal(out.ids, obs.ids)
    assert np.array_equal(out.truth, obs.truth)  # type: ignore[arg-type]
    shift = out.coords["x"] - obs.coords["x"]
    assert 0.03 < shift.std() < 0.3


def test_jitter_sigma_per_axis_and_defaults(field: AnalyticField) -> None:
    obs = PointSampler(20)(field, rng())
    out = PositionJitter([0.0, 0.5])(obs, rng(1))
    assert np.array_equal(out.coords["x"], obs.coords["x"])
    assert not np.array_equal(out.coords["y"], obs.coords["y"])
    same = PositionJitter()(obs, rng(1))
    assert all(np.array_equal(same.coords[k], obs.coords[k]) for k in obs.coords)


def test_jitter_validates(field: AnalyticField) -> None:
    obs = PointSampler(5)(field, rng())
    with pytest.raises(ValueError, match="one entry per axis"):
        PositionJitter([0.1, 0.1, 0.1])(obs, rng())
    with pytest.raises(ValueError, match="non-negative"):
        PositionJitter(-1.0)(obs, rng())


def test_jitter_distribution_is_logged(field: AnalyticField) -> None:
    out = PositionJitter(stats.uniform(0.1, 0.1))(PointSampler(5)(field, rng()), rng(2))
    assert 0.1 <= out.meta[-1]["params"]["sigma"] <= 0.2


def test_insertion_appends_points_with_fresh_ids(field: AnalyticField) -> None:
    obs = PointSampler(30)(field, rng())
    out = RandomInsertion(7)(obs, rng(1))
    check_observation(out)
    assert out.mask.shape == (37,) and out.values.shape == (37, 1)
    assert np.array_equal(out.ids[:30], obs.ids)
    assert np.array_equal(out.ids[30:], np.arange(30, 37))
    assert np.array_equal(out.values[:30], obs.values)
    assert np.isnan(out.truth[30:]).all()  # type: ignore[index]
    assert np.array_equal(out.truth[:30], obs.truth)  # type: ignore[index,arg-type]
    for name, coord in out.coords.items():
        assert coord[30:].min() >= obs.coords[name].min()
        assert coord[30:].max() <= obs.coords[name].max()
    lo, hi = obs.values.min(), obs.values.max()
    assert out.values[30:].min() >= lo and out.values[30:].max() <= hi


def test_insertion_zero_and_errors(field: AnalyticField) -> None:
    obs = PointSampler(5)(field, rng())
    out = RandomInsertion(0)(obs, rng())
    assert np.array_equal(out.values, obs.values) and out.mask.shape == (5,)
    with pytest.raises(ValueError, match="non-negative"):
        RandomInsertion(-1)(obs, rng())
    dead = obs.replace(
        values=np.full_like(obs.values, np.nan), mask=np.zeros(5, dtype=np.bool_)
    )
    with pytest.raises(ValueError, match="valid location"):
        RandomInsertion(1)(dead, rng())


def test_defects_turn_a_grid_into_points(field: AnalyticField) -> None:
    grid = GridSampler((4, 3))(field, rng())
    jittered = PositionJitter(0.05)(grid, rng(1))
    check_observation(jittered)
    assert jittered.layout == "points" and jittered.spatial_shape == (12,)
    assert np.array_equal(jittered.ids, grid.ids.ravel())
    assert np.array_equal(jittered.values, grid.values.reshape(12, 1))
    grid_x = grid.spatial_coords["x"].ravel()
    assert not np.array_equal(jittered.coords["x"], grid_x)
    inserted = RandomInsertion(5)(grid, rng(1))
    check_observation(inserted)
    assert inserted.spatial_shape == (17,)
    assert np.array_equal(inserted.coords["x"][:12], grid_x)
    assert np.array_equal(inserted.ids[12:], np.arange(12, 17))


def test_diff_sees_moves_and_additions(field: AnalyticField) -> None:
    observer = Observer(
        PointSampler(60),
        [PositionJitter(0.05), RandomDropout(0.2), RandomInsertion(10)],
    )
    trace = observer.trace(field, seed=3)
    jitter, dropout, insertion = trace.diffs
    assert jitter.n_moved == 60 and jitter.n_dropped == 0 and jitter.n_added == 0
    assert dropout.n_dropped > 0 and dropout.n_moved == 0
    assert insertion.n_added == 10 and insertion.n_dropped == 0
    assert np.array_equal(insertion.after.ids[-10:], np.arange(60, 70))


def test_insertion_after_dropout_gets_ids_above_the_dropped_ones(
    field: AnalyticField,
) -> None:
    obs = RandomDropout(0.5)(PointSampler(20)(field, rng()), rng(1))
    out = RandomInsertion(3)(obs, rng(1))
    assert out.ids[-3:].min() == 20
    assert out.mask[-3:].all()


def test_reproducible_and_append_stable(field: AnalyticField) -> None:
    observer = Observer(PointSampler(40), [PositionJitter(0.05), RandomInsertion(5)])
    assert_reproducible(observer, field, 4)
    assert_append_stable(observer, RandomDropout(0.1), field, 4)


def test_previews_use_a_points_demo() -> None:
    pytest.importorskip("plotly")
    assert PositionJitter(0.02).preview().data
    assert RandomInsertion(30).preview().data
