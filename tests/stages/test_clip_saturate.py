import numpy as np
import pytest
from scipy import stats

from scisynth import (
    AnalyticField,
    Clip,
    Gain,
    GridSampler,
    Observation,
    Observer,
    PointSampler,
    RandomDropout,
    Saturate,
    Stage,
)
from scisynth.testing.invariants import check_observation


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def obs_of(values: list[float], mask: list[bool] | None = None) -> Observation:
    v = np.asarray(values, dtype=float)
    m = np.ones(v.shape, bool) if mask is None else np.asarray(mask, bool)
    v = np.where(m, v, np.nan)
    v = v[:, None]  # one channel
    return Observation(v, {"x": np.arange(v.shape[0], dtype=float)}, m, truth=v.copy())


# ----------------------------------------------------------------------- Clip


def test_clip_clamps_both_sides() -> None:
    out = Clip(-1.0, 2.0)(obs_of([-5, -1, 0, 2, 7]), rng())
    assert out.values[:, 0].tolist() == [-1, -1, 0, 2, 2]
    check_observation(out)


def test_clip_one_sided_and_unbounded() -> None:
    base = obs_of([-5, 0, 5])
    assert Clip(lo=-1.0)(base, rng()).values[:, 0].tolist() == [-1, 0, 5]
    assert Clip(hi=1.0)(base, rng()).values[:, 0].tolist() == [-5, 0, 1]
    assert Clip()(base, rng()).values[:, 0].tolist() == [-5, 0, 5]
    assert Clip(1.0, 1.0)(base, rng()).values[:, 0].tolist() == [1, 1, 1]  # lo == hi


def test_clip_leaves_masked_entries_mask_and_truth_alone() -> None:
    base = obs_of([-5, 7, 9, -9], [True, False, True, True])
    out = Clip(-1.0, 1.0)(base, rng())
    assert np.isnan(out.values[1])
    assert out.values[[0, 2, 3], 0].tolist() == [-1, 1, -1]
    assert np.array_equal(out.mask, base.mask)
    assert np.array_equal(out.truth, base.truth, equal_nan=True)  # type: ignore[arg-type]
    assert base.values[0] == -5  # input not mutated
    check_observation(out)


def test_clip_bounds_can_be_arrays() -> None:
    # Arrays align with the values' shape (n, d), so a column is one bound per sample.
    out = Clip(lo=np.zeros((3, 1)), hi=np.array([[1.0], [2.0], [3.0]]))(
        obs_of([9, 9, 9]), rng()
    )
    assert out.values[:, 0].tolist() == [1, 2, 3]


def test_clip_bounds_can_differ_per_channel() -> None:
    base = obs_of([9.0, -9.0])
    two = base.replace(
        values=np.repeat(base.values, 2, axis=1),
        truth=np.repeat(base.values, 2, axis=1),
    )
    out = Clip(lo=np.array([-1.0, -5.0]), hi=np.array([1.0, 5.0]))(two, rng())
    assert out.values.tolist() == [[1.0, 5.0], [-1.0, -5.0]]
    check_observation(out)


def test_clip_rejects_lo_above_hi() -> None:
    with pytest.raises(ValueError, match="lo must not exceed hi"):
        Clip(2.0, 1.0)(obs_of([0.0]), rng())
    with pytest.raises(ValueError, match="lo must not exceed hi"):
        Clip(np.array([0.0, 5.0]), np.array([1.0, 1.0]))(obs_of([0.0, 0.0]), rng())


def test_clip_bound_distribution_is_one_logged_draw() -> None:
    out = Clip(hi=stats.uniform(0.0, 1.0))(obs_of([5.0, 6.0, 7.0]), rng(2))
    hi = out.meta[-1]["params"]["hi"]
    assert 0.0 <= hi <= 1.0
    assert np.allclose(out.values, hi)
    assert out.meta[-1]["params"]["lo"] is None


# ------------------------------------------------------------------- Saturate


def test_saturate_matches_the_formula_and_stays_below_the_ceiling() -> None:
    v = np.array([-100.0, -2.0, -0.5, 0.0, 0.5, 2.0, 100.0])
    out = Saturate(3.0)(obs_of(v.tolist()), rng())
    values = out.values[:, 0]
    assert np.allclose(values, 3.0 * np.tanh(v / 3.0))
    assert np.all(np.abs(values) <= 3.0)
    assert np.all(np.abs(values[1:-1]) < 3.0)


def test_saturate_is_identity_near_zero_and_monotonic_and_odd() -> None:
    small = np.array([-1e-3, 0.0, 1e-3])
    out = Saturate(1.0)(obs_of(small.tolist()), rng()).values[:, 0]
    assert np.allclose(out, small, atol=1e-8)
    v = np.linspace(-10, 10, 101)
    sat = Saturate(2.0)(obs_of(v.tolist()), rng()).values[:, 0]
    assert np.all(np.diff(sat) > 0)
    assert np.allclose(sat, -sat[::-1])
    assert np.all(np.sign(sat) == np.sign(v))


def test_saturate_compresses_more_the_lower_the_ceiling() -> None:
    base = obs_of([0.8])
    low = Saturate(1.0)(base, rng()).values[0]
    high = Saturate(100.0)(base, rng()).values[0]
    assert low < high <= 0.8
    assert high == pytest.approx(0.8, abs=1e-3)


def test_saturate_skips_masked_and_keeps_mask_and_truth() -> None:
    base = obs_of([5.0, 6.0, 7.0], [True, False, True])
    out = Saturate(1.0)(base, rng())
    assert np.isnan(out.values[1])
    assert np.array_equal(out.mask, base.mask)
    assert np.array_equal(out.truth, base.truth, equal_nan=True)  # type: ignore[arg-type]
    check_observation(out)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.array([1.0, 0.0])])
def test_saturate_rejects_non_positive_ceiling(bad: object) -> None:
    with pytest.raises(ValueError, match="ceiling must be > 0"):
        Saturate(bad)(obs_of([1.0, 1.0]), rng())  # type: ignore[arg-type]


@pytest.mark.parametrize("ceiling", [np.array([[1.0], [10.0]]), [[1.0], [10.0]]])
def test_saturate_array_ceiling(ceiling: object) -> None:
    out = Saturate(ceiling)(obs_of([50.0, 50.0]), rng())  # type: ignore[arg-type]
    assert out.values[0, 0] == pytest.approx(1.0) and 9.9 < out.values[1, 0] <= 10.0


def test_saturate_ceiling_distribution_is_one_logged_draw() -> None:
    drawn = Saturate(stats.uniform(1.0, 2.0))(obs_of([100.0]), rng(4))
    ceiling = drawn.meta[-1]["params"]["ceiling"]
    assert 1.0 <= ceiling <= 3.0
    assert drawn.values[0] == pytest.approx(ceiling)


# ------------------------------------------------------------- pipeline-level


def test_both_work_on_grids_points_and_in_an_observer(field: AnalyticField) -> None:
    o = Observer(
        GridSampler((8, 8)),
        [Gain(5.0), Saturate(2.0), RandomDropout(0.2), Clip(-1.0, 1.0)],
    )
    assert o.stage_names == ["gain", "saturate", "random_dropout", "clip"]
    obs = o.run(field, 0)
    check_observation(obs)
    assert np.nanmax(np.abs(obs.values)) <= 1.0
    pts = Observer(PointSampler(30), [Saturate(0.5), Clip(hi=0.1)]).run(field, 0)
    check_observation(pts)
    assert np.nanmax(pts.values) <= 0.1


def test_default_demo_shows_the_effect() -> None:
    for stage in (Clip(-0.5, 0.5), Saturate(0.5)):
        assert isinstance(stage, Stage)
        demo = stage.demo_observation()
        out = stage(demo, rng())
        assert np.abs(out.values).max() < np.abs(demo.values).max()
