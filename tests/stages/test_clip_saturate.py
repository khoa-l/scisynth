import numpy as np
import pytest

from scisynth import (
    Clip,
    Observation,
    Saturate,
)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def obs_of(values: list[float], mask: list[bool] | None = None) -> Observation:
    v = np.asarray(values, dtype=float)
    m = np.ones(v.shape, bool) if mask is None else np.asarray(mask, bool)
    v = np.where(m, v, np.nan)
    v = v[:, None]  # one channel
    return Observation(v, {"x": np.arange(v.shape[0], dtype=float)}, m, truth=v.copy())


# ----------------------------------------------------------------------- Clip


def test_clip_clamps_one_or_both_sides() -> None:
    out = Clip(-1.0, 2.0)(obs_of([-5, -1, 0, 2, 7]), rng())
    assert out.values[:, 0].tolist() == [-1, -1, 0, 2, 2]
    base = obs_of([-5, 0, 5])
    assert Clip(lo=-1.0)(base, rng()).values[:, 0].tolist() == [-1, 0, 5]
    assert Clip(hi=1.0)(base, rng()).values[:, 0].tolist() == [-5, 0, 1]
    assert Clip()(base, rng()).values[:, 0].tolist() == [-5, 0, 5]
    assert Clip(1.0, 1.0)(base, rng()).values[:, 0].tolist() == [1, 1, 1]  # lo == hi


def test_clip_rejects_lo_above_hi() -> None:
    with pytest.raises(ValueError, match="lo must not exceed hi"):
        Clip(2.0, 1.0)(obs_of([0.0]), rng())
    with pytest.raises(ValueError, match="lo must not exceed hi"):
        Clip(np.array([0.0, 5.0]), np.array([1.0, 1.0]))(obs_of([0.0, 0.0]), rng())


# ------------------------------------------------------------------- Saturate


def test_saturate_is_ceiling_times_tanh() -> None:
    # tanh makes it odd, monotonic and slope 1 at zero; only the formula is checked
    v = np.array([-100.0, -2.0, -0.5, 0.0, 0.5, 2.0, 100.0])
    values = Saturate(3.0)(obs_of(v.tolist()), rng()).values[:, 0]
    assert np.allclose(values, 3.0 * np.tanh(v / 3.0))
    assert np.all(np.abs(values) <= 3.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.array([1.0, 0.0])])
def test_saturate_rejects_non_positive_ceiling(bad: object) -> None:
    with pytest.raises(ValueError, match="ceiling must be > 0"):
        Saturate(bad)(obs_of([1.0, 1.0]), rng())  # type: ignore[arg-type]
