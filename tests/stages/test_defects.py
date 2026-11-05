from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scisynth import (
    GaussianNoise,
    Observation,
    PoissonNoise,
    Quantize,
    RandomDropout,
)


def obs(values: NDArray[Any], mask: NDArray[Any] | None = None) -> Observation:
    mask = np.ones(values.shape, bool) if mask is None else mask
    v = np.where(mask, values, np.nan)
    v = v[:, None]  # one channel
    return Observation(
        v, {"x": np.arange(v.shape[0], dtype=float)}, mask, truth=v.copy()
    )


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_gaussian_noise_statistics() -> None:
    out = GaussianNoise(sigma=0.5, mean=1.0)(obs(np.zeros(20000)), rng())
    assert out.values.mean() == pytest.approx(1.0, abs=0.02)
    assert out.values.std() == pytest.approx(0.5, abs=0.02)


def test_gaussian_noise_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match="sigma"):
        GaussianNoise(-1.0)(obs(np.ones(3)), rng())


def test_poisson_noise_unbiased_and_count_valued() -> None:
    out = PoissonNoise(scale=10.0)(obs(np.full(20000, 3.0)), rng())
    assert out.values.mean() == pytest.approx(3.0, abs=0.02)
    assert np.allclose(out.values * 10, np.round(out.values * 10))


def test_poisson_noise_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PoissonNoise()(obs(np.array([1.0, -1.0])), rng())
    with pytest.raises(ValueError, match="scale"):
        PoissonNoise(scale=0)(obs(np.ones(2)), rng())


def test_random_dropout_fraction() -> None:
    out = RandomDropout(p=0.3)(obs(np.ones(20000)), rng())
    assert (~out.mask).mean() == pytest.approx(0.3, abs=0.02)


def test_random_dropout_extremes_and_validation() -> None:
    assert RandomDropout(p=0.0)(obs(np.ones(50)), rng()).mask.all()
    assert not RandomDropout(p=1.0)(obs(np.ones(50)), rng()).mask.any()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RandomDropout(p=1.5)(obs(np.ones(3)), rng())


def test_dropout_only_adds_to_what_is_already_dropped() -> None:
    gaps = obs(np.ones(1000), np.arange(1000) % 4 != 0)
    low, high = RandomDropout(0.1)(gaps, rng(5)), RandomDropout(0.5)(gaps, rng(5))
    assert not (low.mask & ~gaps.mask).any()  # masked locations stay masked
    assert not (high.mask & ~low.mask).any()  # dropped at a low p, dropped at a high p


def test_quantize_levels_and_range() -> None:
    values = np.linspace(0, 1, 101)
    out = Quantize(levels=5)(obs(values), rng())
    assert np.allclose(np.unique(out.values), [0, 0.25, 0.5, 0.75, 1.0])
    assert np.abs(out.values[:, 0] - values).max() <= 0.125 + 1e-12


def test_quantize_explicit_range_clips() -> None:
    mask = np.array([True, True, False, True])
    out = Quantize(levels=3, vmin=0.0, vmax=1.0)(
        obs(np.array([-5.0, 0.4, 9.0, 7.0]), mask), rng()
    )
    assert out.values[0] == 0.0 and out.values[3] == 1.0
    assert out.values[1] == 0.5


def test_quantize_one_bound_beyond_the_data_clips_everything_to_it() -> None:
    data = obs(np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.all(Quantize(4, vmin=5.0)(data, rng()).values == 5.0)
    assert np.all(Quantize(4, vmax=-1.0)(data, rng()).values == -1.0)
    # one bound inside the data: the other end still comes from the data
    out = Quantize(4, vmin=1.5)(data, rng())  # levels 1.5, 2.0, 2.5, 3.0
    assert out.values[:, 0].tolist() == [1.5, 1.5, 2.0, 3.0]


def test_quantize_degenerate_inputs() -> None:
    const = obs(np.full(4, 2.0))
    assert np.array_equal(Quantize()(const, rng()).values, const.values)
    with pytest.raises(ValueError, match="levels"):
        Quantize(levels=1)(const, rng())
    with pytest.raises(ValueError, match="vmin"):
        Quantize(vmin=1.0, vmax=0.0)(const, rng())
    with pytest.raises(ValueError, match="vmin"):
        Quantize(vmin=[0.0, 1.0], vmax=[1.0, 1.0])(const, rng())
