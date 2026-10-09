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
from scisynth.testing.invariants import check_observation


def obs(values: NDArray[Any], mask: NDArray[Any] | None = None) -> Observation:
    mask = np.ones(values.shape, bool) if mask is None else mask
    v = np.where(mask, values, np.nan)
    v = v[:, None]  # one channel
    return Observation(
        v, {"x": np.arange(v.shape[0], dtype=float)}, mask, truth=v.copy()
    )


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_gaussian_noise_statistics_and_truth_untouched() -> None:
    base = obs(np.zeros(20000))
    out = GaussianNoise(sigma=0.5, mean=1.0)(base, rng())
    check_observation(out)
    assert out.values.mean() == pytest.approx(1.0, abs=0.02)
    assert out.values.std() == pytest.approx(0.5, abs=0.02)
    assert np.array_equal(out.truth, base.truth)  # type: ignore[arg-type]
    assert base.values.max() == 0  # input not mutated
    assert out.meta[-1] == {
        "name": "GaussianNoise",
        "type": "GaussianNoise",
        "params": {"sigma": 0.5, "mean": 1.0},
    }


def test_gaussian_noise_skips_masked() -> None:
    mask = np.arange(10) % 2 == 0
    out = GaussianNoise(1.0)(obs(np.ones(10), mask), rng())
    assert np.isnan(out.values[~mask]).all()
    assert not np.isnan(out.values[mask]).any()
    check_observation(out)


def test_gaussian_noise_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match="sigma"):
        GaussianNoise(-1.0)(obs(np.ones(3)), rng())


def test_poisson_noise_unbiased_and_count_valued() -> None:
    out = PoissonNoise(scale=10.0)(obs(np.full(20000, 3.0)), rng())
    assert out.values.mean() == pytest.approx(3.0, abs=0.02)
    assert np.allclose(out.values * 10, np.round(out.values * 10))


def test_poisson_noise_validation_and_mask() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PoissonNoise()(obs(np.array([1.0, -1.0])), rng())
    with pytest.raises(ValueError, match="scale"):
        PoissonNoise(scale=0)(obs(np.ones(2)), rng())
    mask = np.array([True, False, True])
    out = PoissonNoise()(obs(np.array([2.0, 5.0, 2.0]), mask), rng())
    assert np.isnan(out.values[1])
    check_observation(out)


def test_random_dropout_fraction_and_nan() -> None:
    out = RandomDropout(p=0.3)(obs(np.ones(20000)), rng())
    check_observation(out)
    assert (~out.mask).mean() == pytest.approx(0.3, abs=0.02)
    assert np.isnan(out.values[~out.mask]).all()


def test_random_dropout_extremes_and_validation() -> None:
    assert RandomDropout(p=0.0)(obs(np.ones(50)), rng()).mask.all()
    assert not RandomDropout(p=1.0)(obs(np.ones(50)), rng()).mask.any()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RandomDropout(p=1.5)(obs(np.ones(3)), rng())


def test_dropout_composes_with_existing_mask() -> None:
    mask = np.arange(100) < 50
    out = RandomDropout(p=0.0)(obs(np.ones(100), mask), rng())
    assert np.array_equal(out.mask, mask)


def test_quantize_levels_and_range() -> None:
    values = np.linspace(0, 1, 101)
    out = Quantize(levels=5)(obs(values), rng())
    assert np.allclose(np.unique(out.values), [0, 0.25, 0.5, 0.75, 1.0])
    assert np.abs(out.values[:, 0] - values).max() <= 0.125 + 1e-12


def test_quantize_explicit_range_clips_and_keeps_mask() -> None:
    mask = np.array([True, True, False, True])
    out = Quantize(levels=3, vmin=0.0, vmax=1.0)(
        obs(np.array([-5.0, 0.4, 9.0, 7.0]), mask), rng()
    )
    assert out.values[0] == 0.0 and out.values[3] == 1.0
    assert out.values[1] == 0.5
    assert np.isnan(out.values[2])
    assert np.array_equal(out.mask, mask)


def test_quantize_degenerate_inputs() -> None:
    const = obs(np.full(4, 2.0))
    assert np.array_equal(Quantize()(const, rng()).values, const.values)
    with pytest.raises(ValueError, match="levels"):
        Quantize(levels=1)(const, rng())
    with pytest.raises(ValueError, match="vmin"):
        Quantize(vmin=1.0, vmax=0.0)(const, rng())
    with pytest.raises(ValueError, match="vmin"):
        Quantize(vmin=[0.0, 1.0], vmax=[1.0, 1.0])(const, rng())


def test_stage_logs_resolved_params_not_distributions() -> None:
    from scipy import stats

    out = GaussianNoise(sigma=stats.uniform(1, 1))(obs(np.zeros(3)), rng())
    sigma = out.meta[-1]["params"]["sigma"]
    assert isinstance(sigma, float) and 1 <= sigma <= 2
