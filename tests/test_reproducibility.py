from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from scisynth import (
    AnalyticField,
    GaussianField,
    GaussianNoise,
    GridSampler,
    Observer,
    PointSampler,
    Quantize,
    RandomDropout,
)
from scisynth.testing.invariants import (
    assert_append_stable,
    assert_prefix_stable,
    assert_reproducible,
    observations_equal,
)


def observer() -> Observer:
    return Observer(GridSampler((12, 12)), [GaussianNoise(0.2), RandomDropout(0.1)])


def test_same_seed_identical_different_seed_differs(field: AnalyticField) -> None:
    o = observer()
    assert observations_equal(o.run(field, 1), o.run(field, 1))
    assert not observations_equal(o.run(field, 1), o.run(field, 2))
    assert_reproducible(o, field, 1)


def test_run_does_not_use_global_random_state(field: AnalyticField) -> None:
    np.random.seed(0)  # noqa: NPY002
    a = observer().run(field, 1)
    np.random.seed(999)  # noqa: NPY002
    assert observations_equal(a, observer().run(field, 1))


def test_adding_a_stage_does_not_change_existing_stages(field: AnalyticField) -> None:
    o = observer()
    assert_append_stable(o, GaussianNoise(0.5), field, seed=4)
    assert_append_stable(o, Quantize(8), field, seed=4)


def test_removing_trailing_stages_does_not_change_the_rest(
    field: AnalyticField,
) -> None:
    assert_prefix_stable(observer(), field, seed=4)


def test_stage_randomness_is_independent_of_other_stages_outcomes(
    field: AnalyticField,
) -> None:
    """Noise on entries that survive dropout is the same whatever the dropout rate."""

    def residual(p: float) -> NDArray[Any]:
        o = Observer(GridSampler((20, 20)), [RandomDropout(p), GaussianNoise(0.3)])
        out = o.run(field, seed=8)
        assert out.truth is not None
        return np.where(out.mask, out.values - out.truth, np.nan)

    low, high = residual(0.1), residual(0.6)
    both = ~np.isnan(low) & ~np.isnan(high)
    assert both.sum() > 0
    assert np.array_equal(low[both], high[both])


def test_scipy_distribution_works_as_any_parameter(
    field: AnalyticField, domain: object
) -> None:
    o = Observer(
        PointSampler(stats.randint(20, 40)),
        [
            GaussianNoise(sigma=stats.uniform(0.1, 0.2), mean=stats.norm(0, 0.01)),
            RandomDropout(p=stats.beta(2, 8)),
            Quantize(levels=stats.randint(4, 9)),
        ],
    )
    a, b = o.run(field, 1), o.run(field, 1)
    assert observations_equal(a, b)
    assert 20 <= a.values.size < 40
    assert not observations_equal(a, o.run(field, 2))
    assert 0.1 <= a.meta[1]["params"]["sigma"] <= 0.3


def test_latent_generator_seed_reproducibility_end_to_end(domain: object) -> None:
    gen = GaussianField(domain, length_scale=0.2, resolution=24)  # type: ignore[arg-type]
    o = Observer(GridSampler(10), [GaussianNoise(0.1)])
    a = o.run(gen.realize(1), seed=2)
    assert observations_equal(a, o.run(gen.realize(1), seed=2))
    assert not observations_equal(a, o.run(gen.realize(2), seed=2))
