import numpy as np
import pytest

from scisynth import (
    AnalyticField,
    GaussianNoise,
    GridSampler,
    Observer,
    Quantize,
    RandomDropout,
)
from scisynth.testing.invariants import check_observation


def make_observer() -> Observer:
    return Observer(
        sampler=GridSampler((16, 16)),
        stages=[GaussianNoise(0.1), RandomDropout(0.2), Quantize(8)],
    )


def test_end_to_end_run(field: AnalyticField) -> None:
    obs = make_observer().run(field, seed=0)
    check_observation(obs)
    assert obs.values.shape == (16, 16, 1)
    assert 0 < (~obs.mask).sum() < obs.mask.size
    clean = GridSampler((16, 16))(field, np.random.default_rng(0)).values
    assert np.array_equal(obs.truth, clean)  # type: ignore[arg-type]
    assert not np.allclose(obs.values[obs.mask], clean[obs.mask])


def test_with_stages_keeps_the_sampler_and_runs_a_prefix(field: AnalyticField) -> None:
    observer = make_observer()
    sub = observer.with_stages(observer.stages[:2])
    assert sub.sampler is observer.sampler
    assert sub.stages == observer.stages[:2]
    assert len(sub.run(field, 0).meta) == 3
    renamed = Observer(("grid", GridSampler(4)), observer.stages)
    assert renamed.with_stages([]).names == ["grid"]


def test_observer_validates_its_parts() -> None:
    with pytest.raises(TypeError, match="Sampler"):
        Observer(GaussianNoise())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Stage"):
        Observer(GridSampler(4), [GridSampler(4)])  # type: ignore[list-item]
