import json

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
from scisynth.testing.invariants import check_observation, observations_equal


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
    assert [m["name"] for m in obs.meta] == [
        "sampler",
        "gaussian_noise",
        "random_dropout",
        "quantize",
    ]
    assert [m["type"] for m in obs.meta] == [
        "GridSampler",
        "GaussianNoise",
        "RandomDropout",
        "Quantize",
    ]
    clean = GridSampler((16, 16))(field, np.random.default_rng(0)).values
    assert np.array_equal(obs.truth, clean)  # type: ignore[arg-type]
    assert not np.allclose(obs.values[obs.mask], clean[obs.mask])


def test_trace_has_one_entry_per_step(field: AnalyticField) -> None:
    observer = make_observer()
    trace = observer.trace(field, seed=0)
    assert len(trace) == 1 + len(observer.stages)
    assert observations_equal(trace[-1], observer.run(field, seed=0))
    assert [len(t.meta) for t in trace] == [1, 2, 3, 4]


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


def test_to_dict_from_dict_round_trip(field: AnalyticField) -> None:
    observer = make_observer()
    data = json.loads(json.dumps(observer.to_dict()))
    restored = Observer.from_dict(data)
    assert restored == observer
    assert observations_equal(restored.run(field, 3), observer.run(field, 3))
