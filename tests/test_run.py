import numpy as np

from scisynth import (
    AnalyticField,
    GaussianNoise,
    GridSampler,
    Observation,
    Observer,
    PointSampler,
    spawn_rngs,
)
from scisynth.testing.invariants import observations_equal


def same_data(a: Observation, b: Observation) -> bool:
    return (
        np.array_equal(a.values, b.values, equal_nan=True)
        and np.array_equal(a.mask, b.mask)
        and a.coords.keys() == b.coords.keys()
        and all(np.array_equal(a.coords[k], b.coords[k]) for k in a.coords)
    )


def test_sampler_run_matches_a_one_sampler_observer(field: AnalyticField) -> None:
    sampler = PointSampler(25)
    direct, via_observer = sampler.run(field, 3), Observer(sampler).run(field, 3)
    assert same_data(direct, via_observer)
    assert direct.meta[-1]["name"] == "PointSampler"  # standalone: the class name
    assert via_observer.meta[-1]["name"] == "sampler"
    assert direct.meta[-1]["params"] == via_observer.meta[-1]["params"]


def test_stage_run_is_the_call_with_a_spawned_stream(field: AnalyticField) -> None:
    obs = GridSampler((6, 6)).run(field, 0)
    stage = GaussianNoise(0.3)
    assert observations_equal(stage.run(obs, 5), stage(obs, spawn_rngs(5, 1)[0]))
