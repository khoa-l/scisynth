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
from scisynth.testing.invariants import check_observation, observations_equal


def same_data(a: Observation, b: Observation) -> bool:
    return (
        np.array_equal(a.values, b.values, equal_nan=True)
        and np.array_equal(a.mask, b.mask)
        and a.coords.keys() == b.coords.keys()
        and all(np.array_equal(a.coords[k], b.coords[k]) for k in a.coords)
    )


def test_sampler_run_is_the_call_with_a_spawned_stream(field: AnalyticField) -> None:
    sampler = PointSampler(25)
    via_call = sampler(field, spawn_rngs(7, 1)[0])
    assert observations_equal(sampler.run(field, 7), via_call)
    check_observation(sampler.run(field, 7))


def test_sampler_run_is_reproducible_and_seed_dependent(field: AnalyticField) -> None:
    sampler = PointSampler(25)
    assert observations_equal(sampler.run(field, 1), sampler.run(field, 1))
    assert not observations_equal(sampler.run(field, 1), sampler.run(field, 2))
    assert sampler.run(field).values.shape == (25, 1)  # seed=None is allowed


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
    assert observations_equal(stage.run(obs, 5), stage.run(obs, 5))
    assert not observations_equal(stage.run(obs, 5), stage.run(obs, 6))
    assert len(stage.run(obs, 5).meta) == len(obs.meta) + 1
    assert stage.run(obs).values.shape == obs.values.shape  # seed=None is allowed


def test_stage_run_does_not_mutate_its_input(field: AnalyticField) -> None:
    obs = GridSampler((6, 6)).run(field, 0)
    before = obs.values.copy()
    GaussianNoise(1.0).run(obs, 1)
    assert np.array_equal(obs.values, before)


def test_stage_run_is_not_the_streams_it_gets_in_an_observer(
    field: AnalyticField,
) -> None:
    observer = Observer(GridSampler((6, 6)), [GaussianNoise(0.3)])
    in_pipeline = observer.run(field, 4)
    standalone = GaussianNoise(0.3).run(observer.with_stages([]).run(field, 4), 4)
    assert not np.array_equal(in_pipeline.values, standalone.values)  # stream 1 vs 0
