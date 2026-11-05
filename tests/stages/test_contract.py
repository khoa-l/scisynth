"""What every stage promises, checked once for each stage.

Behavior a single stage adds on top of this is tested in that stage's own file.
"""

import copy

import numpy as np
import pytest
from scipy import stats

from scisynth import (
    Clip,
    Drift,
    Gain,
    GaussianNoise,
    Observation,
    PoissonNoise,
    Quantize,
    RandomDropout,
    Saturate,
    Stage,
    ValueOffset,
)
from scisynth.stages.demo import smooth_grid
from scisynth.testing.invariants import check_observation, observations_equal

VALUE_STAGES: list[Stage] = [
    GaussianNoise(0.3),
    PoissonNoise(2.0),
    Quantize(8),
    Clip(-0.5, 0.5),
    Saturate(0.5),
    ValueOffset(0.5),
    Gain(2.0),
    Drift(0.1, axis="x"),
]


def with_gaps(stage: Stage) -> Observation:
    """Return the stage's demo input with some locations masked out."""
    return RandomDropout(0.3).run(stage.demo_observation(), 1)


def test_a_stage_keeps_its_input_and_returns_a_valid_observation(stage: Stage) -> None:
    obs = with_gaps(stage)
    before = copy.deepcopy(obs)
    out = stage.run(obs, 2)
    assert observations_equal(obs, before)
    check_observation(out)
    assert out.meta[:-1] == obs.meta and out.meta[-1]["type"] == type(stage).__name__
    assert observations_equal(stage.run(obs, 2), out)


@pytest.mark.parametrize("stage", VALUE_STAGES, ids=lambda s: type(s).__name__)
def test_value_stages_change_valid_values_only(stage: Stage) -> None:
    obs = with_gaps(stage)
    out = stage.run(obs, 2)
    assert np.array_equal(out.mask, obs.mask) and np.array_equal(out.ids, obs.ids)
    assert np.isnan(out.values[~obs.mask]).all()
    assert np.array_equal(out.truth, obs.truth, equal_nan=True)  # type: ignore[arg-type]
    assert all(np.array_equal(out.coords[k], obs.coords[k]) for k in obs.coords)


def test_a_distribution_is_drawn_once_per_call_and_logged() -> None:
    obs = smooth_grid()
    out = ValueOffset(stats.norm(0, 1)).run(obs, 3)
    drawn = out.meta[-1]["params"]["value"]
    assert isinstance(drawn, float)
    assert np.allclose(out.values - obs.values, drawn)
