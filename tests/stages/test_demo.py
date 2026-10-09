from dataclasses import dataclass

import numpy as np
import pytest

from scisynth import (
    Clip,
    CoordinateShift,
    Downsample,
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
from scisynth.stages.demo import bumps, ramp, smooth_grid
from scisynth.testing.invariants import check_observation

DEMO_SHAPE = (48, 48)  # default size of smooth_grid / bumps / ramp

IMPLEMENTED: list[Stage] = [
    GaussianNoise(0.2),
    PoissonNoise(2.0),
    Quantize(4),
    RandomDropout(0.3),
    ValueOffset(0.5),
    Gain(2.0),
    Drift(0.1),
    CoordinateShift(0.05),
    Clip(-0.5, 0.5),
    Saturate(1.0),
    Downsample(4),
]


def assert_demo_grid(demo: Observation) -> None:
    check_observation(demo)
    assert demo.layout == "grid" and len(demo.coords) == 2
    assert demo.mask.all() and demo.truth is not None


def test_smooth_grid_is_a_deterministic_signed_2d_grid() -> None:
    demo = smooth_grid()
    assert_demo_grid(demo)
    assert demo.spatial_shape == DEMO_SHAPE
    assert -1.0 <= demo.values.min() < -0.9 and 0.9 < demo.values.max() <= 1.0
    assert np.array_equal(demo.values, smooth_grid().values)
    assert smooth_grid((5, 7)).spatial_shape == (5, 7)


def test_bumps_and_ramp() -> None:
    b, r = bumps(), ramp()
    for demo in (b, r):
        assert_demo_grid(demo)
        assert demo.spatial_shape == DEMO_SHAPE
    assert b.values.min() >= 0 and 3.5 < b.values.max() <= 6.0
    assert r.values.min() == 0.0 and r.values.max() == 1.0
    assert np.all(np.diff(r.values, axis=0) > 0)
    assert np.all(np.diff(r.values, axis=1) > 0)


def test_stages_declare_their_own_demo_observation() -> None:
    assert np.array_equal(PoissonNoise().demo_observation().values, bumps().values)
    assert np.array_equal(Quantize(4).demo_observation().values, ramp().values)
    for default in (GaussianNoise(), RandomDropout(0.3)):
        assert np.array_equal(default.demo_observation().values, smooth_grid().values)


def test_subclasses_inherit_and_can_override_the_declaration() -> None:
    @dataclass
    class MyPoisson(PoissonNoise):
        pass

    @dataclass
    class FlatNoise(GaussianNoise):
        def demo_observation(self) -> Observation:
            return ramp((4, 4))

    assert np.array_equal(MyPoisson().demo_observation().values, bumps().values)
    assert FlatNoise().demo_observation().spatial_shape == (4, 4)
    assert (
        GaussianNoise().demo_observation().spatial_shape == DEMO_SHAPE
    )  # parent unchanged


@pytest.mark.parametrize("stage", IMPLEMENTED, ids=lambda s: type(s).__name__)
def test_every_implemented_stage_runs_on_its_demo(stage: Stage) -> None:
    demo = stage.demo_observation()
    assert_demo_grid(demo)
    out = stage(demo, np.random.default_rng(0))
    check_observation(out)
    assert out.n_channels == demo.n_channels


def test_demo_observation_does_not_need_plotting_libraries() -> None:
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys\n"
        "from scisynth import GaussianNoise\n"
        "GaussianNoise().demo_observation()\n"
        "assert 'plotly' not in sys.modules\n"
        "assert 'scisynth.viz' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={"PYTHONPATH": str(src), "PATH": ""},
    )
