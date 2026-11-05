from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scisynth import (
    AnalyticField,
    GaussianNoise,
    GridSampler,
)
from scisynth.stages.demo import smooth_grid

pytest.importorskip("plotly")


def colors(fig: object, i: int) -> NDArray[Any]:
    return np.asarray(fig.data[i].marker.color)  # type: ignore[attr-defined]


def test_preview_without_obs_runs_on_the_demo_signal() -> None:
    fig = GaussianNoise(0.3).preview()
    assert len(fig.data) == 2
    assert [a.text for a in fig.layout.annotations] == ["before", "after GaussianNoise"]
    before, after = colors(fig, 0), colors(fig, 1)
    assert np.allclose(before, smooth_grid().values.ravel())
    residual = after - before
    assert residual.std() == pytest.approx(0.3, abs=0.04)
    assert abs(residual.mean()) < 0.06


def test_preview_with_obs_uses_it_instead_of_the_demo(field: AnalyticField) -> None:
    obs = GridSampler((5, 4))(field, np.random.default_rng(0))
    fig = GaussianNoise(0.1).preview(obs)
    assert len(fig.data[0].x) == 20
    assert np.allclose(colors(fig, 0), obs.values.ravel())
