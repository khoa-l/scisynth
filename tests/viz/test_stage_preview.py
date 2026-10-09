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
from scisynth.viz import plot, plot_observations

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


def test_preview_is_repeatable_and_seedable() -> None:
    a = colors(GaussianNoise(0.3).preview(), 1)
    b = colors(GaussianNoise(0.3).preview(), 1)
    c = colors(GaussianNoise(0.3).preview(seed=1), 1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_the_stage_plot_itself_needs_an_observation() -> None:
    with pytest.raises(TypeError):
        plot(GaussianNoise(0.2))


def test_axis_titles_and_layout_kwargs() -> None:
    fig = GaussianNoise().preview(height=321)
    assert fig.layout.height == 321
    for suffix in ("", "2"):  # every panel titles its own axes
        assert fig.layout["xaxis" + suffix].title.text == "x"
        assert fig.layout["yaxis" + suffix].title.text == "y"


def test_preview_does_not_modify_the_observation(field: AnalyticField) -> None:
    obs = GridSampler((6, 6))(field, np.random.default_rng(0))
    before = obs.values.copy()
    GaussianNoise(1.0).preview(obs)
    assert np.array_equal(obs.values, before)


def test_empty_panels_rejected() -> None:
    with pytest.raises(ValueError, match="nothing"):
        plot_observations([])
