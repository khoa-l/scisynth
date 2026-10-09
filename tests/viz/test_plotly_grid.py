from typing import Any

import numpy as np
import pytest

from scisynth import (
    AnalyticField,
    Domain,
    GridSampler,
    Observation,
    PointSampler,
    RandomDropout,
    spawn_rngs,
)
from scisynth.viz import plot, plot_observations
from scisynth.viz._data import flatten

pytest.importorskip("plotly")


def sample(field: AnalyticField, shape: tuple[int, int] = (8, 6)) -> Observation:
    return GridSampler(shape)(field, np.random.default_rng(0))


def test_sampler_plot_is_a_scatter_of_the_grid(field: AnalyticField) -> None:
    fig = GridSampler((8, 6)).plot(field)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.type == "scatter"
    assert trace.mode == "markers"
    obs = sample(field)
    xs, ys = np.meshgrid(obs.coords["x"], obs.coords["y"], indexing="ij")
    assert np.allclose(trace.x, xs.ravel())
    assert np.allclose(trace.y, ys.ravel())
    assert np.allclose(trace.marker.color, obs.values.ravel())
    assert len(trace.x) == 8 * 6
    assert trace.marker.showscale
    assert fig.layout.title.text == "GridSampler"
    assert fig.layout.xaxis.title.text == "x"
    assert fig.layout.yaxis.title.text == "y"


@pytest.mark.parametrize("sampler", [GridSampler((10, 10)), PointSampler(100)])
def test_masked_samples_are_not_drawn(
    field: AnalyticField, sampler: GridSampler | PointSampler
) -> None:
    obs = RandomDropout(0.3)(
        sampler(field, np.random.default_rng(0)), np.random.default_rng(1)
    )
    assert (~obs.mask).any()
    trace = plot(obs).data[0]
    cols, values = flatten(obs)
    assert len(trace.x) == int(obs.mask.sum())
    assert np.allclose(trace.x, cols["x"]) and np.allclose(trace.y, cols["y"])
    assert not np.isnan(np.asarray(trace.marker.color)).any()
    assert np.allclose(trace.marker.color, values)


def test_direct_dispatch_title_and_layout_kwargs(field: AnalyticField) -> None:
    fig = plot(sample(field), "my title", height=321)
    assert fig.layout.title.text == "my title"
    assert fig.layout.height == 321


def test_plot_observations_single_panel(field: AnalyticField) -> None:
    fig = plot_observations([sample(field)], ["only"])
    assert len(fig.data) == 1
    assert [a.text for a in fig.layout.annotations] == ["only"]


def test_one_dimensional_grid_not_implemented() -> None:
    line = AnalyticField(lambda x: x, Domain.from_extents([(0, 1)]))
    with pytest.raises(NotImplementedError, match="2-D"):
        GridSampler(5).plot(line)


def test_point_sampler_plot_is_a_scatter_of_the_samples(field: AnalyticField) -> None:
    fig = PointSampler(30).plot(field)
    assert fig.layout.title.text == "PointSampler"
    trace = fig.data[0]
    assert trace.type == "scatter" and trace.mode == "markers"
    obs = PointSampler(30)(field, spawn_rngs(0, 1)[0])  # what .plot(seed=0) uses
    assert np.allclose(trace.x, obs.coords["x"])
    assert np.allclose(trace.y, obs.coords["y"])
    assert np.allclose(trace.marker.color, obs.values[:, 0])
    assert fig.layout.xaxis.title.text == "x" and fig.layout.yaxis.title.text == "y"


def test_one_dimensional_points_not_implemented() -> None:
    line = AnalyticField(lambda x: x, Domain.from_extents([(0, 1)]))
    with pytest.raises(NotImplementedError, match="2-D"):
        PointSampler(5).plot(line)


def axes_of(fig: Any) -> list[Any]:
    return [fig.layout[k] for k in fig.layout if k.startswith(("xaxis", "yaxis"))]


def assert_static(fig: Any) -> None:
    assert fig.layout.dragmode is False
    assert fig.layout.hovermode is False
    assert {"zoom2d", "pan2d", "select2d", "lasso2d"} <= set(fig.layout.modebar.remove)
    assert axes_of(fig) and all(a.fixedrange is True for a in axes_of(fig))
    assert all(t.hoverinfo == "skip" for t in fig.data)


def assert_interactive(fig: Any) -> None:
    assert fig.layout.dragmode is None and fig.layout.hovermode is None
    assert not fig.layout.modebar.remove
    assert all(a.fixedrange is None for a in axes_of(fig))
    assert all(t.hoverinfo is None for t in fig.data)


def test_plots_are_interactive_by_default(field: AnalyticField) -> None:
    assert_interactive(GridSampler((6, 6)).plot(field))
    assert_interactive(plot(sample(field)))


def test_interactive_false_on_every_plotly_entry_point(field: AnalyticField) -> None:
    from scisynth import GaussianNoise, PointSampler

    obs = sample(field)
    for fig in [
        plot(obs, interactive=False),
        plot(PointSampler(20)(field, np.random.default_rng(0)), interactive=False),
        plot([obs, obs], interactive=False),
        GridSampler((6, 6)).plot(field, interactive=False),
        PointSampler(20).plot(field, interactive=False),
        GaussianNoise(0.1).preview(interactive=False),
        GaussianNoise(0.1).preview(obs, interactive=False),
        plot_observations([obs], interactive=False),
    ]:
        assert_static(fig)


def test_interactive_false_combines_with_layout_kwargs_and_wins_conflicts(
    field: AnalyticField,
) -> None:
    fig = plot(sample(field), interactive=False, height=222, dragmode="zoom")
    assert fig.layout.height == 222
    assert fig.layout.dragmode is False  # interactive=False is applied last


def test_interactive_true_keeps_user_layout_kwargs(field: AnalyticField) -> None:
    fig = plot(sample(field), interactive=True, dragmode="pan")
    assert fig.layout.dragmode == "pan"


def test_static_figure_still_has_the_data(field: AnalyticField) -> None:
    obs = sample(field)
    a, b = plot(obs), plot(obs, interactive=False)
    assert np.array_equal(a.data[0].x, b.data[0].x)
    assert np.array_equal(a.data[0].marker.color, b.data[0].marker.color)


def test_channel_selects_the_plotted_values() -> None:
    from scisynth.latent.compose import Multichannel

    domain = Domain.from_extents([(0.0, 1.0), (0.0, 1.0)])
    two = Multichannel(
        AnalyticField(lambda x, y: x, domain), AnalyticField(lambda x, y: 5 + y, domain)
    )
    obs = GridSampler(4)(two, np.random.default_rng(0))
    first = plot(obs).data[0].marker.color
    second = plot(obs, channel=1).data[0].marker.color
    assert first.max() <= 1.0 and second.min() >= 5.0
    panels = plot([obs, obs], channel=1)
    assert all(t.marker.color.min() >= 5.0 for t in panels.data)
    last = plot(obs, channel=-1).data[0].marker.color
    assert np.array_equal(last, second)  # -1 is the last of two channels
    with pytest.raises(ValueError, match="channel 2"):
        plot(obs, channel=2)
