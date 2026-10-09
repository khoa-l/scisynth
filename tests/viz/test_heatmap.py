from typing import Any

import numpy as np
import pytest

from scisynth import (
    AnalyticField,
    CoordinateShift,
    Downsample,
    GaussianNoise,
    GridSampler,
    Observation,
    Observer,
    PointSampler,
    PositionJitter,
    RandomDropout,
)
from scisynth.viz import plot, plot_observations, registered_plots

pytest.importorskip("plotly")


def sample(field: AnalyticField, shape: tuple[int, int] = (8, 6)) -> Observation:
    return GridSampler(shape)(field, np.random.default_rng(0))


def test_a_heatmap_has_one_cell_per_location(field: AnalyticField) -> None:
    obs = sample(field)
    fig = plot(obs, kind="heatmap", title="t")
    (trace,) = fig.data
    assert trace.type == "heatmap"
    assert np.allclose(trace.x, obs.coords["x"]) and np.allclose(
        trace.y, obs.coords["y"]
    )
    assert np.allclose(trace.z, obs.values[..., 0].T)  # rows are y
    assert trace.showscale
    assert fig.layout.xaxis.title.text == "x" and fig.layout.title.text == "t"


def test_scatter_is_still_the_default(field: AnalyticField) -> None:
    assert plot(sample(field)).data[0].type == "scatter"


def test_masked_locations_are_left_empty(field: AnalyticField) -> None:
    obs = RandomDropout(0.4).run(sample(field), 0)
    (trace,) = plot(obs, kind="heatmap").data
    z = np.asarray(trace.z, dtype=float)
    assert np.array_equal(np.isnan(z), ~obs.mask.T)


def test_points_are_averaged_into_bins(field: AnalyticField) -> None:
    points = PointSampler(400)(field, np.random.default_rng(0))
    (trace,) = plot(points, kind="heatmap").data
    z = np.asarray(trace.z, dtype=float)
    counts = np.asarray(trace.customdata)
    assert z.shape == (10, 10) == counts.shape  # about four points per bin
    assert counts.sum() == 400
    assert np.array_equal(np.isnan(z), counts == 0)
    # each cell is the mean of the points inside it
    x, y = points.coords["x"], points.coords["y"]
    cx, cy = np.asarray(trace.x), np.asarray(trace.y)
    wx, wy = cx[1] - cx[0], cy[1] - cy[0]
    i, j = 3, 4
    inside = (abs(x - cx[i]) <= wx / 2) & (abs(y - cy[j]) <= wy / 2)
    if counts[j, i]:  # z is indexed [y, x]
        assert np.isclose(z[j, i], points.values[inside, 0].mean(), atol=1e-6)


def test_bins_option_and_validation(field: AnalyticField) -> None:
    points = PointSampler(100)(field, np.random.default_rng(0))
    assert np.asarray(plot(points, kind="heatmap", bins=7).data[0].z).shape == (7, 7)
    z = plot(points, kind="heatmap", bins=(3, 5)).data[0].z
    assert np.asarray(z).shape == (5, 3)  # rows are y
    with pytest.raises(ValueError, match="bins must be positive"):
        plot(points, kind="heatmap", bins=0)
    with pytest.raises(ValueError, match="bins must be an integer"):
        plot(points, kind="heatmap", bins=2.5)


def test_masked_points_are_left_out_of_the_bins(field: AnalyticField) -> None:
    points = RandomDropout(0.5).run(
        PointSampler(400)(field, np.random.default_rng(0)), 0
    )
    counts = np.asarray(plot(points, kind="heatmap").data[0].customdata)
    assert counts.sum() == points.mask.sum()


def test_all_points_in_one_place_and_no_points(field: AnalyticField) -> None:
    one = Observation(
        np.ones((5, 1)), {"x": np.zeros(5), "y": np.zeros(5)}, np.ones(5, bool)
    )
    assert np.asarray(plot(one, kind="heatmap").data[0].z).sum() == 1  # one mean
    none = one.replace(mask=np.zeros(5, bool), values=np.full((5, 1), np.nan))
    assert plot(none, kind="heatmap").data[0].type == "heatmap"


def test_heatmap_needs_exactly_two_coordinates() -> None:
    line = Observation(np.ones((4, 1)), {"x": np.arange(4.0)}, np.ones(4, bool))
    with pytest.raises(NotImplementedError, match="2-D observations only"):
        plot(line, kind="heatmap")


def test_channel_and_interactive_options(field: AnalyticField) -> None:
    obs = sample(field)
    obs = obs.replace(values=np.concatenate([obs.values, obs.values + 10], axis=-1))
    fig = plot(obs, kind="heatmap", channel=-1, interactive=False)
    assert np.allclose(fig.data[0].z, obs.values[..., 1].T)
    assert fig.layout.dragmode is False


def test_sampler_option_draws_a_heatmap(field: AnalyticField) -> None:
    sampler = GridSampler((8, 6))
    fig = sampler.plot(field, kind="heatmap")
    assert [t.type for t in fig.data] == ["heatmap"]
    assert fig.layout.title.text == "GridSampler"
    assert sampler.plot(field).data[0].type == "scatter"
    points = PointSampler(200).plot(field, kind="heatmap")
    assert [t.type for t in points.data] == ["heatmap"]


def test_stage_option_draws_before_and_after_heatmaps(field: AnalyticField) -> None:
    obs = sample(field, (12, 12))
    fig = GaussianNoise(0.5).preview(obs, kind="heatmap")
    assert [t.type for t in fig.data] == ["heatmap", "heatmap"]
    # one shared color range and one color bar
    assert fig.data[0].zmin == fig.data[1].zmin and fig.data[0].zmax == fig.data[1].zmax
    assert [bool(t.showscale) for t in fig.data] == [False, True]
    assert GaussianNoise(0.5).preview(obs).data[0].type == "scatter"


def test_stages_that_make_points_are_binned_beside_the_grid(
    field: AnalyticField,
) -> None:
    fig = PositionJitter(0.02).preview(sample(field, (20, 20)), kind="heatmap")
    assert [t.type for t in fig.data] == ["heatmap", "heatmap"]
    assert fig.data[0].zmin == fig.data[1].zmin and fig.data[0].zmax == fig.data[1].zmax
    assert np.asarray(fig.data[1].customdata).sum() == 400


def test_stages_that_keep_a_grid_work(field: AnalyticField) -> None:
    obs = sample(field, (12, 12))
    for stage in (Downsample(3), CoordinateShift(0.2), RandomDropout(0.3)):
        fig = stage.preview(obs, kind="heatmap")
        assert [t.type for t in fig.data] == ["heatmap", "heatmap"]


def test_panels_share_one_frame_that_covers_whole_cells(field: AnalyticField) -> None:
    obs = sample(field, (3, 3))  # coarse: half a cell is wide
    fig = plot_observations([obs, obs], kind="heatmap")
    x = obs.coords["x"]
    half = (x[1] - x[0]) / 2
    lo, hi = fig.layout.xaxis.range
    assert lo < x.min() - half and hi > x.max() + half


def test_panel_kind_must_be_declared(field: AnalyticField) -> None:
    with pytest.raises(ValueError, match="cannot be plotted as 'bogus'"):
        plot_observations([sample(field)], kind="bogus")


def test_one_heatmap_plot_serves_every_subject(field: AnalyticField) -> None:
    keys = set(registered_plots())
    assert ("observation", "heatmap") in keys
    # sampler, stage and panels are not registered per kind: they pass the kind on
    assert not any(kind == "heatmap" and s != "observation" for s, kind in keys)
    trace = Observer(GridSampler(6), [GaussianNoise(0.1)]).trace(field, 0)
    assert [t.type for t in trace.plot(kind="heatmap").data] == ["heatmap"] * 2
    assert [t.type for t in Observer(GridSampler(6)).plot(field, kind="heatmap").data][
        -1
    ] == "heatmap"


def test_a_grid_can_be_averaged_into_blocks(field: AnalyticField) -> None:
    obs = sample(field, (12, 12))
    (trace,) = plot(obs, kind="heatmap", bins=4).data
    z = np.asarray(trace.z, dtype=float)
    assert z.shape == (4, 4)
    assert (np.asarray(trace.customdata) == 9).all()  # 3 x 3 cells each
    # the same numbers as downsampling by 3
    small = Downsample(3).run(obs, 0)
    assert np.allclose(z, small.values[..., 0].T)
    assert np.allclose(trace.x, small.coords["x"]) and np.allclose(
        trace.y, small.coords["y"]
    )


def test_grid_blocks_skip_masked_cells_and_cap_at_the_grid(
    field: AnalyticField,
) -> None:
    obs = RandomDropout(0.5).run(sample(field, (10, 10)), 0)
    trace = plot(obs, kind="heatmap", bins=(5, 2)).data[0]
    assert np.asarray(trace.customdata).sum() == obs.mask.sum()
    assert np.asarray(trace.z).shape == (2, 5)
    same = plot(obs, kind="heatmap", bins=50).data[0]  # more bins than cells
    assert np.asarray(same.z).shape == (10, 10)
    assert np.array_equal(np.asarray(same.customdata), obs.mask.T.astype(int))
    uneven = plot(sample(field, (7, 7)), kind="heatmap", bins=3).data[0]
    assert sorted(set(np.asarray(uneven.customdata).ravel())) == [4, 6, 9]  # 2 or 3


def test_a_grid_and_points_can_share_a_cell_size(field: AnalyticField) -> None:
    grid = sample(field, (20, 20))
    points = PointSampler(400)(field, np.random.default_rng(0))
    a = plot(grid, kind="heatmap", bins=10).data[0]
    b = plot(points, kind="heatmap", bins=10).data[0]
    assert np.asarray(a.z).shape == np.asarray(b.z).shape == (10, 10)


def shapes(fig: Any) -> list[tuple[int, ...]]:
    return [np.asarray(t.z).shape for t in fig.data]


def test_bins_reach_every_panel_of_every_entry_point(field: AnalyticField) -> None:
    grid = sample(field, (20, 20))
    fig = PositionJitter(0.01).preview(grid, kind="heatmap", bins=5)
    assert shapes(fig) == [(5, 5), (5, 5)]  # a grid and points, one cell size
    two = plot_observations([grid, grid], kind="heatmap", bins=(4, 2))
    assert shapes(two) == [(2, 4), (2, 4)]  # rows are y
    observer = Observer(GridSampler(12), [GaussianNoise(0.1)])
    assert shapes(observer.trace(field, 0).plot(kind="heatmap", bins=3)) == [
        (3, 3),
        (3, 3),
    ]
    assert shapes(observer.plot(field, kind="heatmap", bins=6))[-1] == (6, 6)
    with pytest.raises(ValueError, match="bins applies to heatmaps"):
        plot_observations([grid], kind="scatter", bins=3)
