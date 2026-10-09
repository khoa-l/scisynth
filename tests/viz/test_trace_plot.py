from typing import Any

import numpy as np
import pytest

from scisynth import (
    AnalyticField,
    CoordinateShift,
    Domain,
    Downsample,
    GaussianField,
    GaussianNoise,
    GridSampler,
    Multichannel,
    Observation,
    Observer,
    PointSampler,
    RandomDropout,
    RandomInsertion,
    Trace,
)
from scisynth.viz import plot
from scisynth.viz.trace import GRID_LINES

pytest.importorskip("plotly")


def test_trace_plot_shows_drops_and_moves(field: AnalyticField) -> None:
    observer = Observer(
        GridSampler(12), [RandomDropout(0.3), CoordinateShift([0.5, 0.0])]
    )
    fig = observer.trace(field, 0).plot(max_points=100)
    by_name = {t.name: t for t in fig.data}
    assert {"stayed line", "moved line", "kept", "dropped"} <= set(by_name)
    planes = [t for t in fig.data if t.type == "surface"]
    assert [float(p.z[0][0]) for p in planes] == [0.0, 1.0, 2.0]  # one per layer
    assert not any(p.contours[a].highlight for p in planes for a in "xyz")
    for axis in (fig.layout.scene.xaxis, fig.layout.scene.zaxis):
        assert not (axis.showbackground or axis.showgrid or axis.showline)
        assert not axis.showspikes
    # dropped points are in the layer of the stage that drops them
    assert set(by_name["dropped"].z) == {1.0}
    assert 0 < len(by_name["dropped"].x) <= 100
    assert fig.layout.scene.zaxis.ticktext == (
        "sampler",
        "random_dropout",
        "coordinate_shift",
    )


def test_trace_plot_needs_two_steps_and_panels_are_other_kinds(
    field: AnalyticField,
) -> None:
    trace = Observer(GridSampler(6), [GaussianNoise(0.1)]).trace(field, 0)
    assert len(trace.plot(kind="scatter").data) == 2
    with pytest.raises(ValueError, match="at least one stage"):
        trace[:1].plot()


def test_observer_plots_as_a_trace_with_kind(field: AnalyticField) -> None:
    observer = Observer(GridSampler(8), [RandomDropout(0.3)])
    fig = observer.plot(field, seed=2, kind="layers")
    direct = observer.trace(field, 2).plot()
    assert [t.name for t in fig.data] == [t.name for t in direct.data]
    assert all(
        np.array_equal(np.array(a.x, float), np.array(b.x, float), equal_nan=True)
        for a, b in zip(fig.data, direct.data, strict=True)
    )
    assert len(observer.plot(field).data) == 3  # the default is one panel per step


def test_added_points_are_drawn_in_their_own_layer(field: AnalyticField) -> None:
    observer = Observer(PointSampler(40), [RandomInsertion(8)])
    by_name = {t.name: t for t in observer.trace(field, 1).plot().data}
    assert set(by_name["added"].z) == {1.0} and len(by_name["added"].x) == 8
    assert "dropped" not in by_name


def test_trace_plot_needs_a_latent_for_an_observer() -> None:
    with pytest.raises(ValueError, match="needs a latent"):
        plot(Observer(GridSampler(4), [GaussianNoise(0.1)]), kind="layers")


def tabular_trace() -> Trace:
    rng = np.random.default_rng(0)
    source = Observation(
        values=rng.normal(size=(60, 4)),
        coords={},
        mask=np.ones(60, dtype=bool),
        meta=[{"name": "source", "type": "T", "params": {}}],
    )
    out = RandomDropout(0.3).run(source, 1)
    named = {"name": "random_dropout", "type": "RandomDropout", "params": {}}
    return Trace([source, out.replace(meta=[*source.meta, named])])


def test_data_without_coordinates_is_laid_out_by_id() -> None:
    trace = tabular_trace()
    fig = trace.plot(max_points=60)
    by_name = {t.name: t for t in fig.data}
    assert "moved line" not in by_name
    assert len(by_name["dropped"].x) == trace.diffs[0].n_dropped
    # an id keeps its place in every layer, so the layers line up
    first = {
        (x, y)
        for x, y, z in zip(*(by_name["kept"][k] for k in "xyz"), strict=True)
        if z == 0
    }
    last = {
        (x, y)
        for x, y, z in zip(*(by_name["kept"][k] for k in "xyz"), strict=True)
        if z == 1
    }
    assert last <= first
    assert not fig.layout.scene.xaxis.visible


def test_one_axis_falls_back_to_the_id_grid(field: AnalyticField) -> None:
    line = AnalyticField(lambda x: x, Domain.from_extents([(0.0, 1.0)]))
    fig = Observer(PointSampler(30), [GaussianNoise(0.1)]).trace(line, 0).plot()
    assert len(fig.data) >= 2 and not fig.layout.scene.xaxis.visible


def test_merged_points_are_joined_by_purple_lines(field: AnalyticField) -> None:
    observer = Observer(GridSampler(16), [Downsample(4), GaussianNoise(0.1)])
    trace = observer.trace(field, 0)
    by_name = {t.name: t for t in trace.plot(max_points=20).data}
    assert "merged line" in by_name and "moved line" not in by_name
    # every drawn location of the coarse layer is fed by a drawn one
    coarse = {
        (x, y)
        for x, y, z in zip(*(by_name["kept"][k] for k in "xyz"), strict=True)
        if z == 1
    }
    assert 0 < len(coarse) <= 16


def axes_labels(fig: Any) -> list[tuple[float, list[str]]]:
    """The layer and axis names of every frame drawn."""
    return [(t.z[0], list(t.meta)) for t in fig.data if t.name == "axes"]


def test_every_frame_is_labelled_and_gridded_at_its_first_layer(
    field: AnalyticField,
) -> None:
    fig = Observer(GridSampler(10), [GaussianNoise(0.1)]).trace(field, 0).plot()
    assert axes_labels(fig) == [(0.0, ["x", "y"])]
    assert not fig.layout.scene.xaxis.visible  # the labels replace the scene's axes
    (grid,) = [t for t in fig.data if t.name == "grid"]
    assert set(grid.z) == {0.0}
    assert np.sum(np.isnan(grid.x)) == 2 * GRID_LINES  # a break after every line
    assert {0.0, 1.0} <= set(grid.x[~np.isnan(grid.x)])


def test_legend_symbols_are_larger_than_the_markers_and_lines_are_thin(
    field: AnalyticField,
) -> None:
    fig = Observer(GridSampler(10), [GaussianNoise(0.1)]).trace(field, 0).plot()
    symbol, data = [t for t in fig.data if t.name == "kept"]
    assert (symbol.showlegend, data.showlegend) == (None, False)
    assert symbol.marker.size > data.marker.size
    lines = [t for t in fig.data if str(t.name).endswith(" line")]
    assert lines and all(t.line.width < 2 for t in lines)


def test_small_shifts_share_a_frame_and_large_ones_get_their_own(
    field: AnalyticField,
) -> None:
    def trace_with(shift: float) -> Trace:
        observer = Observer(GridSampler(10), [CoordinateShift([shift, 0.0])])
        return observer.trace(field, 0)

    small, large = trace_with(0.1), trace_with(50.0)
    assert axes_labels(small.plot()) == [(0.0, ["x", "y"])]
    assert axes_labels(large.plot()) == [(0.0, ["x", "y"]), (1.0, ["x", "y"])]
    # the tolerance decides: a huge one never splits, zero always does
    assert len(axes_labels(large.plot(frame_tolerance=1e6))) == 1
    assert len(axes_labels(small.plot(frame_tolerance=0.0))) == 2
    assert [t.z[0] for t in large.plot().data if t.name == "grid"] == [0.0, 1.0]
    with pytest.raises(ValueError, match="frame_tolerance"):
        small.plot(frame_tolerance=-1)


def test_a_new_frame_keeps_lines_inside_the_unit_square(field: AnalyticField) -> None:
    observer = Observer(GridSampler(10), [CoordinateShift([50.0, 0.0])])
    fig = observer.trace(field, 0).plot()
    line = next(t for t in fig.data if t.name == "moved line")
    xy = np.array([line.x, line.y])
    assert np.nanmin(xy) >= 0 and np.nanmax(xy) <= 1
    # its labels give the true extent of the shifted layer
    labels = [t for t in fig.data if t.name == "axes"][1]
    assert (
        abs(float(labels.text[0]) - 50) < 0.2 and abs(float(labels.text[1]) - 51) < 0.2
    )


def test_hover_gives_true_coordinates(field: AnalyticField) -> None:
    observer = Observer(GridSampler(10), [CoordinateShift([50.0, 0.0])])
    trace = observer.trace(field, 0)
    kept = {t.name: t for t in trace.plot().data}["kept"]
    on_last = kept.customdata[kept.z == 1][:, 0]
    assert np.isin(on_last, trace[1].coords["x"]).all()
    assert on_last.min() > 40  # true x, not the position drawn


@pytest.fixture
def multichannel() -> Multichannel:
    domain = Domain.from_extents([(0.0, 1.0), (0.0, 1.0)])
    fields = [GaussianField(domain, length_scale=0.3).realize(s) for s in range(3)]
    return Multichannel(*fields)


def test_axes_plot_one_channel_against_another(multichannel: Multichannel) -> None:
    observer = Observer(GridSampler(10), [GaussianNoise(0.1), RandomDropout(0.2)])
    trace = observer.trace(multichannel, 0)
    # at the coordinates, noise moves nothing; in value space it moves every point
    assert "moved line" not in {t.name for t in trace.plot().data}
    fig = trace.plot(axes=(0, 1))
    assert axes_labels(fig) == [(0.0, ["ch0", "ch1"])]
    moved = next(t for t in fig.data if t.name == "moved line")
    assert np.sum(~np.isnan(moved.x)) > 0
    assert len(next(t for t in fig.data if t.name == "dropped").x) > 0
    # hovering gives the plotted values: channel 0 and 1 of the layer-0 points
    kept = {t.name: t for t in fig.data}["kept"]
    first = kept.customdata[kept.z == 0]
    assert np.isin(first[:, 0], trace[0].values[..., 0]).all()
    assert np.isin(first[:, 1], trace[0].values[..., 1]).all()


def test_negative_channels_count_from_the_end(multichannel: Multichannel) -> None:
    trace = Observer(GridSampler(10), [GaussianNoise(0.1)]).trace(multichannel, 0)
    fig = trace.plot(axes=(-3, -1))  # three channels: the same as (0, 2)
    assert axes_labels(fig) == [(0.0, ["ch0", "ch2"])]
    same = trace.plot(axes=(0, 2))
    for a, b in zip(fig.data, same.data, strict=True):
        assert a.name == b.name
    with pytest.raises(ValueError, match="channel -4 is out of range"):
        trace.plot(axes=(0, -4))


def test_axes_can_mix_a_coordinate_and_a_channel(multichannel: Multichannel) -> None:
    trace = Observer(GridSampler(10), [GaussianNoise(0.1)]).trace(multichannel, 0)
    assert axes_labels(trace.plot(axes=("x", 2))) == [(0.0, ["x", "ch2"])]


def test_axes_must_exist_in_every_step(multichannel: Multichannel) -> None:
    trace = Observer(GridSampler(6), [GaussianNoise(0.1)]).trace(multichannel, 0)
    for axes, message in [
        ((0,), "two entries"),
        (("x", "z"), "'z' is not a coordinate"),
        ((0, 3), "channel 3 is out of range"),
        ((0, -4), "channel -4 is out of range"),
    ]:
        with pytest.raises(ValueError, match=message):
            trace.plot(axes=axes)
