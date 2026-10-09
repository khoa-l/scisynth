from typing import Any

import numpy as np
import pytest

from scisynth import (
    AnalyticField,
    CoordinateShift,
    GaussianNoise,
    GridSampler,
    Observation,
)
from scisynth.viz import plot

pytest.importorskip("plotly")


def sample(field: AnalyticField) -> Observation:
    return GridSampler((6, 5))(field, np.random.default_rng(0))


def test_shifted_points_move_within_one_fixed_frame(field: AnalyticField) -> None:
    obs = sample(field)
    fig = CoordinateShift((0.3, -0.2)).preview(obs)
    before, after = fig.data
    xs, ys = np.meshgrid(obs.coords["x"], obs.coords["y"], indexing="ij")
    assert np.allclose(before.x, xs.ravel()) and np.allclose(before.y, ys.ravel())
    assert np.allclose(after.x, xs.ravel() + 0.3)
    assert np.allclose(after.y, ys.ravel() - 0.2)
    # both panels use the same coordinate frame ...
    assert tuple(fig.layout.xaxis.range) == tuple(fig.layout.xaxis2.range)
    assert tuple(fig.layout.yaxis.range) == tuple(fig.layout.yaxis2.range)
    # ... which covers where the points were and where they moved
    (x_lo, x_hi), (y_lo, y_hi) = fig.layout.xaxis.range, fig.layout.yaxis.range
    for trace in (before, after):
        assert x_lo <= np.min(trace.x) and np.max(trace.x) <= x_hi
        assert y_lo <= np.min(trace.y) and np.max(trace.y) <= y_hi


def test_frame_is_the_union_with_a_small_pad(field: AnalyticField) -> None:
    obs = sample(field)
    fig = CoordinateShift(1.0).preview(obs)
    x = obs.coords["x"]
    lo, hi = float(x.min()), float(x.max()) + 1.0
    pad = 0.05 * (hi - lo)
    assert tuple(fig.layout.xaxis.range) == pytest.approx((lo - pad, hi + pad))


def test_panels_without_movement_still_share_the_frame() -> None:
    fig = GaussianNoise(0.1).preview()
    assert tuple(fig.layout.xaxis.range) == tuple(fig.layout.xaxis2.range)
    assert tuple(fig.layout.yaxis.range) == tuple(fig.layout.yaxis2.range)
    assert fig.layout.xaxis.range[0] < 0.0 and fig.layout.xaxis.range[1] > 1.0


def test_masked_points_do_not_stretch_the_frame() -> None:
    values = np.array([[1.0], [np.nan], [2.0]])
    coords = {"x": np.array([0.0, 100.0, 1.0]), "y": np.array([0.0, 100.0, 1.0])}
    obs = Observation(values, coords, np.array([True, False, True]))
    fig = plot([obs, obs])
    assert fig.layout.xaxis.range[1] < 2.0 and fig.layout.yaxis.range[1] < 2.0


def test_no_shared_frame_for_a_single_panel_or_different_coordinate_names(
    field: AnalyticField,
) -> None:
    obs = sample(field)
    assert plot([obs]).layout.xaxis.range is None
    other = obs.replace(coords={"lon": obs.coords["x"], "lat": obs.coords["y"]})
    fig = plot([obs, other])
    assert fig.layout.xaxis.range is None and fig.layout.xaxis2.matches is None


def test_user_range_overrides_the_shared_frame(field: AnalyticField) -> None:
    fig = CoordinateShift(0.2).preview(sample(field), xaxis_range=[-5, 5])
    assert tuple(fig.layout.xaxis.range) == (-5, 5)


def panels(field: AnalyticField, n: int, **kwargs: Any) -> Any:
    obs = sample(field)
    return plot([obs] * n, [f"step {i}" for i in range(n)], **kwargs)


def positions(fig: Any) -> list[tuple[float, float]]:
    """(left, top) of each panel's axes domain, in reading order."""
    out = []
    for i in range(1, len(fig.data) + 1):
        suffix = "" if i == 1 else str(i)
        x = fig.layout["xaxis" + suffix].domain
        y = fig.layout["yaxis" + suffix].domain
        out.append((round(x[0], 3), round(y[1], 3)))
    return out


@pytest.mark.parametrize(
    ("n", "rows", "cols"),
    [(1, 1, 1), (3, 1, 3), (4, 1, 4), (5, 2, 4), (8, 2, 4), (16, 4, 4), (17, 5, 4)],
)
def test_panels_fill_rows_of_at_most_four(
    field: AnalyticField, n: int, rows: int, cols: int
) -> None:
    fig = panels(field, n)
    pos = positions(fig)
    assert len(pos) == n
    assert len({left for left, _ in pos}) == cols  # columns
    assert len({top for _, top in pos}) == rows  # rows
    assert pos == sorted(pos, key=lambda p: (-p[1], p[0]))  # reading order
    assert (fig.layout.height is None) == (rows == 1)  # tall figures set a height


def test_one_shared_color_bar_for_the_whole_grid(field: AnalyticField) -> None:
    fig = panels(field, 9)
    assert [t.marker.showscale for t in fig.data].count(True) == 1
    bar = fig.data[-1].marker.colorbar
    assert bar.len == 1 and bar.y == 0.5  # spans the whole grid, at the right
    assert len({(t.marker.cmin, t.marker.cmax) for t in fig.data}) == 1


def test_every_panel_titles_its_own_axes(field: AnalyticField) -> None:
    fig = panels(field, 6)  # 2 rows: 4 on top, 2 below
    for i in range(1, 7):
        suffix = "" if i == 1 else str(i)
        assert fig.layout["xaxis" + suffix].title.text == "x"
        assert fig.layout["yaxis" + suffix].title.text == "y"


def test_panels_with_other_coordinates_keep_their_own_titles(
    field: AnalyticField,
) -> None:
    obs = sample(field)
    other = obs.to_points().replace(
        coords={"c0": obs.to_points().coords["x"], "c1": obs.to_points().coords["y"]}
    )
    fig = plot([obs, other])
    assert (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text) == ("x", "y")
    assert (fig.layout.xaxis2.title.text, fig.layout.yaxis2.title.text) == ("c0", "c1")


def test_zoom_is_linked_across_every_panel(field: AnalyticField) -> None:
    fig = panels(field, 6)
    for i in range(2, 7):
        assert fig.layout[f"xaxis{i}"].matches == "x"
        assert fig.layout[f"yaxis{i}"].matches == "y"


def test_ncols_and_height_can_be_set(field: AnalyticField) -> None:
    assert len({t for _, t in positions(panels(field, 6, ncols=3))}) == 2
    assert len({t for _, t in positions(panels(field, 6, ncols=2))}) == 3
    assert panels(field, 6, height=900).layout.height == 900
    with pytest.raises(ValueError, match="ncols"):
        panels(field, 2, ncols=0)
