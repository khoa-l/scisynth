"""The 3-D plot of one observer run."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core._types import Array, FloatArray
from ..core.rng import SeedLike
from ..latent.base import Latent
from ..observer.observer import Observer
from ..observer.trace import Trace
from ._data import require
from ._trace_layout import (
    Frame,
    PlotAxis,
    axis_names,
    frames_of,
    place,
    resolve_axes,
    select,
)
from .registry import get_plot, register_plot

MAX_POINTS = 150
_MARGIN = {"l": 0, "r": 0, "t": 30, "b": 0}

# Colors: the kinds of line and marker, then the planes, text and grid.
_GREY, _RED, _GREEN, _ORANGE = "#7f8fa6", "#e45756", "#54a24b", "#f58518"
_PURPLE, _BLUE = "#b279a2", "#8AA5F5"
_PLANE, _TEXT, _GRID = "#4c78a8", "#222", "#8a93a3"
GRID_LINES = 5  # per axis, edges included
_GAP = (np.nan, np.nan)  # plotly draws a break at a NaN row
_GRID_XY = np.array(  # a unit square's grid lines, as one polyline with breaks
    [
        row
        for t in np.linspace(0.0, 1.0, GRID_LINES)
        for row in ((t, 0.0), (t, 1.0), _GAP, (0.0, t), (1.0, t), _GAP)
    ]
)
_LABEL_SIZE = 10  # px, the same for every frame's numbers and axis names
_BARE = {  # no walls, gridlines, axis lines or hover spikes
    "showbackground": False,
    "showgrid": False,
    "zeroline": False,
    "showline": False,
    "showspikes": False,
}
_LINES = {
    "stayed": _GREY,
    "moved": _ORANGE,
    "merged": _PURPLE,
    "project": _BLUE,
    "dropped": _RED,
}
_LINE_WIDTH = 1.5
_LEGEND_MARKER = 8  # the plotted markers are smaller than a legend symbol can be
_MARKERS = {
    "kept": (_GREY, "circle"),
    "dropped": (_RED, "x"),
    "added": (_GREEN, "diamond"),
}


def _at(xy: FloatArray, z: int) -> FloatArray:
    return np.column_stack([xy, np.full(len(xy), z)])


def _mark(xy: FloatArray, raw: FloatArray, z: int, source: int) -> FloatArray:
    """Return marker rows ``(x, y, z, true x, true y, layer of the true position)``."""
    n = len(xy)
    return np.column_stack([xy, np.full(n, z), raw, np.full(n, source)])


def _segments(start: FloatArray, end: FloatArray, z: int) -> FloatArray:
    """Return rows ``(x, y, z)`` of segments from ``start`` to ``end``.

    Each segment runs from layer ``z`` to layer ``z + 1``, and is followed by a NaN
    row, which plotly draws as a gap.
    """
    gap = np.full((len(start), 3), np.nan)
    return np.stack([_at(start, z), _at(end, z + 1), gap], axis=1).reshape(-1, 3)


def _link_lines(
    trace: Trace,
    i: int,
    picked: Sequence[Array],
    xys: Sequence[FloatArray],
    raws: Sequence[FloatArray],
    names: Sequence[tuple[str, ...]],
) -> dict[str, FloatArray]:
    """Return segments from layer ``i``'s drawn locations to what they lead to, by kind.

    A one-to-one link has moved if its plotted position differs, whatever the axes.
    """
    diff, pos = trace.diffs[i], picked[i]
    rows, cols = diff.links[:, pos].nonzero()
    there = np.searchsorted(picked[i + 1], rows)
    single = (diff.parents.ravel()[rows] == 1) & (diff.children.ravel()[pos[cols]] == 1)
    moved = single & np.any(raws[i][cols] != raws[i + 1][there], axis=1)
    # other axis names: the next layer is a projection of this one
    projected = names[i] != names[i + 1]
    kinds = {
        "stayed": single & ~moved & (not projected),
        "moved": moved & (not projected),
        "merged": ~single & (not projected),
        "project": np.full(len(rows), projected),
    }
    return {
        name: _segments(xys[i][cols[sel]], xys[i + 1][there[sel]], i)
        for name, sel in kinds.items()
    }


def _marks(
    trace: Trace,
    picked: Sequence[Array],
    xys: Sequence[FloatArray],
    raws: Sequence[FloatArray],
    names: Sequence[tuple[str, ...]],
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    """Return marker rows and line rows by kind, from the trace's diffs.

    ``xys`` are the drawn positions and ``raws`` the true ones.
    """
    markers: dict[str, list[FloatArray]] = {k: [] for k in _MARKERS}
    lines: dict[str, list[FloatArray]] = {k: [] for k in _LINES}
    for i, (pos, xy, raw) in enumerate(zip(picked, xys, raws, strict=True)):
        added = trace.diffs[i - 1].added.ravel()[pos] if i else np.zeros(len(pos), bool)
        markers["added"].append(_mark(xy[added], raw[added], i, i))
        markers["kept"].append(_mark(xy[~added], raw[~added], i, i))
        if i == len(trace.diffs):
            continue
        # a dropped point ends in a cross, in the layer of the stage that dropped it
        gone = trace.diffs[i].dropped.ravel()[pos]
        markers["dropped"].append(_mark(xy[gone], raw[gone], i + 1, i))
        lines["dropped"].append(_segments(xy[gone], xy[gone], i))
        for name, rows in _link_lines(trace, i, picked, xys, raws, names).items():
            lines[name].append(rows)
    return (
        {k: np.concatenate(v) for k, v in markers.items()},
        {k: np.concatenate(v) for k, v in lines.items()},
    )


def _add_planes(fig: Any, go: Any, n_layers: int) -> None:
    """Add one faint unit-square plane per layer."""
    for z in range(n_layers):
        fig.add_trace(
            go.Surface(
                x=[0, 1],
                y=[0, 1],
                z=np.full((2, 2), float(z)),
                colorscale=[[0, _PLANE], [1, _PLANE]],
                opacity=0.12,
                showscale=False,
                hoverinfo="skip",
                contours={a: {"highlight": False} for a in "xyz"},
            )
        )


def _add_grid(fig: Any, go: Any, frame: Frame) -> None:
    """Add an even grid to a frame's plane, at its first layer."""
    fig.add_trace(
        go.Scatter3d(
            x=_GRID_XY[:, 0],
            y=_GRID_XY[:, 1],
            z=np.full(len(_GRID_XY), float(frame.first)),
            mode="lines",
            line={"color": _GRID, "width": 1},
            opacity=0.6,
            hoverinfo="skip",
            showlegend=False,
            name="grid",
        )
    )


def _add_labels(fig: Any, go: Any, frame: Frame, axes: tuple[str, ...]) -> None:
    """Add labels for a frame's axis names and true extent, at its first layer.

    The x axis runs along the front edge and the y axis along the right edge. Texts
    are anchored so that labels meeting at the front corner (x max, y min) and at the
    left corner (x min) lean away from each other and from the z labels.
    """
    (x0, y0), (x1, y1) = frame.lo, frame.hi
    rows = [
        (0.0, -0.06, f"{x0:.3g}", "middle right"),
        (1.0, -0.06, f"{x1:.3g}", "middle left"),
        (0.5, -0.06, axes[0], "middle center"),
        (1.06, 0.0, f"{y0:.3g}", "middle right"),
        (1.06, 1.0, f"{y1:.3g}", "middle right"),
        (1.06, 0.5, axes[1], "middle center"),
    ]
    fig.add_trace(
        go.Scatter3d(
            x=[r[0] for r in rows],
            y=[r[1] for r in rows],
            z=[float(frame.first)] * len(rows),
            text=[r[2] for r in rows],
            mode="text",
            textfont={"size": _LABEL_SIZE, "color": _TEXT},
            textposition=[r[3] for r in rows],
            hoverinfo="skip",
            showlegend=False,
            name="axes",
            meta=list(axes),
        )
    )


def _add_lines(fig: Any, go: Any, lines: dict[str, FloatArray]) -> None:
    """Add one trace per kind of line that has any rows."""
    for name, color in _LINES.items():
        xyz = lines[name]
        if len(xyz):
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="lines",
                    name=f"{name} line",
                    line={"color": color, "width": _LINE_WIDTH},
                    opacity=0.35,
                    hoverinfo="skip",
                )
            )


def _hover_text(
    rows: FloatArray,
    names: Sequence[str],
    frames: Sequence[Frame],
    owner: Sequence[int],
) -> list[str]:
    """Return the hover text of each marker row: its step, and its axes if placed."""
    steps = [names[k] for k in rows[:, 2].astype(int)]
    if frames[0].axes is None:
        return steps
    axes = [frames[owner[k]].axes or () for k in rows[:, 5].astype(int)]
    return [f"{step} ({', '.join(a)})" for step, a in zip(steps, axes, strict=True)]


def _add_markers(
    fig: Any,
    go: Any,
    markers: dict[str, FloatArray],
    names: Sequence[str],
    frames: Sequence[Frame],
    owner: Sequence[int],
) -> None:
    """Add one trace per kind of marker that has any rows.

    Hovering shows the step and, with coordinates, the true position and its axes.
    """
    placed = frames[0].axes is not None
    hover = "%{text}" + (
        "<br>%{customdata[0]:.3g}, %{customdata[1]:.3g}" if placed else ""
    )
    for name, (color, symbol) in _MARKERS.items():
        rows = markers[name]
        if not len(rows):
            continue
        look = {"color": color, "symbol": symbol}
        # an empty trace carries the legend entry, at a size the markers cannot have
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                name=name,
                legendgroup=name,
                marker={"size": _LEGEND_MARKER, **look},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=rows[:, 0],
                y=rows[:, 1],
                z=rows[:, 2],
                mode="markers",
                name=name,
                legendgroup=name,
                showlegend=False,
                marker={"size": 3, **look},
                text=_hover_text(rows, names, frames, owner),
                customdata=rows[:, 3:5],
                hovertemplate=hover + "<extra></extra>",
            )
        )


def _scene(names: Sequence[str]) -> dict[str, Any]:
    """Build the 3-D scene: steps along z; x and y are drawn per frame instead."""
    return {
        "xaxis": {**_BARE, "title": "", "visible": False},
        "yaxis": {**_BARE, "title": "", "visible": False},
        "zaxis": {
            **_BARE,
            "title": "",
            "tickvals": list(range(len(names))),
            "ticktext": list(names),
        },
        "camera": {"eye": {"x": 1.7, "y": -1.7, "z": 1.3}},
        "aspectmode": "manual",
        "aspectratio": {"x": 1, "y": 1, "z": min(1.5, 0.3 * len(names))},
    }


@register_plot("trace", "layers")
@register_plot("observer", "layers")
def plot_trace(
    source: Trace | Observer,
    latent: Latent | None = None,
    *,
    seed: SeedLike = 0,
    max_points: int | None = MAX_POINTS,
    axes: Sequence[PlotAxis] | Sequence[Sequence[PlotAxis] | None] | None = None,
    frame_tolerance: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Plot where each stage keeps, drops, moves and adds points, in 3-D.

    Every step of the trace is a layer, stacked along z, holding the step's valid
    locations. A line joins each location to what it leads to in the next layer
    (the stage's links, see :mod:`scisynth.observer.links`):

    - grey if it stayed, orange if it moved in the plotted axes, purple if the
      stage merged it with others;
    - blue if the next layer has other axes, such as an embedding added with
      :meth:`Projection.extend <scisynth.projection.Projection.extend>`;
    - a location the stage drops ends in a red cross, in that stage's layer, at its
      last position, and one it adds is a green diamond.

    Locations sit at their coordinates, or at the columns chosen with ``axes``.
    Without exactly two coordinate axes and without ``axes`` they sit on a square grid
    in id order instead, so any observation can be drawn.

    Layers are grouped into frames, each drawn as a unit square with a grid, and its
    axis names and true extent labelled at its first layer. A layer starts a new
    frame if its axis names differ from the frame's, or its extent is far from the
    frame's first layer, such as after a large ``CoordinateShift``; otherwise the
    movement would stretch the lines. Lines between frames join the same relative
    positions. Hovering a point shows its step and true position.

    Parameters
    ----------
    source : Trace or Observer
        The observations of one run, with at least two steps. An observer is run on
        ``latent`` first.
    latent : Latent, optional
        The ground truth to observe; required for an observer.
    seed : int, SeedSequence or None, default=0
        Seed for the observer's random streams; ignored for a trace.
    max_points : int or None, default=150
        How many locations to draw. At most this many are taken from the first step,
        spread over the plotted axes (the extremes of each axis, then one per cell of
        a grid over the extent) so that sparse regions are drawn too, and as many
        again from the last step if its axes differ (a projection), traced back to
        the first. Up to this many added by each stage are drawn at random (seed 0).
        Everything the chosen locations lead to is drawn as well. None draws every
        location.
    axes : pair of (str or int), or one pair or None per step, optional
        What to plot along x and y. A string is a coordinate name (``"x"``) and an
        int a channel index (``0``, labelled ``ch0``), so ``(0, 1)`` plots channel 0
        against channel 1 and ``("x", 2)`` channel 2 along x. A single pair applies
        to every step and must exist in all of them. A list gives each step its own
        pair, or None to leave it at its own two coordinates, as for an embedding
        added to a trace of channels: ``[(0, 1)] * (len(trace) - 1) + [None]``. By
        default every step is at its own coordinates, which may differ between steps.
    frame_tolerance : float, default=1.0
        How far a layer may drift from its frame before it gets its own: its center
        by this many spans of the frame's first layer, and its span by a factor of
        ``1 + frame_tolerance``. Zero gives every layer that differs at all its own
        frame; a large value keeps one frame per set of axis names.
    **kwargs
        Passed to ``fig.update_layout``.

    Returns
    -------
    plotly.graph_objects.Figure
        One layer of points per step.

    Raises
    ------
    ValueError
        If the trace has fewer than two steps, ``axes`` are not columns present in the
        steps they are for, ``frame_tolerance`` is negative, or ``source`` is an
        observer and ``latent`` is missing.

    Examples
    --------
    >>> from scisynth.core import Domain
    >>> from scisynth.latent import GaussianField, Multichannel
    >>> from scisynth.observer import Observer
    >>> from scisynth.samplers import GridSampler
    >>> from scisynth.stages import GaussianNoise
    >>> domain = Domain.from_extents([(0, 1), (0, 1)])
    >>> latent = Multichannel(*(GaussianField(domain).realize(s) for s in (1, 2)))
    >>> trace = Observer(GridSampler(16), [GaussianNoise(0.1)]).trace(latent, seed=0)
    >>> fig = trace.plot()  # at the coordinates: noise moves nothing
    >>> fig = trace.plot(axes=(0, 1))  # channel 0 against channel 1: noise moves points
    >>> sorted({t.name for t in fig.data if str(t.name).endswith(" line")})
    ['moved line']
    """
    go = require("plotly.graph_objects")
    if isinstance(source, Observer):
        if latent is None:
            raise ValueError("plotting an observer as a trace needs a latent")
        source = source.trace(latent, seed)
    trace = source
    if len(trace) < 2:
        raise ValueError("a trace needs a sampler and at least one stage to compare")
    if frame_tolerance < 0:
        raise ValueError(f"frame_tolerance must be non-negative, got {frame_tolerance}")
    per_step = resolve_axes(trace, axes)
    names = axis_names(trace, per_step)
    picked = select(trace, max_points, per_step, names)
    raws, by_id = place(trace, picked, per_step)
    frames, owner = frames_of(names, raws, frame_tolerance, by_id)
    xys = [frames[k].unit(raw) for k, raw in zip(owner, raws, strict=True)]
    markers, lines = _marks(trace, picked, xys, raws, names)

    fig = go.Figure()
    _add_planes(fig, go, len(trace))
    for frame in frames:
        if frame.axes is not None:
            _add_grid(fig, go, frame)
            _add_labels(fig, go, frame, frame.axes)
    _add_lines(fig, go, lines)
    _add_markers(fig, go, markers, trace.names, frames, owner)
    layout = {
        "height": 650,
        "margin": _MARGIN,
        "scene": _scene(trace.names),
        **kwargs,
    }
    fig.update_layout(**layout)
    return fig


@register_plot("trace", "*")
def plot_trace_steps(trace: Trace, *, kind: str, **kwargs: Any) -> Any:
    """Plot the steps of a trace side by side, one panel per step.

    Parameters
    ----------
    trace : Trace
        The run to draw. The panels are titled with the step names.
    kind : {"scatter", "heatmap"}
        How each panel is drawn. Filled in by the registry.
    **kwargs
        Passed to the ``panels`` plot, e.g. ``channel``, ``ncols`` or
        ``interactive``.

    Returns
    -------
    plotly.graph_objects.Figure
        One panel per step.
    """
    return get_plot("panels", kind)(trace, **kwargs)
