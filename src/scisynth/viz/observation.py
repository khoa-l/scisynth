"""Plots for observations: ``scatter`` and ``heatmap`` for one, ``panels`` for several.

Each function imports its plotting library itself, when called.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core._numbers import as_int
from ..core.observation import Observation
from ..observer.trace import Trace
from ._data import bin_means, block_means, default_bins, flatten, require
from .registry import register_plot, resolve_plot

PANELS_PER_ROW = 4
_PANEL_PX = 230  # height of one panel's plot area
_GAP_PX = 70  # between rows, for panel titles and tick labels
_MARGIN_PX = 110  # top and bottom margins together
_COLORSCALE = "Viridis"
_COLORBAR = {"thickness": 14, "len": 1.0, "y": 0.5}  # one bar, full height

# Modebar buttons that pan, zoom or select; the image download button is kept.
_INTERACTIVE_BUTTONS = [
    "zoom2d",
    "pan2d",
    "select2d",
    "lasso2d",
    "zoomIn2d",
    "zoomOut2d",
    "autoScale2d",
    "resetScale2d",
]


def _disable_interaction(fig: Any) -> None:
    """Turn off drag (zoom/pan/select), hover and the matching modebar buttons.

    These settings are part of the figure itself. Plotly's fully static mode
    (``staticPlot``, which also hides the modebar) is a display-time option instead:
    ``fig.show(config={"staticPlot": True})``.
    """
    fig.update_layout(
        dragmode=False, hovermode=False, modebar={"remove": _INTERACTIVE_BUTTONS}
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_traces(hoverinfo="skip")


@register_plot("observation", "scatter")
def plotly_scatter(
    obs: Observation,
    title: str = "",
    *,
    interactive: bool = True,
    channel: int = 0,
    **kwargs: Any,
) -> Any:
    """Plot a 2-D observation as a scatter, one marker per valid sample.

    Markers sit at the locations (grid cells or points) and are colored by
    value. Masked samples are not drawn.

    Parameters
    ----------
    obs : Observation
        A 2-D observation, in either layout.
    title : str, default=""
        Figure title.
    interactive : bool, default=True
        False disables zoom, pan, selection and hover.
    channel : int, default=0
        Which channel gives the colors.
    **kwargs
        Passed to ``fig.update_layout``.

    Returns
    -------
    plotly.graph_objects.Figure
        A scatter colored by value.

    Raises
    ------
    NotImplementedError
        If the observation is not 2-D.
    """
    go = require("plotly.graph_objects")
    if len(obs.coords) != 2:
        raise NotImplementedError(
            f"plotly scatter supports 2-D observations only, got {len(obs.coords)}-D"
        )
    cols, values = flatten(obs, channel)
    x_name, y_name = cols
    fig = go.Figure(
        go.Scatter(
            x=cols[x_name],
            y=cols[y_name],
            mode="markers",
            marker={"color": values, "colorscale": _COLORSCALE, "showscale": True},
        )
    )
    return _style(fig, x_name, y_name, title, interactive, kwargs)


@register_plot("observation", "heatmap")
def plotly_heatmap(
    obs: Observation,
    title: str = "",
    *,
    interactive: bool = True,
    channel: int = 0,
    bins: int | tuple[int, int] | None = None,
    **kwargs: Any,
) -> Any:
    """Plot a 2-D observation as a heatmap.

    A grid gets one cell per location. Points are averaged into a regular grid of
    bins laid over them, so each cell shows the mean value of the points in it.
    With ``bins``, a grid is also averaged, into coarser blocks of neighbouring
    cells. Cells are colored with the same color scale as :func:`plotly_scatter`,
    and masked locations and empty bins are left blank.

    Parameters
    ----------
    obs : Observation
        A 2-D observation, in either layout.
    title : str, default=""
        Figure title.
    interactive : bool, default=True
        False disables zoom, pan, selection and hover.
    channel : int, default=0
        Which channel gives the colors.
    bins : int or (int, int), optional
        Bins along x and y (one number is used for both). For points the default
        holds about four points per bin. A grid shows every cell unless this is
        given, and then it is averaged into at most this many blocks per axis (a
        value above the number of cells leaves that axis as it is).
    **kwargs
        Passed to ``fig.update_layout``.

    Returns
    -------
    plotly.graph_objects.Figure
        A heatmap colored by value.

    Raises
    ------
    NotImplementedError
        If the observation does not have exactly two coordinates.
    ValueError
        If ``bins`` is not a positive whole number.
    """
    go = require("plotly.graph_objects")
    if len(obs.coords) != 2:
        raise NotImplementedError(
            f"plotly heatmap supports 2-D observations only, got {len(obs.coords)}-D"
        )
    x_name, y_name = obs.coords
    cells = _grid_cells if obs.layout == "grid" else _point_cells
    x, y, z, counts, tip = cells(obs, channel, bins)
    fig = go.Figure(
        go.Heatmap(
            x=x,
            y=y,
            z=z.T,  # plotly wants rows to be y
            customdata=None if counts is None else counts.T,
            hovertemplate=tip,
            colorscale=_COLORSCALE,
            hoverongaps=False,
            showscale=True,
            colorbar=_COLORBAR,
        )
    )
    return _style(fig, x_name, y_name, title, interactive, kwargs)


_Cells = tuple[Any, Any, Any, Any, str]  # x, y, values, counts or None, hover text


def _mean_tip(unit: str) -> str:
    """Return the hover text of a cell that averages ``unit``s.

    It uses plotly's own ``%{}`` fields.
    """
    return (
        "%{x:.3g}, %{y:.3g}<br>mean %{z}<br>"
        f"%{{customdata}} {unit}(s)<extra></extra>"
    )


def _grid_cells(
    obs: Observation, channel: int, bins: int | tuple[int, int] | None
) -> _Cells:
    """Return the cells of a grid: one per location, or blocks if ``bins`` is given."""
    x_name, y_name = obs.coords
    flatten(obs, channel)  # checks the channel
    x, y = obs.coords[x_name], obs.coords[y_name]
    if bins is None:
        z = np.where(obs.mask, obs.values[..., channel], np.nan)
        return x, y, z, None, "%{x}, %{y}<br>%{z}<extra></extra>"
    values = obs.values[..., channel]
    x, y, z, counts = block_means(x, y, values, obs.mask, _bins_pair(bins, 0))
    return x, y, z, counts, _mean_tip("cell")


def _point_cells(
    obs: Observation, channel: int, bins: int | tuple[int, int] | None
) -> _Cells:
    """Return the cells of points: their mean value in a regular grid of bins."""
    x_name, y_name = obs.coords
    cols, values = flatten(obs, channel)
    n_bins = _bins_pair(bins, len(values))
    x, y, z, counts = bin_means(cols[x_name], cols[y_name], values, n_bins)
    return x, y, z, counts, _mean_tip("point")


def _bins_pair(bins: int | tuple[int, int] | None, n_points: int) -> tuple[int, int]:
    """Return the bins along x and y: as given, or the default.

    One number is used for both axes.
    """
    if bins is None:
        bins = default_bins(n_points)
    pair = bins if isinstance(bins, tuple) else (bins, bins)
    x_bins, y_bins = (as_int(n, "bins") for n in pair)
    if x_bins < 1 or y_bins < 1:
        raise ValueError(f"bins must be positive, got {bins!r}")
    return x_bins, y_bins


def _style(
    fig: Any,
    x_name: str,
    y_name: str,
    title: str,
    interactive: bool,
    layout: dict[str, Any],
) -> Any:
    """Title the axes and figure; ``layout`` goes to ``fig.update_layout``."""
    fig.update_xaxes(title_text=x_name)
    fig.update_yaxes(title_text=y_name)
    fig.update_layout(title_text=title or None, **layout)
    if not interactive:
        _disable_interaction(fig)  # after the layout, so it always wins
    return fig


def _half_cell(column: Any) -> float:
    """Return half the typical spacing of a column of coordinates.

    This is how far a heatmap cell reaches beyond its center.
    """
    spacing = np.diff(np.unique(column))
    return float(np.median(spacing)) / 2 if spacing.size else 0.0


def _shared_ranges(
    observations: Sequence[Observation], cells: bool = False
) -> list[tuple[float, float]] | None:
    """Return one padded ``(lo, hi)`` per axis, covering every panel's drawn points.

    With ``cells``, each point is a heatmap cell, so the range also covers the half
    of a cell beyond its center. Returns ``None`` unless all panels use the same
    coordinate names (their axes then mean the same thing) and at least one point
    is drawn.
    """
    if len({tuple(o.coords) for o in observations}) != 1:
        return None
    ranges: list[tuple[float, float]] = []
    for name in observations[0].coords:
        columns = [flatten(o)[0][name] for o in observations]
        columns = [c for c in columns if c.size]
        if not columns:
            return None
        reach = [_half_cell(c) if cells else 0.0 for c in columns]
        lo = float(min(c.min() - r for c, r in zip(columns, reach, strict=True)))
        hi = float(max(c.max() + r for c, r in zip(columns, reach, strict=True)))
        pad = 0.05 * (hi - lo) or 0.5
        ranges.append((lo - pad, hi + pad))
    return ranges


def _grid_shape(n: int, ncols: int) -> tuple[int, int]:
    """Return the rows and columns for ``n`` panels, with at most ``ncols`` per row."""
    cols = min(n, ncols)
    return math.ceil(n / cols), cols


def _cell(k: int, cols: int) -> tuple[int, int]:
    """Return the 1-based ``(row, col)`` of panel ``k``, filling rows left to right."""
    return k // cols + 1, k % cols + 1


def _share_color_scale(fig: Any) -> None:
    """Give every panel the same color range, and the last one the only color bar."""
    if fig.data[0].type == "heatmap":
        values = np.concatenate(
            [np.asarray(t.z, dtype=float).ravel() for t in fig.data]
        )
        values = values[np.isfinite(values)]
        if values.size:
            fig.update_traces(zmin=values.min(), zmax=values.max())
        fig.update_traces(showscale=False)
        fig.data[-1].update(showscale=True, colorbar=_COLORBAR)
        return
    colors = np.concatenate([np.asarray(t.marker.color, dtype=float) for t in fig.data])
    fig.update_traces(
        marker={"cmin": colors.min(), "cmax": colors.max(), "showscale": False}
    )
    fig.data[-1].marker.update(showscale=True, colorbar=_COLORBAR)


def _share_frame(
    fig: Any, observations: Sequence[Observation], cols: int, cells: bool
) -> None:
    """Give every panel the same axis ranges, with zooming linked to the first.

    Does nothing when the panels' coordinates differ (see :func:`_shared_ranges`).
    """
    ranges = _shared_ranges(observations, cells)
    if ranges is None:
        return
    (x_lo, x_hi), (y_lo, y_hi) = ranges
    fig.update_xaxes(range=[x_lo, x_hi])
    fig.update_yaxes(range=[y_lo, y_hi])
    for k in range(1, len(observations)):
        row, col = _cell(k, cols)
        fig.update_xaxes(matches="x", row=row, col=col)
        fig.update_yaxes(matches="y", row=row, col=col)


@register_plot("panels", "*")
def plotly_panels(
    observations: Sequence[Observation],
    titles: Sequence[str] | None = None,
    *,
    ncols: int = PANELS_PER_ROW,
    interactive: bool = True,
    channel: int = 0,
    bins: int | tuple[int, int] | None = None,
    kind: str,
    **kwargs: Any,
) -> Any:
    """Plot observations as a grid of subplots in one figure.

    Panels fill rows left to right, at most ``ncols`` per row, and rows are added as
    needed (4 panels make one row, 17 make five). Each panel is drawn by the plot
    registered for its own kind. All panels share one color scale and one color bar
    at the right, and one coordinate frame with linked zooming. This way a stage
    that moves points, such as ``CoordinateShift``, shows them moving from the same
    coordinates, instead of each panel re-centering on its own data. Every panel
    titles its own axes, because panels may use different coordinates (an embedding
    next to a sample).

    Parameters
    ----------
    observations : sequence of Observation
        The observations to draw, one per panel, in reading order. A ``Trace`` is a
        sequence of observations, so it draws one panel per step.
    titles : sequence of str, optional
        One title per panel; for a ``Trace``, the step names by default.
    ncols : int, default=4
        Maximum number of panels per row.
    interactive : bool, default=True
        False disables zoom, pan, selection and hover on every panel.
    channel : int, default=0
        The channel drawn in every panel.
    bins : int or (int, int), optional
        For heatmaps: the same bins along x and y in every panel, so a grid and
        points get the same cell size (see :func:`plotly_heatmap`). By default every
        panel chooses its own.
    kind : {"scatter", "heatmap"}
        How each panel is drawn, filled in by the registry. For points a heatmap
        averages them into bins.
    **kwargs
        Passed to ``fig.update_layout``; can override the shared ranges and the size.

    Returns
    -------
    plotly.graph_objects.Figure
        One figure with a subplot per observation.

    Raises
    ------
    ValueError
        If ``observations`` is empty, ``ncols`` is less than 1, ``kind`` is not a
        kind that observations declare, or ``bins`` is given for the scatter.
    NotImplementedError
        If a panel does not have exactly two coordinates.
    """
    if not observations:
        raise ValueError("nothing to plot")
    if ncols < 1:
        raise ValueError(f"ncols must be >= 1, got {ncols}")
    if bins is not None and kind != "heatmap":
        raise ValueError(f"bins applies to heatmaps, not kind {kind!r}")
    options = {"channel": channel} | ({} if bins is None else {"bins": bins})
    n = len(observations)
    rows, cols = _grid_shape(n, ncols)
    if titles is None and isinstance(observations, Trace):
        titles = observations.names
    labels = list(titles) if titles is not None else []
    labels += [""] * (n - len(labels))
    plot_height = rows * _PANEL_PX + (rows - 1) * _GAP_PX
    fig = require("plotly.subplots").make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=labels,
        horizontal_spacing=0.11 / cols**0.5,  # room for the next panel's y title
        vertical_spacing=_GAP_PX / plot_height if rows > 1 else 0.0,
    )
    for k, obs in enumerate(observations):
        row, col = _cell(k, cols)
        single = resolve_plot(obs, kind)(obs, **options)
        fig.add_trace(single.data[0], row=row, col=col)
        fig.update_xaxes(title_text=single.layout.xaxis.title.text, row=row, col=col)
        fig.update_yaxes(title_text=single.layout.yaxis.title.text, row=row, col=col)
    _share_color_scale(fig)
    if n > 1:
        _share_frame(fig, observations, cols, cells=kind == "heatmap")
    if n > 2:
        fig.update_annotations(font_size=13)  # panel titles
    if rows > 1:
        fig.update_layout(
            height=plot_height + _MARGIN_PX, margin={"t": 60, "b": _MARGIN_PX - 60}
        )
    fig.update_layout(showlegend=False, **kwargs)
    if not interactive:
        _disable_interaction(fig)  # after kwargs, so it always wins
    return fig
