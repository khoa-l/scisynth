"""Layout of the 3-D trace plot: which locations, at what (x, y), in which frame.

The plotted axes are columns of the observations: a coordinate name (``"x"``) or a
channel index (``0``). By default each layer uses its own coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..core._numbers import as_index
from ..core._types import Array, BoolArray, FloatArray
from ..core.observation import Observation
from ..observer.trace import Trace

PlotAxis = str | int  # a coordinate name, or a channel index


Pair = tuple[PlotAxis, PlotAxis]


def _axis(axis: PlotAxis, step: str, obs: Observation) -> PlotAxis:
    """Return a plotted axis, checked against a step.

    A negative channel counts from the end of the step's channels.
    """
    if isinstance(axis, str):
        if axis not in obs.coords:
            raise ValueError(
                f"axis {axis!r} is not a coordinate of step {step!r}, whose "
                f"coordinates are {list(obs.coords)}"
            )
        return axis
    where = f" for step {step!r}, which has {obs.n_channels} channel(s)"
    return as_index(axis, obs.n_channels, "channel", where)


def resolve_axes(
    trace: Trace, axes: Sequence[PlotAxis] | Sequence[Sequence[PlotAxis] | None] | None
) -> list[Pair | None]:
    """Return the pair of plotted axes for each step of ``trace``.

    Parameters
    ----------
    trace : Trace
        The run being plotted.
    axes : None, a pair, or a sequence with one entry per step
        None leaves every step at its own coordinates. A pair (each a coordinate
        name or a channel number) applies to every step. A sequence gives each step
        its own pair, or None for its own coordinates.

    Returns
    -------
    list of (pair or None)
        One entry per step. None means the step uses its own coordinates. A negative
        channel counts from the end of that step's channels, so ``-1`` is the last
        one, and the returned pairs hold the non-negative number.

    Raises
    ------
    ValueError
        If ``axes`` does not have one of those shapes, a coordinate name is missing
        from a step, a channel number is out of range for it, or a step left at its
        coordinates has other than two.
    """
    n = len(trace)
    if axes is None:
        per_step: list[Pair | None] = [None] * n
    elif all(isinstance(a, str | int) for a in axes):
        if len(axes) != 2:
            raise ValueError(f"axes needs two entries, got {len(axes)}")
        per_step = [(axes[0], axes[1])] * n  # type: ignore[list-item]
    elif len(axes) == n:
        per_step = [None if a is None else (a[0], a[1]) for a in axes]  # type: ignore[index]
    else:
        raise ValueError(
            f"axes needs a pair or one entry per step ({n}), got {len(axes)}"
        )
    for k, (name, obs, pair) in enumerate(
        zip(trace.names, trace, per_step, strict=True)
    ):
        if pair is None and axes is not None and len(obs.coords) != 2:
            raise ValueError(
                f"step {name!r} has {len(obs.coords)} coordinates, so it needs axes"
            )
        if pair is not None:
            per_step[k] = (_axis(pair[0], name, obs), _axis(pair[1], name, obs))
    return per_step


def axis_names(trace: Trace, per_step: Sequence[Pair | None]) -> list[tuple[str, ...]]:
    """Return the names of each step's plotted axes.

    Parameters
    ----------
    trace : Trace
        The run being plotted.
    per_step : sequence of (pair or None)
        The axes of each step, from :func:`resolve_axes`.

    Returns
    -------
    list of tuple of str
        Coordinate names, or ``ch0``, ``ch1``, ... for channels, for each step.
    """
    return [
        tuple(obs.coords)
        if pair is None
        else tuple(a if isinstance(a, str) else f"ch{a}" for a in pair)
        for obs, pair in zip(trace, per_step, strict=True)
    ]


def columns(obs: Observation, pos: Array, axes: Pair | None) -> FloatArray:
    """Return the plotted columns of the locations at flat positions ``pos``.

    Parameters
    ----------
    obs : Observation
        The step to read.
    pos : ndarray of int
        Flat (storage-order) positions of the locations.
    axes : pair or None
        The two coordinates or channels to read, or None for the observation's own
        coordinates.

    Returns
    -------
    ndarray of shape (len(pos), k)
        One column per plotted axis.
    """
    if axes is None:
        cols = [c.ravel()[pos] for c in obs.spatial_coords.values()]
    else:
        cols = [
            obs.spatial_coords[a].ravel()[pos]
            if isinstance(a, str)
            else obs.values[..., a].ravel()[pos]
            for a in axes
        ]
    return np.column_stack(cols) if cols else np.empty((len(pos), 0))


def spread(coords: FloatArray, n: int | None, rng: np.random.Generator) -> Array:
    """Pick up to ``n`` rows of ``coords`` (all if None), spread over their range.

    Candidates are ranked. First come the rows with the smallest and largest value
    on each axis, so the whole range is drawn. Then one random row per occupied cell
    of a grid of about ``n`` cells over the extent, so sparse regions are drawn as
    well as dense ones. Then everything else, at random. The first ``n`` are taken.

    Parameters
    ----------
    coords : ndarray of shape (m, d)
        One row per candidate.
    n : int or None
        How many rows to pick. None picks all.
    rng : numpy.random.Generator
        Source of the random choices.

    Returns
    -------
    ndarray of int
        The picked row numbers, sorted.
    """
    m, d = coords.shape
    if n is None or n >= m:
        return np.arange(m)
    order = rng.permutation(m)
    if d == 0:
        return np.sort(order[:n])
    extremes = np.concatenate([coords.argmin(axis=0), coords.argmax(axis=0)])
    bins = math.ceil(n ** (1 / d))
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    unit = (coords - lo) / np.where(hi > lo, hi - lo, 1.0)
    cell = np.minimum((unit * bins).astype(np.intp), bins - 1)
    key = np.ravel_multi_index(tuple(cell.T), (bins,) * d)
    one_per_cell = rng.permutation(order[np.unique(key[order], return_index=True)[1]])
    ranked = np.concatenate([extremes, one_per_cell, order])
    first = np.sort(np.unique(ranked, return_index=True)[1])  # keep the first of each
    return np.sort(ranked[first][:n])


def select(
    trace: Trace,
    max_points: int | None,
    per_step: Sequence[Pair | None],
    names: Sequence[tuple[str, ...]],
) -> list[Array]:
    """Pick the locations to draw in each step.

    Start with a sample of the first step's valid locations, spread over the plotted
    axes (see :func:`spread`). If the last step has other axes (a projection), add a
    sample spread over those, traced back to the first step. For each later step, add
    everything the picked locations lead to, plus a sample of the locations that
    stage added. Every drawn location is therefore joined to the layer before it.

    Parameters
    ----------
    trace : Trace
        The run being plotted.
    max_points : int or None
        The size of each sample. None picks every location.
    per_step : sequence of (pair or None)
        The axes of each step, from :func:`resolve_axes`.
    names : sequence of tuple of str
        The axis names of each step, from :func:`axis_names`.

    Returns
    -------
    list of ndarray of int
        The flat positions to draw in each step, sorted.
    """
    rng = np.random.default_rng(0)

    def cover(obs: Observation, axes: Pair | None) -> Array:
        pos = np.flatnonzero(obs.mask.ravel())
        return pos[spread(columns(obs, pos, axes), max_points, rng)]

    def sample(flat: BoolArray) -> Array:
        pos = np.flatnonzero(flat)
        n = pos.size if max_points is None else min(max_points, pos.size)
        return np.sort(rng.choice(pos, n, replace=False))

    first, last = cover(trace[0], per_step[0]), trace[-1]
    if names[-1] != names[0]:
        back = trace.links_between(0, -1)[cover(last, per_step[-1])].nonzero()[1]
        first = np.union1d(first, back)
    picked = [first]
    for diff in trace.diffs:
        children = diff.links[:, picked[-1]].nonzero()[0]
        picked.append(np.union1d(children, sample(diff.added.ravel())))
    return picked


def place(
    trace: Trace, picked: Sequence[Array], per_step: Sequence[Pair | None]
) -> tuple[list[FloatArray], bool]:
    """Return an (x, y) for each picked location, and whether they were placed by id.

    Locations sit at the chosen axes, or at their coordinates if every step has
    exactly two. Otherwise they sit on a square grid in id order, in the same place
    in every layer.

    Parameters
    ----------
    trace : Trace
        The run being plotted.
    picked : sequence of ndarray of int
        The flat positions to draw in each step, from :func:`select`.
    per_step : sequence of (pair or None)
        The axes of each step, from :func:`resolve_axes`.

    Returns
    -------
    positions : list of ndarray of shape (n, k)
        The plotted position of each picked location, for each step.
    by_id : bool
        True if the locations were placed on a grid by id.
    """
    if any(per_step) or all(len(o.coords) == 2 for o in trace):
        return [
            columns(o, pos, pair)
            for o, pos, pair in zip(trace, picked, per_step, strict=True)
        ], False
    ids = [o.ids.ravel()[pos] for o, pos in zip(trace, picked, strict=True)]
    everyone = np.unique(np.concatenate(ids))
    k = np.arange(len(everyone))
    width = max(1, math.isqrt(len(everyone) - 1) + 1)
    grid = np.column_stack([k % width, k // width]).astype(np.float64)
    return [grid[np.searchsorted(everyone, i)] for i in ids], True


@dataclass(frozen=True)
class Frame:
    """A coordinate frame shared by consecutive layers: its axes and extent."""

    first: int  # the layer the axes are drawn at
    axes: tuple[str, ...] | None  # axis names; None when layers are placed by id
    lo: FloatArray
    hi: FloatArray

    def unit(self, xy: FloatArray) -> FloatArray:
        """Map positions in this frame to the unit square the planes are drawn as.

        Parameters
        ----------
        xy : ndarray of shape (n, 2)
            Positions in the frame's own units.

        Returns
        -------
        ndarray of shape (n, 2)
            The same positions in the unit square.
        """
        out: FloatArray = (xy - self.lo) / (self.hi - self.lo)
        return out


def _box(xy: FloatArray) -> tuple[FloatArray, FloatArray]:
    return xy.min(axis=0), xy.max(axis=0)


def _near(
    box: tuple[FloatArray, FloatArray],
    ref: tuple[FloatArray, FloatArray],
    tolerance: float,
) -> bool:
    """Return whether a layer's extent stays close to a frame's reference extent.

    Its center may be off by ``tolerance`` reference spans and its span may grow or
    shrink by a factor of ``1 + tolerance``, per axis.
    """
    (lo, hi), (rlo, rhi) = box, ref
    span, rspan = hi - lo, rhi - rlo
    grow = 1 + tolerance
    centered = np.abs(lo + hi - rlo - rhi) / 2 <= tolerance * rspan
    return bool(np.all(centered & (span <= rspan * grow) & (span * grow >= rspan)))


def _joins(
    names: Sequence[tuple[str, ...]],
    raws: Sequence[FloatArray],
    first: int,
    i: int,
    tolerance: float,
) -> bool:
    """Return whether layer ``i`` is in the frame that starts at layer ``first``."""
    if names[i] != names[first]:
        return False
    if not len(raws[i]) or not len(raws[first]):
        return True
    return _near(_box(raws[i]), _box(raws[first]), tolerance)


def _frame(
    names: Sequence[tuple[str, ...]],
    raws: Sequence[FloatArray],
    layers: list[int],
    by_id: bool,
) -> Frame:
    """Build the frame of some layers: their extent plus 5% padding, and their axes."""
    drawn = np.concatenate([raws[i] for i in layers])
    lo, hi = _box(drawn) if len(drawn) else (np.zeros(2), np.ones(2))
    span, center = np.where(hi > lo, hi - lo, 1.0), (lo + hi) / 2
    axes = None if by_id else names[layers[0]][:2]
    return Frame(layers[0], axes, center - 0.55 * span, center + 0.55 * span)


def frames_of(
    names: Sequence[tuple[str, ...]],
    raws: Sequence[FloatArray],
    tolerance: float,
    by_id: bool,
) -> tuple[list[Frame], list[int]]:
    """Group consecutive layers into frames.

    A layer joins the frame of the one before it unless it has other axis names or
    its extent is not near that frame's first layer (see :func:`_near`). Layers
    placed by id always share one frame.

    Parameters
    ----------
    names : sequence of tuple of str
        The axis names of each layer.
    raws : sequence of ndarray
        The true plotted positions of each layer.
    tolerance : float
        How far a layer may drift from its frame before it gets its own.
    by_id : bool
        Whether the layers were placed by id.

    Returns
    -------
    frames : list of Frame
        The frames, in order.
    owner : list of int
        The frame each layer is in.
    """
    groups: list[list[int]] = []
    for i in range(len(raws)):
        if groups and (by_id or _joins(names, raws, groups[-1][0], i, tolerance)):
            groups[-1].append(i)
        else:
            groups.append([i])
    owner = [k for k, layers in enumerate(groups) for _ in layers]
    return [_frame(names, raws, layers, by_id) for layers in groups], owner
