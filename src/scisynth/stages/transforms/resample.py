"""Resampling that changes the number of locations: ``Downsample`` and ``Regrid``."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from ...core._coords import has_position
from ...core._numbers import as_int
from ...core._types import Array, BoolArray
from ...core.observation import Observation
from ...core.params import Param
from .._checks import per_axis, placed
from ..base import Stage


def _factors(value: Any, n_axes: int) -> tuple[int, ...]:
    factors = tuple(as_int(f, "factor") for f in per_axis(value, n_axes, "factor"))
    if min(factors, default=1) < 1:
        raise ValueError(f"factor must be a positive integer, got {value!r}")
    return factors


def _grid_bins(
    shape: Sequence[int], factors: Sequence[int]
) -> tuple[Array, tuple[int, ...]]:
    """Give each cell of a grid its block, and return the shape of the blocks."""
    blocks = tuple(-(-n // f) for n, f in zip(shape, factors, strict=True))
    cells = np.indices(shape).reshape(len(shape), -1)
    bins = np.ravel_multi_index(
        tuple(c // f for c, f in zip(cells, factors, strict=True)), blocks
    )
    return bins, blocks


def _point_bins(obs: Observation, factors: Sequence[int]) -> tuple[Array, tuple[int]]:
    """Give each point its bin, and return the output shape (the occupied bins).

    The bounding box of the points is cut into as many bins per axis as a grid of
    the same number of points would have blocks, ``ceil(n ** (1 / ndim) / factor)``.
    Only points with a position (finite coordinates) are binned and counted; the
    others, such as masked projection output, get bin ``-1``.
    """
    coords = np.stack([c.ravel() for c in obs.spatial_coords.values()])
    positioned = has_position(obs).ravel()
    bins = np.full(coords.shape[1], -1, dtype=np.intp)
    if not positioned.any():
        return bins, (0,)
    coords = coords[:, positioned]
    per_axis_n = max(1, round(coords.shape[1] ** (1 / len(coords))))
    cells = np.array([-(-per_axis_n // f) for f in factors])
    lo = coords.min(axis=1)
    span = np.where(coords.max(axis=1) > lo, coords.max(axis=1) - lo, 1.0)
    idx = np.floor((coords - lo[:, None]) / span[:, None] * cells[:, None]).astype(int)
    flat = np.ravel_multi_index(tuple(idx.clip(max=cells[:, None] - 1)), tuple(cells))
    occupied, bins[positioned] = np.unique(flat, return_inverse=True)
    return bins, (len(occupied),)


def _binning(
    obs: Observation, factors: Sequence[int]
) -> tuple[Observation, Array, tuple[int, ...], bool]:
    """Bin the locations of ``obs``, deciding grid or points once.

    Returns the observation to work on (points are flattened to a 1-D list), the
    bin of each of its locations in storage order, the shape of the output, and
    whether it is a grid.
    """
    on_grid = obs.layout == "grid"
    if on_grid:
        bins, shape = _grid_bins(obs.spatial_shape, factors)
        return obs, bins, shape, True
    obs = obs.to_points()
    bins, shape = _point_bins(obs, factors)
    return obs, bins, shape, False


def _average(bins: Array, n_bins: int, mask: BoolArray) -> sparse.csr_matrix:
    """Weights that average each bin over its valid locations.

    The matrix has shape ``(n_bins, mask.size)``; its row for a bin holds
    ``1 / n_valid`` at each valid location in the bin. Locations without a bin
    (``-1``) are never included.
    """
    valid = np.flatnonzero(mask.ravel() & (bins >= 0))
    counts = np.bincount(bins[valid], minlength=n_bins)
    weights = 1.0 / counts[bins[valid]]
    return sparse.csr_matrix((weights, (bins[valid], valid)), shape=(n_bins, mask.size))


@dataclass
class Downsample(Stage):
    """Reduce resolution by averaging nearby locations.

    On a grid, each output is the mean of the valid inputs in a block of cells. On
    points, the bounding box is cut into bins and each occupied bin gives one output
    point, so the result is a flat list of points. In both, the output's coordinates
    are the mean of its inputs' coordinates and ``truth`` is averaged over all of
    them.

    Parameters
    ----------
    factor : int, sequence of int or distribution, default=2
        Block size along every axis, or one per axis. For points it divides the
        resolution of a grid with as many points as there are locations.

    Raises
    ------
    ValueError
        If ``factor`` has the wrong length or is not a positive integer, or a valid
        location has a non-finite coordinate.

    Notes
    -----
    An output location is valid if any input in its bin is. At the edge of a grid
    whose size is not a multiple of ``factor``, the last block is smaller. The
    outputs get new ids above the current maximum; :meth:`links_from` ties each of them
    to the inputs it averaged, with weight ``1 / n_valid``.
    """

    factor: Param = 2

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        factors = _factors(params["factor"], len(obs.coords))
        placed(obs)
        obs, bins, shape, on_grid = _binning(obs, factors)
        n, d, n_bins = obs.size, obs.n_channels, math.prod(shape)
        weights = _average(bins, n_bins, obs.mask)
        everywhere = _average(bins, n_bins, np.ones(n, dtype=np.bool_))
        mask = np.asarray(weights.getnnz(axis=1) > 0)
        values = weights @ obs.values.reshape(n, d)
        values[~mask] = np.nan
        truth = None
        if obs.truth is not None:
            truth = (everywhere @ obs.truth.reshape(n, d)).reshape(*shape, d)
        if on_grid:
            coords = {}
            for (name, coord), f in zip(obs.coords.items(), factors, strict=True):
                block = np.arange(len(coord)) // f
                coords[name] = np.bincount(block, weights=coord) / np.bincount(block)
        else:
            coords = {k: everywhere @ c.ravel() for k, c in obs.coords.items()}
        first = int(obs.ids.max()) + 1 if obs.size else 0
        return obs.replace(
            values=values.reshape(*shape, d),
            coords=coords,
            mask=mask.reshape(shape),
            truth=truth,
            ids=first + np.arange(n_bins).reshape(shape),
        )

    def links_from(
        self, before: Observation, after: Observation, params: Mapping[str, Any]
    ) -> sparse.csr_matrix:
        factors = _factors(params["factor"], len(before.coords))
        before, bins, shape, _ = _binning(before, factors)
        return _average(bins, math.prod(shape), before.mask)


class Regrid(Stage):
    """Resample onto a new grid. Not implemented yet."""
