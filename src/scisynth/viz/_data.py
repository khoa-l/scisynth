"""Helpers that do not draw: flattening observations, lazy imports."""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np

from ..core._numbers import as_index
from ..core._types import Array
from ..core.observation import Observation


def require(module: str, extra: str = "viz") -> Any:
    """Import ``module`` now, or raise ``ImportError`` with an install hint.

    Plot functions call this themselves so importing :mod:`scisynth.viz` never needs
    plotly.

    Parameters
    ----------
    module : str
        Dotted module name, e.g. ``"plotly.graph_objects"``.
    extra : str, default="viz"
        The scisynth extra named in the hint.

    Returns
    -------
    module
        The imported module.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        if (exc.name or module).split(".")[0] != module.split(".")[0]:
            raise
        raise ImportError(
            f"{module.split('.')[0]} is required for these plots; install it "
            f'with `pip install "scisynth[{extra}]"`.'
        ) from exc


def flatten(obs: Observation, channel: int = 0) -> tuple[dict[str, Array], Array]:
    """Flatten an observation to its valid locations.

    Parameters
    ----------
    obs : Observation
        The observation to flatten.
    channel : int, default=0
        Which channel to return.

    Returns
    -------
    coords : dict of str to ndarray
        One flat column per coordinate.
    values : ndarray
        The chosen channel at the valid locations.

    Raises
    ------
    ValueError
        If ``channel`` is out of range.
    """
    channel = as_index(
        channel, obs.n_channels, "channel", f" for {obs.n_channels} channel(s)"
    )
    valid = obs.mask.ravel()
    cols = {n: a.ravel()[valid] for n, a in obs.spatial_coords.items()}
    return cols, obs.values[..., channel].ravel()[valid]


def default_bins(n_points: int) -> int:
    """Return a number of bins per axis that holds about four points per bin.

    Parameters
    ----------
    n_points : int
        How many points there are.

    Returns
    -------
    int
        ``sqrt(n_points / 4)`` rounded, and at least 1.
    """
    return max(1, round(math.sqrt(n_points / 4)))


def _mean_per_bin(
    flat: Array, values: Array, shape: tuple[int, int]
) -> tuple[Array, Array]:
    """Mean value and count per bin, given each value's flat bin number.

    Bins with no value have the mean NaN.
    """
    size = shape[0] * shape[1]
    counts = np.bincount(flat, minlength=size)
    sums = np.bincount(flat, weights=values, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    return means.reshape(shape), counts.reshape(shape)


def bin_means(
    x: Array, y: Array, values: Array, bins: tuple[int, int]
) -> tuple[Array, Array, Array, Array]:
    """Average ``values`` over a regular grid of bins laid over the points.

    Parameters
    ----------
    x, y : ndarray of shape (n,)
        Finite positions of the points.
    values : ndarray of shape (n,)
        The value at each point.
    bins : (int, int)
        Bins along x and along y. The grid spans the bounding box of the points.

    Returns
    -------
    x_centers, y_centers : ndarray
        The center of each bin along x and y.
    means : ndarray of shape (bins[0], bins[1])
        Mean value in each bin, indexed ``[i, j]`` at ``(x_centers[i],
        y_centers[j])``. NaN where a bin holds no point.
    counts : ndarray of shape (bins[0], bins[1])
        How many points fall in each bin.
    """
    centers: list[Array] = []
    index: list[Array] = []
    for coord, n in zip((x, y), bins, strict=True):
        lo, hi = (coord.min(), coord.max()) if coord.size else (0.0, 1.0)
        if hi <= lo:  # all at one place: give the bin some width
            lo, hi = lo - 0.5, hi + 0.5
        centers.append(lo + (np.arange(n) + 0.5) * (hi - lo) / n)
        index.append(np.minimum(((coord - lo) / (hi - lo) * n).astype(np.intp), n - 1))
    means, counts = _mean_per_bin(index[0] * bins[1] + index[1], values, bins)
    return centers[0], centers[1], means, counts


def block_means(
    x: Array, y: Array, values: Array, mask: Array, bins: tuple[int, int]
) -> tuple[Array, Array, Array, Array]:
    """Average the cells of a grid into at most ``bins`` blocks per axis.

    Neighbouring cells are grouped into blocks that differ in size by at most one
    cell. Only valid cells count.

    Parameters
    ----------
    x, y : ndarray
        The grid's axis coordinates, with ``len(x)`` and ``len(y)`` cells.
    values : ndarray of shape (len(x), len(y))
        One value per cell.
    mask : ndarray of bool, same shape
        True where the cell is valid.
    bins : (int, int)
        Blocks along x and y, capped at the number of cells on that axis.

    Returns
    -------
    x_centers, y_centers : ndarray
        The mean coordinate of the cells in each block.
    means : ndarray of shape (bx, by)
        Mean value of the valid cells in each block, NaN where there are none.
    counts : ndarray of shape (bx, by)
        How many valid cells each block holds.
    """
    nx, ny = values.shape
    bx, by = min(bins[0], nx), min(bins[1], ny)
    gx, gy = np.arange(nx) * bx // nx, np.arange(ny) * by // ny  # block of each cell
    flat = (gx[:, None] * by + gy[None, :])[mask]
    means, counts = _mean_per_bin(flat, values[mask], (bx, by))
    x_centers = np.bincount(gx, weights=x) / np.bincount(gx)
    y_centers = np.bincount(gy, weights=y) / np.bincount(gy)
    return x_centers, y_centers, means, counts
