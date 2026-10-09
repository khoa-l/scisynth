"""Shape helpers for grids that grow as ``size ** ndim``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._numbers import as_int


def as_shape(value: Any, ndim: int, name: str = "shape") -> tuple[int, ...]:
    """Normalize an int or a sequence of ints to one positive size per axis.

    Parameters
    ----------
    value : int or sequence of int
        One size per axis, or a single size repeated for every axis.
    ndim : int
        Number of axes.
    name : str, default="shape"
        The setting's name, used in error messages, e.g. ``"resolution"``.

    Returns
    -------
    tuple of int
        One positive size per axis.

    Raises
    ------
    ValueError
        If the number of sizes does not match ``ndim`` or a size is not positive.
    """
    shape = tuple(as_int(n, name) for n in np.atleast_1d(value))
    if len(shape) == 1 and ndim > 1:
        shape = shape * ndim
    if len(shape) != ndim or any(n < 1 for n in shape):
        raise ValueError(f"Expected {ndim} positive sizes for {name}, got {value!r}")
    return shape


def check_cells(shape: Sequence[int], max_cells: int, name: str) -> int:
    """Return the number of cells in ``shape``, or raise if it exceeds ``max_cells``.

    Called before allocating, so a too-large request fails with a message instead
    of exhausting memory.

    Parameters
    ----------
    shape : sequence of int
        Cells per axis.
    max_cells : int
        Upper limit on the total number of cells.
    name : str
        Name of the setting to lower, used in the error message, e.g.
        ``"resolution"``.

    Returns
    -------
    int
        The number of cells.

    Raises
    ------
    ValueError
        If the number of cells exceeds ``max_cells``.
    """
    cells = math.prod(shape)
    if cells > max_cells:
        raise ValueError(
            f"{name} {tuple(shape)} means {cells:,} cells, over the limit of "
            f"{max_cells:,}. Lower {name}, or raise max_cells if you have the memory."
        )
    return cells
