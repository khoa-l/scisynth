"""Whole-number inputs: integers and positions that may count from the end."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_int(value: Any, name: str) -> int:
    """Return ``value`` as an ``int``, or raise if it is not a whole number.

    Parameters
    ----------
    value : object
        An integer, or a float with an integral value such as ``3.0`` (what a JSON
        round trip or a distribution can give). Booleans, other types, non-finite
        and fractional numbers, and arrays with more than one element are rejected.
    name : str
        The parameter's name, for the error message.

    Returns
    -------
    int
        The value.

    Raises
    ------
    ValueError
        If ``value`` is not a whole number.

    Examples
    --------
    >>> from scisynth.core._numbers import as_int
    >>> as_int(3.0, "n")
    3
    """
    arr = np.asarray(value)
    if arr.ndim != 0 or arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if arr.dtype.kind == "f" and not (np.isfinite(arr) and arr == np.round(arr)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(arr)


def as_index(index: Any, size: int, name: str, where: str = "") -> int:
    """Return ``index`` counted from the start, accepting negative numbers.

    Like list indexing, ``-1`` is the last of ``size`` items.

    Parameters
    ----------
    index : int
        The position, from ``-size`` to ``size - 1``.
    size : int
        How many items there are.
    name : str
        What the index picks (``"channel"``, ``"axis"``), for the error message.
    where : str, default=""
        Added to the end of the error message, e.g. ``" for step 'noise'"``.

    Returns
    -------
    int
        The position from ``0`` to ``size - 1``.

    Raises
    ------
    ValueError
        If ``index`` is not a whole number or is out of range.

    Examples
    --------
    >>> from scisynth.core._numbers import as_index
    >>> as_index(-1, 3, "channel")
    2
    """
    position = as_int(index, name)
    if not -size <= position < size:
        raise ValueError(f"{name} {position} is out of range{where}")
    return position % size
