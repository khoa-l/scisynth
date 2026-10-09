"""Checks on stage inputs, shared by the stages."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core._coords import has_position
from ..core._types import BoolArray, FloatArray
from ..core.observation import Observation


def per_axis(value: Any, n_axes: int, name: str) -> FloatArray:
    """Broadcast a number, or check a sequence, to one float per coordinate axis.

    Parameters
    ----------
    value : float or sequence of float
        One number for every axis, or one per axis.
    n_axes : int
        Number of coordinate axes.
    name : str
        The parameter's name, for the error message.

    Returns
    -------
    ndarray of shape (n_axes,)
        The values.

    Raises
    ------
    ValueError
        If a sequence does not have ``n_axes`` entries.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n_axes, float(arr))
    if arr.shape != (n_axes,):
        raise ValueError(
            f"{name} must be a number or have one entry per axis ({n_axes}), "
            f"got shape {arr.shape}"
        )
    return arr


def placed(obs: Observation) -> BoolArray:
    """Return which locations have a position, raising if a valid one does not.

    Parameters
    ----------
    obs : Observation
        The observation to check.

    Returns
    -------
    ndarray of bool, spatial shape
        True where every coordinate is finite. Masked locations may be False (they
        have no position, e.g. masked projection output).

    Raises
    ------
    ValueError
        If a valid location has a NaN or infinite coordinate.
    """
    positioned = has_position(obs)
    bad = int((obs.mask & ~positioned).sum())
    if bad:
        raise ValueError(f"{bad} valid location(s) have non-finite coordinates")
    return positioned
