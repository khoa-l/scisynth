"""Checks on the coordinates of an observation."""

from __future__ import annotations

import numpy as np

from ._types import BoolArray
from .observation import Observation


def has_position(obs: Observation) -> BoolArray:
    """Return which locations have a position: every coordinate is finite.

    Masked locations may have none (NaN coordinates), as in masked projection
    output.

    Parameters
    ----------
    obs : Observation
        The observation to check.

    Returns
    -------
    ndarray of bool, spatial shape
        True where every coordinate is finite.
    """
    finite = [np.isfinite(c) for c in obs.spatial_coords.values()]
    return np.asarray(np.logical_and.reduce(finite), dtype=np.bool_)
