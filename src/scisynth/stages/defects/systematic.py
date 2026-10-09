"""Systematic errors: the same distortion applied across the whole observation.

Parameters are resolved once per call, so a distribution gives one draw that is
then shared by every sample ("a fixed draw per run"). Arrays broadcast against
the values, so an array with one entry per channel acts per channel. Value defects
touch valid entries only; masked entries stay NaN.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core._numbers import as_index
from ...core._types import FloatArray
from ...core.observation import Observation
from ...core.params import Param
from .._checks import per_axis, placed
from ..base import Stage


@dataclass
class ValueOffset(Stage):
    """Add a constant to every valid entry.

    Parameters
    ----------
    value : float, array-like or distribution, default=0.0
        An array broadcasts against ``values``, so one entry per channel acts per
        channel.
    """

    value: Param = 0.0

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        return obs.replace(
            values=np.where(obs.valid, obs.values + params["value"], obs.values)
        )


@dataclass
class Gain(Stage):
    """Multiply every valid entry by a constant.

    Parameters
    ----------
    factor : float, array-like or distribution, default=1.0
        An array broadcasts against ``values``, so one entry per channel acts per
        channel. 1.0 leaves values unchanged.
    """

    factor: Param = 1.0

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        return obs.replace(
            values=np.where(obs.valid, obs.values * params["factor"], obs.values)
        )


@dataclass
class Drift(Stage):
    """Add a linear trend ``rate * t`` to every valid entry.

    Parameters
    ----------
    rate : float, array-like or distribution, default=0.0
        Change in value per unit of ``t``; one number or one per channel.
    axis : str or int, optional
        A coordinate name or an index into the coordinates (negative counts from the
        end). Then ``t`` is the coordinate measured from that axis' smallest value,
        so the drift is zero where the axis starts. Masked locations without a
        position (NaN coordinates, as in masked projection output) are ignored when
        finding the smallest value; a valid location without one raises. With None,
        ``t`` is the location's position in storage order (C order for grids) and
        ``rate`` is the change per location.

    Raises
    ------
    ValueError
        If ``axis`` is not one of the coordinates, or a valid location has a
        non-finite coordinate.
    """

    rate: Param = 0.0
    axis: str | int | None = None

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        drift = params["rate"] * self._position(obs, params["axis"])[..., None]
        return obs.replace(values=np.where(obs.valid, obs.values + drift, obs.values))

    @staticmethod
    def _position(obs: Observation, axis: str | int | None) -> FloatArray:
        """Return the per-location ``t``, with the spatial shape of ``obs``."""
        if axis is None:
            return np.arange(obs.size, dtype=np.float64).reshape(obs.mask.shape)
        names = list(obs.coords)
        if isinstance(axis, str):
            if axis not in obs.coords:
                raise ValueError(f"axis {axis!r} is not one of the coordinates {names}")
            index = names.index(axis)
        else:
            index = as_index(axis, len(names), "axis", f" for coordinates {names}")
        coord = np.asarray(obs.coords[names[index]], dtype=np.float64)
        if obs.layout == "grid":
            shape = [1] * obs.mask.ndim
            shape[index] = -1
            coord = np.broadcast_to(coord.reshape(shape), obs.mask.shape)
        ok = placed(obs)  # masked points may have no position (NaN)
        origin = coord[ok].min() if ok.any() else 0.0
        out: FloatArray = coord - origin
        return out


@dataclass
class CoordinateShift(Stage):
    """Shift the coordinates, like a registration or pointing error.

    Each value is reported at a location displaced from where it was measured.
    Values, mask and ``truth`` are untouched, so ``truth`` stays tied to the
    original locations.

    Parameters
    ----------
    shift : float, sequence of float or callable, default=0.0
        One number applied to every axis, or one per axis (a sequence, or a
        callable returning an array). A distribution gives one scalar shift.

    Raises
    ------
    ValueError
        If ``shift`` has the wrong length.
    """

    shift: Param = 0.0

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        shift = per_axis(params["shift"], len(obs.coords), "shift")
        coords = {
            name: np.asarray(coord, dtype=np.float64) + s
            for (name, coord), s in zip(obs.coords.items(), shift, strict=True)
        }
        return obs.replace(coords=coords)
