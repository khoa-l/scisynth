"""Quantization and range defects: quantizing, hard clipping, soft saturation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core._numbers import as_int
from ...core.observation import Observation
from ...core.params import Param
from ..base import Stage


@dataclass
class Quantize(Stage):
    """Round valid values to evenly spaced levels, like an ADC.

    Parameters
    ----------
    levels : int, default=256
        Number of levels; at least 2.
    vmin, vmax : float or array-like, optional
        Range covered by the levels. By default it is the range of the valid data,
        taken per channel. Values outside are clipped to the end levels. Give one
        value per channel to set per-channel ranges.

    Raises
    ------
    ValueError
        If ``levels`` is less than 2, or ``vmin`` is not below ``vmax``.

    Notes
    -----
    The result stays float: the level values, not integer codes. A channel whose
    data range is empty is left unchanged. It uses no randomness.
    """

    levels: Param = 256
    vmin: Param | None = None
    vmax: Param | None = None

    def demo_observation(self) -> Observation:
        """Return a monotone gradient, so levels show up as discrete bands.

        Returns
        -------
        Observation
            A 2-D grid with a diagonal gradient.
        """
        from ..demo import ramp

        return ramp()

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        levels = as_int(params["levels"], "levels")
        if levels < 2:
            raise ValueError(f"levels must be >= 2, got {levels}")
        if (
            params["vmin"] is not None
            and params["vmax"] is not None
            and np.any(np.asarray(params["vmin"]) >= np.asarray(params["vmax"]))
        ):
            raise ValueError(
                f"vmin must be below vmax, got vmin={params['vmin']!r}, "
                f"vmax={params['vmax']!r}"
            )
        valid = obs.values[obs.mask]  # (n_valid, d)
        if valid.shape[0] == 0:
            return obs.replace()
        lo = (
            valid.min(axis=0)
            if params["vmin"] is None
            else np.asarray(params["vmin"], float)
        )
        hi = (
            valid.max(axis=0)
            if params["vmax"] is None
            else np.asarray(params["vmax"], float)
        )
        flat = hi <= lo
        step = np.where(flat, 1.0, (hi - lo) / (levels - 1))
        codes = np.clip(np.round((obs.values - lo) / step), 0, levels - 1)
        quantized = np.where(flat, obs.values, lo + codes * step)
        return obs.replace(values=np.where(obs.valid, quantized, obs.values))


@dataclass
class Clip(Stage):
    """Hard-clamp valid values to ``[lo, hi]``, like a sensor's fixed range.

    Parameters
    ----------
    lo, hi : float, array-like or distribution, optional
        Bounds; None leaves that side unbounded, and with neither bound nothing
        changes. Arrays broadcast against ``values``, so one entry per channel acts
        per channel.

    Raises
    ------
    ValueError
        If ``lo`` exceeds ``hi`` anywhere.

    See Also
    --------
    Saturate : A smooth, soft limit instead of a hard clamp.

    Notes
    -----
    Clamped values stay valid.
    """

    lo: Param | None = None
    hi: Param | None = None

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        lo, hi = params["lo"], params["hi"]
        if (
            lo is not None
            and hi is not None
            and np.any(np.asarray(lo) > np.asarray(hi))
        ):
            raise ValueError(f"lo must not exceed hi, got lo={lo!r}, hi={hi!r}")
        return obs.replace(
            values=np.where(obs.valid, np.clip(obs.values, lo, hi), obs.values)
        )


@dataclass
class Saturate(Stage):
    """Soft saturation, like a detector approaching full well.

    Computes ``ceiling * tanh(value / ceiling)`` on valid entries: the identity near
    zero (slope 1), flattening towards ``+-ceiling`` for large ``|value|``. The
    response is smooth, monotonic and odd, so signs are kept.

    Parameters
    ----------
    ceiling : float, array-like or distribution, default=1.0
        Positive limit. An array broadcasts against ``values``.

    Raises
    ------
    ValueError
        If ``ceiling`` is not positive.

    See Also
    --------
    Clip : A hard clamp instead of a soft limit.
    """

    ceiling: Param = 1.0

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        ceiling = np.asarray(params["ceiling"], dtype=np.float64)
        if np.any(ceiling <= 0):
            raise ValueError(f"ceiling must be > 0, got {params['ceiling']!r}")
        soft = ceiling * np.tanh(obs.values / ceiling)
        return obs.replace(values=np.where(obs.valid, soft, obs.values))
