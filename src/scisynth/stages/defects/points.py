"""Point-set defects: random errors in where points are, and extra points.

Both turn a grid into a flat list of points first, since a grid cannot hold them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core._numbers import as_int
from ...core.observation import Observation
from ...core.params import Param
from .._checks import per_axis
from ..base import Stage


@dataclass
class PositionJitter(Stage):
    """Displace every point by independent Gaussian noise, like localization error.

    Values, mask, ids and ``truth`` are untouched, so ``truth`` stays tied to the
    original positions. A grid becomes a flat list of points
    (:meth:`Observation.to_points <scisynth.core.observation.Observation.to_points>`).

    Parameters
    ----------
    sigma : float, sequence of float or distribution, default=0.0
        Standard deviation of the displacement: one number for every axis, or one
        per axis. A distribution gives one scalar for the whole run.

    Raises
    ------
    ValueError
        If ``sigma`` has the wrong length or is negative.

    Notes
    -----
    A displacement is drawn for every location, masked ones included, so the random
    stream does not depend on which locations are valid.
    """

    sigma: Param = 0.0

    def demo_observation(self) -> Observation:
        from ..demo import scatter

        return scatter()

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        obs = obs.to_points()
        n = len(obs.coords)
        sigma = per_axis(params["sigma"], n, "sigma")
        if np.any(sigma < 0):
            raise ValueError("sigma must be non-negative")
        noise = rng.normal(size=(n, *obs.mask.shape))
        coords = {
            name: np.asarray(coord, dtype=np.float64) + s * z
            for (name, coord), s, z in zip(
                obs.coords.items(), sigma, noise, strict=True
            )
        }
        return obs.replace(coords=coords)


@dataclass
class RandomInsertion(Stage):
    """Insert ``n`` spurious points at random positions, like false detections.

    Parameters
    ----------
    n : int or distribution, default=10
        Number of points to add; must be a whole number.

    Raises
    ------
    ValueError
        If the observation has no valid location or ``n`` is negative.

    Notes
    -----
    A grid becomes a flat list of points first. New points are appended, uniform in
    the bounding box of the valid coordinates, with values uniform in each channel's
    valid range. They get fresh ids above the current maximum and ``truth`` is NaN
    there: nothing real is being measured.
    """

    n: Param = 10

    def demo_observation(self) -> Observation:
        from ..demo import scatter

        return scatter()

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        obs = obs.to_points()
        n = as_int(params["n"], "n")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if not obs.mask.any():
            raise ValueError("RandomInsertion needs at least one valid location")
        unit = rng.random((len(obs.coords), n))
        fresh = rng.random((n, obs.n_channels))

        coords = {}
        for (name, coord), u in zip(obs.coords.items(), unit, strict=True):
            lo, hi = coord[obs.mask].min(), coord[obs.mask].max()
            coords[name] = np.concatenate([coord, lo + (hi - lo) * u])
        valid = obs.values[obs.mask]
        lo, hi = valid.min(axis=0), valid.max(axis=0)
        values = np.concatenate([obs.values, lo + (hi - lo) * fresh])
        truth = (
            None
            if obs.truth is None
            else np.concatenate([obs.truth, np.full((n, obs.n_channels), np.nan)])
        )
        return obs.replace(
            values=values,
            coords=coords,
            mask=np.concatenate([obs.mask, np.ones(n, dtype=np.bool_)]),
            truth=truth,
            ids=np.concatenate([obs.ids, obs.ids.max() + 1 + np.arange(n)]),
        )
