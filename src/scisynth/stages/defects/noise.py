"""Noise defects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.observation import Observation
from ...core.params import Param
from ..base import Stage


@dataclass
class GaussianNoise(Stage):
    """Add Gaussian noise to valid entries.

    Parameters
    ----------
    sigma : float, array-like or distribution, default=1.0
        Standard deviation, non-negative. An array broadcasts against ``values``,
        so one entry per channel acts per channel.
    mean : float, array-like or distribution, default=0.0
        Mean of the noise; broadcasts like ``sigma``.

    Raises
    ------
    ValueError
        If ``sigma`` is negative.

    Notes
    -----
    A full-shape array is always drawn, so the noise at a given position does not
    depend on which entries other stages masked out.
    """

    sigma: Param = 1.0
    mean: Param = 0.0

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        if np.any(np.asarray(params["sigma"]) < 0):
            raise ValueError("sigma must be non-negative")
        noise = rng.normal(params["mean"], params["sigma"], size=obs.values.shape)
        return obs.replace(values=np.where(obs.valid, obs.values + noise, obs.values))


@dataclass
class PoissonNoise(Stage):
    """Shot noise: treat valid values as rates and draw Poisson counts.

    Parameters
    ----------
    scale : float or distribution, default=1.0
        Counts per unit of value. The output is ``Poisson(values * scale) / scale``,
        so it stays unbiased and its relative noise falls as ``scale`` grows.

    Raises
    ------
    ValueError
        If ``scale`` is not positive or a valid value is negative.
    """

    scale: Param = 1.0

    def demo_observation(self) -> Observation:
        """Return non-negative rates (two bumps), since counts need rates >= 0.

        Returns
        -------
        Observation
            A 2-D grid with non-negative values.
        """
        from ..demo import bumps

        return bumps()

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        scale = float(params["scale"])
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        rate = np.where(obs.valid, obs.values * scale, 0.0)
        if np.any(rate < 0):
            raise ValueError("PoissonNoise requires non-negative values")
        counts = rng.poisson(rate)
        return obs.replace(values=np.where(obs.valid, counts / scale, obs.values))


class UniformNoise(Stage):
    """Additive uniform noise. Not implemented yet."""


class MultiplicativeNoise(Stage):
    """Multiplicative noise, ``values * (1 + eps)``. Not implemented yet."""
