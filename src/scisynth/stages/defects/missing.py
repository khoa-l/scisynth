"""Missing-data defects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.observation import Observation
from ...core.params import Param
from ..base import Stage


@dataclass
class RandomDropout(Stage):
    """Independently drop each location with probability ``p``.

    Parameters
    ----------
    p : float or distribution, default=0.1
        Drop probability, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``p`` is outside ``[0, 1]``.

    Notes
    -----
    Dropped locations become NaN in every channel and are masked out. A uniform
    array with the spatial shape is always drawn, so locations dropped at a low
    ``p`` stay dropped at a higher ``p``.
    """

    p: Param = 0.1

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        prob = float(params["p"])
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {prob}")
        keep = obs.mask & (rng.random(obs.mask.shape) >= prob)
        return obs.replace(
            values=np.where(keep[..., None], obs.values, np.nan), mask=keep
        )


class BlockGaps(Stage):
    """Contiguous blocks of missing data. Not implemented yet."""


class RegionMask(Stage):
    """Mask out entries inside given regions. Not implemented yet."""
