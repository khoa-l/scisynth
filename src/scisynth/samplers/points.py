"""Scattered-point sampling of a field."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np

from ..core._numbers import as_int
from ..core.observation import Observation
from ..core.params import Param
from ..latent.base import FieldLatent, Kind, Latent
from .base import Sampler


@dataclass
class PointSampler(Sampler):
    """Evaluate a field at points drawn uniformly at random from its domain.

    Parameters
    ----------
    n : int or distribution
        Number of points.

    Notes
    -----
    The output is a points layout: every coordinate array has shape ``(n,)`` and
    ``values`` has shape ``(n, d)``.

    Examples
    --------
    >>> from scisynth.core import Domain
    >>> from scisynth.latent import AnalyticField
    >>> from scisynth.samplers import PointSampler
    >>> domain = Domain.from_extents([(0, 1), (0, 1)])
    >>> latent = AnalyticField(lambda x, y: x + y, domain)
    >>> obs = PointSampler(500).run(latent, seed=0)
    >>> obs.values.shape
    (500, 1)
    """

    n: Param
    accepts: ClassVar[frozenset[Kind]] = frozenset({"field"})

    def sample(
        self, latent: Latent, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        field = cast(FieldLatent, latent)
        domain = field.domain
        n = as_int(params["n"], "n")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        lo, hi = np.array(domain.extents).T
        points = rng.uniform(lo, hi, size=(n, domain.ndim))
        values = field.evaluate(points)
        return Observation(
            values=values,
            coords={name: points[:, i] for i, name in enumerate(domain.names)},
            mask=np.ones(n, dtype=np.bool_),
            truth=values.copy(),
        )


class JitteredSampler(Sampler):
    """Regular grid with random per-point jitter. Not implemented yet."""

    accepts: ClassVar[frozenset[Kind]] = frozenset({"field"})
