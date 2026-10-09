"""Regular-grid sampling of a field."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np

from ..core._grid import as_shape, check_cells
from ..core._numbers import as_int
from ..core.observation import Observation
from ..latent.base import FieldLatent, Kind, Latent
from .base import Sampler

MAX_GRID_CELLS = 10_000_000


@dataclass
class GridSampler(Sampler):
    """Evaluate a field on a regular grid spanning its domain (endpoints included).

    Parameters
    ----------
    shape : int or sequence of int
        Samples per axis; an int is repeated for every axis.
    max_cells : int, default=10_000_000
        Upper limit on ``prod(shape)``, checked before allocating. The grid grows as
        ``n ** ndim``, so use a :class:`PointSampler` for high-dimensional domains.

    Notes
    -----
    The output is a grid layout: one 1-D coordinate array per axis, and
    ``values`` of shape ``(*shape, d)``.

    Examples
    --------
    >>> from scisynth.core import Domain
    >>> from scisynth.latent import AnalyticField
    >>> from scisynth.samplers import GridSampler
    >>> domain = Domain.from_extents([(0, 1), (0, 1)])
    >>> latent = AnalyticField(lambda x, y: x + y, domain)
    >>> obs = GridSampler((64, 64)).run(latent, seed=0)
    >>> obs.values.shape
    (64, 64, 1)
    """

    shape: int | Sequence[int]
    max_cells: int = MAX_GRID_CELLS
    accepts: ClassVar[frozenset[Kind]] = frozenset({"field"})

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.shape, int):
            self.shape = tuple(as_int(n, "shape") for n in self.shape)

    def sample(
        self, latent: Latent, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        field = cast(FieldLatent, latent)
        domain = field.domain
        shape = as_shape(params["shape"], domain.ndim)
        check_cells(shape, self.max_cells, "shape")
        axes = [
            np.linspace(lo, hi, n)
            for (lo, hi), n in zip(domain.extents, shape, strict=True)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=-1)
        values = field.evaluate(points)
        values = values.reshape((*shape, values.shape[-1]))
        return Observation(
            values=values,
            coords=dict(zip(domain.names, axes, strict=True)),
            mask=np.ones(shape, dtype=np.bool_),
            truth=values.copy(),
        )
