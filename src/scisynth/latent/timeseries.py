"""Time-series latents (stubs)."""

from __future__ import annotations

from typing import Any, ClassVar

from ..core.domain import Domain
from .base import Kind, LatentGenerator, LatentPlotMixin


class TimeSeriesLatent(LatentPlotMixin):
    """A 1-D series over a time axis, queryable at any time. Not implemented yet."""

    kind: ClassVar[Kind] = "field"
    domain: Domain

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError


class OrnsteinUhlenbeck(LatentGenerator):
    """Generator for Ornstein-Uhlenbeck process realizations. Not implemented yet."""
