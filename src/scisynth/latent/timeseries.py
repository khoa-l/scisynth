"""Time-series latents (stubs)."""

from __future__ import annotations

from typing import ClassVar

from ..core._unimplemented import Unimplemented
from ..core.domain import Domain
from .base import Kind, LatentGenerator, LatentPlotMixin


class TimeSeriesLatent(Unimplemented, LatentPlotMixin):
    """A 1-D series over a time axis, queryable at any time. Not implemented yet."""

    kind: ClassVar[Kind] = "field"
    domain: Domain


class OrnsteinUhlenbeck(Unimplemented, LatentGenerator):
    """Generator for Ornstein-Uhlenbeck process realizations. Not implemented yet."""
