"""Array-backed gridded latents (stubs)."""

from __future__ import annotations

from typing import ClassVar

from ..core._unimplemented import Unimplemented
from ..core.domain import Domain
from .base import Kind, LatentPlotMixin


class GriddedLatent(Unimplemented, LatentPlotMixin):
    """Data on a fixed regular grid, interpolated when queried. Not implemented yet."""

    kind: ClassVar[Kind] = "gridded"
    domain: Domain
