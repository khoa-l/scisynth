"""Array-backed gridded latents (stubs)."""

from __future__ import annotations

from typing import Any, ClassVar

from ..core.domain import Domain
from .base import Kind, LatentPlotMixin


class GriddedLatent(LatentPlotMixin):
    """Data on a fixed regular grid, interpolated when queried. Not implemented yet."""

    kind: ClassVar[Kind] = "gridded"
    domain: Domain

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError
