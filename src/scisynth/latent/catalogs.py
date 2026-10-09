"""Catalog latents (stubs)."""

from __future__ import annotations

from typing import Any, ClassVar

from ..core.domain import Domain
from .base import Kind, LatentGenerator, LatentPlotMixin


class CatalogLatent(LatentPlotMixin):
    """A set of objects/events with positions and attributes. Not implemented yet."""

    kind: ClassVar[Kind] = "catalog"
    domain: Domain

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError


class PointProcessLatent(LatentGenerator):
    """Generator for point-process catalog realizations. Not implemented yet."""
