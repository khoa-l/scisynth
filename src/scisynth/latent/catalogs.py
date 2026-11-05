"""Catalog latents (stubs)."""

from __future__ import annotations

from typing import ClassVar

from ..core._unimplemented import Unimplemented
from ..core.domain import Domain
from .base import Kind, LatentGenerator, LatentPlotMixin


class CatalogLatent(Unimplemented, LatentPlotMixin):
    """A set of objects/events with positions and attributes. Not implemented yet."""

    kind: ClassVar[Kind] = "catalog"
    domain: Domain


class PointProcessLatent(Unimplemented, LatentGenerator):
    """Generator for point-process catalog realizations. Not implemented yet."""
