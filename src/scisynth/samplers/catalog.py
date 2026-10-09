"""Sampling catalogs of objects or events (stubs)."""

from __future__ import annotations

from typing import ClassVar

from ..latent.base import Kind
from .base import Sampler


class CatalogSampler(Sampler):
    """Render catalog objects onto a grid or count events. Not implemented yet."""

    accepts: ClassVar[frozenset[Kind]] = frozenset({"catalog"})
