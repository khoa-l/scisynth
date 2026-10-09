"""Binning dataset rows (stubs)."""

from __future__ import annotations

from typing import ClassVar

from ..latent.base import Kind
from .base import Sampler


class BinSampler(Sampler):
    """Aggregate a dataset's rows into bins. Not implemented yet."""

    accepts: ClassVar[frozenset[Kind]] = frozenset({"dataset"})
