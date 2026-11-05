"""Selecting rows from a dataset (stubs)."""

from __future__ import annotations

from typing import ClassVar

from ..core._unimplemented import Unimplemented
from ..latent.base import Kind
from .base import Sampler


class SubsetSampler(Unimplemented, Sampler):
    """Select a subset of a dataset's rows. Not implemented yet."""

    accepts: ClassVar[frozenset[Kind]] = frozenset({"dataset"})
