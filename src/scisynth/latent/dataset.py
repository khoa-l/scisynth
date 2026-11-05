"""Tabular dataset latents (stubs)."""

from __future__ import annotations

from typing import ClassVar

from ..core._unimplemented import Unimplemented
from ..core.domain import Domain
from .base import Kind, LatentPlotMixin


class DatasetLatent(Unimplemented, LatentPlotMixin):
    """Tabular data that can be selected or binned. Not implemented yet."""

    kind: ClassVar[Kind] = "dataset"
    domain: Domain
