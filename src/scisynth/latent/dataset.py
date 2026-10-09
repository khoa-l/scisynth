"""Tabular dataset latents (stubs)."""

from __future__ import annotations

from typing import Any, ClassVar

from ..core.domain import Domain
from .base import Kind, LatentPlotMixin


class DatasetLatent(LatentPlotMixin):
    """Tabular data that can be selected or binned. Not implemented yet."""

    kind: ClassVar[Kind] = "dataset"
    domain: Domain

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError
