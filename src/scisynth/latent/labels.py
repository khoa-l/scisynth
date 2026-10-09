"""Label latents (stubs)."""

from __future__ import annotations

from typing import Any, ClassVar

from ..core.domain import Domain
from .base import Kind, LatentPlotMixin


class LabelLatent(LatentPlotMixin):
    """Discrete labels over a domain (segmentation, classes). Not implemented yet."""

    kind: ClassVar[Kind] = "label"
    domain: Domain

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError
