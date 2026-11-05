"""Label latents (stubs)."""

from __future__ import annotations

from typing import ClassVar

from ..core._unimplemented import Unimplemented
from ..core.domain import Domain
from .base import Kind, LatentPlotMixin


class LabelLatent(Unimplemented, LatentPlotMixin):
    """Discrete labels over a domain (segmentation, classes). Not implemented yet."""

    kind: ClassVar[Kind] = "label"
    domain: Domain
