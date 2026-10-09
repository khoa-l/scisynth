"""Plots for realized latents, registered by latent kind (stubs)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..latent.base import Latent
from .registry import register_plot


@register_plot("field", "*")
def plot_field(
    latent: Latent,
    *,
    kind: str,
    shape: int | Sequence[int] | None = None,
    **kwargs: Any,
) -> Any:
    """Plot a field evaluated on a grid, drawn as ``kind``. Not implemented yet."""
    raise NotImplementedError
