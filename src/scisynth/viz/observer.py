"""Plot of a whole observer: its latent and every step's output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.rng import SeedLike
from ..latent.base import Latent
from ..observer.observer import Observer
from ..samplers.grid import GridSampler
from .registry import get_plot, register_plot

LATENT_GRID = 64


@register_plot("observer", "*")
def plot_observer(
    observer: Observer,
    latent: Latent,
    *,
    kind: str,
    seed: SeedLike = 0,
    show_latent: bool = True,
    shape: int | Sequence[int] | None = None,
    **kwargs: Any,
) -> Any:
    """Plot every step of an observer: the latent, then each step's output.

    Runs :meth:`~scisynth.observer.observer.Observer.trace` and draws the
    observations as a grid of panels, titled with the step names. The layout (at
    most four panels per row, one shared color bar) is that of the ``panels`` plot.

    Parameters
    ----------
    observer : Observer
        The pipeline to run.
    latent : Latent
        The ground truth to observe.
    kind : {"scatter", "heatmap"}
        How each panel is drawn. Filled in by the registry.
    seed : int, SeedSequence or None, default=0
        Seed for the observer's random streams.
    show_latent : bool, default=True
        Lead with a panel showing the latent itself, evaluated on a regular grid.
    shape : int or sequence of int, optional
        Grid used to evaluate the latent; defaults to 64 per axis.
    **kwargs
        Passed to the ``panels`` plot, e.g. ``channel``, ``ncols`` or
        ``interactive``.

    Returns
    -------
    plotly.graph_objects.Figure
        One panel per step, plus the latent panel if requested.
    """
    trace = observer.trace(latent, seed)
    panels = get_plot("panels", kind)
    if not show_latent:
        return panels(trace, **kwargs)
    grid = GridSampler(LATENT_GRID if shape is None else shape)
    return panels([grid.run(latent), *trace], ["latent", *trace.names], **kwargs)
