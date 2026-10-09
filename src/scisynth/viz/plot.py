"""Convenience wrappers over the registry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .registry import get_plot, resolve_plot

if TYPE_CHECKING:
    from ..core.observation import Observation
    from ..latent.base import Latent


def plot_observations(
    observations: Sequence[Observation],
    titles: Sequence[str] | None = None,
    *,
    kind: str = "scatter",
    **kwargs: Any,
) -> Any:
    """Plot 2-D observations side by side (masked entries are not drawn).

    Panels share one color scale and one coordinate frame.

    Parameters
    ----------
    observations : sequence of Observation
        The observations to draw, one per panel.
    titles : sequence of str, optional
        One title per panel.
    kind : {"scatter", "heatmap"}, default="scatter"
        How each panel is drawn. A heatmap averages points into bins.
    **kwargs
        ``channel`` (default 0) picks the channel drawn, ``interactive=False``
        disables zoom, pan, selection and hover, and the rest go to
        ``fig.update_layout``.

    Returns
    -------
    plotly.graph_objects.Figure
        One figure with a subplot per observation.
    """
    return get_plot("panels", kind)(observations, titles, **kwargs)


def plot_latent(
    latent: Latent,
    *,
    kind: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot a realized latent.

    Parameters
    ----------
    latent : Latent
        A realized latent.
    kind : str, optional
        How to draw it, one of the kinds the latent offers; by default the first
        that has a plot.
    **kwargs
        Passed to the plot function; for fields, ``shape`` sets the evaluation grid.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure of the latent.

    Notes
    -----
    The field plot is not implemented yet.
    """
    return resolve_plot(latent, kind)(latent, **kwargs)
