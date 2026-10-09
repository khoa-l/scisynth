"""Plots for components: a domain, a sampler's output, a stage's effect."""

from __future__ import annotations

from typing import Any

from ..core.domain import Domain
from ..core.observation import Observation
from ..core.rng import SeedLike
from ..latent.base import Latent
from ..samplers.base import Sampler
from ..stages.base import Stage
from .registry import get_plot, register_plot


@register_plot("domain", "outline")
def plotly_domain(domain: Domain, **kwargs: Any) -> Any:
    """Plot a domain's extent. Not implemented yet.

    Parameters
    ----------
    domain : Domain
        The domain to plot.
    **kwargs
        Passed to the plot function.
    """
    raise NotImplementedError


@register_plot("sampler", "*")
def plot_sampler(
    sampler: Sampler,
    latent: Latent,
    *,
    kind: str,
    seed: SeedLike = 0,
    **kwargs: Any,
) -> Any:
    """Sample ``latent`` with ``sampler`` and plot the observation.

    Parameters
    ----------
    sampler : Sampler
        The sampler to run.
    latent : Latent
        The ground truth to sample.
    kind : str
        How to draw the observation (``"scatter"`` or ``"heatmap"``). Filled in by
        the registry.
    seed : int, SeedSequence or None, default=0
        Seed for the sampler's random stream.
    **kwargs
        Passed to the observation plot.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure of the sampled observation.

    Raises
    ------
    NotImplementedError
        If the observation does not have exactly two coordinates.
    """
    obs = sampler.run(latent, seed)
    return get_plot("observation", kind)(obs, type(sampler).__name__, **kwargs)


@register_plot("stage", "*")
def plot_stage(
    stage: Stage,
    obs: Observation,
    *,
    kind: str,
    seed: SeedLike = 0,
    **kwargs: Any,
) -> Any:
    """Apply ``stage`` to ``obs`` and plot before and after side by side.

    Parameters
    ----------
    stage : Stage
        The stage to apply.
    obs : Observation
        The observation before the stage.
    kind : str
        How to draw each panel (``"scatter"`` or ``"heatmap"``). Filled in by the
        registry.
    seed : int, SeedSequence or None, default=0
        Seed for the stage's random stream.
    **kwargs
        Passed to the ``panels`` plot.

    Returns
    -------
    plotly.graph_objects.Figure
        Two panels: before and after.

    Raises
    ------
    NotImplementedError
        If the observation before or after the stage does not have exactly two
        coordinates.
    """
    after = stage.run(obs, seed)
    panels = get_plot("panels", kind)
    return panels([obs, after], ["before", f"after {type(stage).__name__}"], **kwargs)
