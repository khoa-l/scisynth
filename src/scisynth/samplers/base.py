"""Sampler base class."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from ..core.component import Component
from ..core.observation import Observation
from ..core.rng import SeedLike, spawn_rngs
from ..latent.base import Kind, Latent


class IncompatibleLatentError(TypeError):
    """A sampler was given a latent of a kind it does not accept."""


class Sampler(Component, register=False):
    """Base class for samplers: query a latent once to make an Observation.

    Subclasses set ``accepts`` (the latent kinds they can query) and implement
    :meth:`sample`. A sampler only queries; layout-changing steps after it are
    ordinary stages. Calling a sampler, ``sampler(latent, rng)``, checks the latent,
    resolves the parameters, samples, and appends a record to ``meta``.
    """

    accepts: ClassVar[frozenset[Kind]] = frozenset()
    plot_subjects: ClassVar[tuple[str, ...]] = ("sampler",)
    plot_kinds: ClassVar[tuple[str, ...]] = ("scatter", "heatmap")

    def check(self, latent: Latent) -> None:
        """Raise if ``latent`` is of a kind this sampler does not accept.

        Parameters
        ----------
        latent : Latent
            The latent to check.

        Raises
        ------
        IncompatibleLatentError
            If ``latent.kind`` is not in :attr:`accepts`.
        """
        if latent.kind not in self.accepts:
            raise IncompatibleLatentError(
                f"{type(self).__name__} accepts latents of kind "
                f"{sorted(self.accepts)}, "
                f"got kind {latent.kind!r} ({type(latent).__name__})"
            )

    def sample(
        self, latent: Latent, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        """Query ``latent`` (implemented by subclasses).

        Parameters
        ----------
        latent : Latent
            The ground truth to query.
        rng : numpy.random.Generator
            Source of randomness for this call.
        params : mapping
            The resolved parameters.

        Returns
        -------
        Observation
            The sampled observation, before the sampler's own ``meta`` record is
            appended.
        """
        raise NotImplementedError

    def __call__(self, latent: Latent, rng: np.random.Generator) -> Observation:
        self.check(latent)
        params = self.resolved_params(rng)
        obs = self.sample(latent, rng, params)
        return obs.replace(meta=[*obs.meta, self.log_entry(params)])

    def run(self, latent: Latent, seed: SeedLike = None) -> Observation:
        """Sample ``latent`` from a seed.

        Parameters
        ----------
        latent : Latent
            The ground truth to sample.
        seed : int, SeedSequence or None, default=None
            Seed for the random stream; None draws fresh OS entropy.

        Returns
        -------
        Observation
            The sampled observation.

        Notes
        -----
        Uses the same random stream an :class:`~scisynth.observer.observer.Observer`
        gives its sampler. For one seed the result matches
        ``Observer(sampler).run(latent, seed)``, apart from the step name in ``meta``.
        """
        return self(latent, spawn_rngs(seed, 1)[0])

    def plot(
        self,
        latent: Latent,
        seed: SeedLike = 0,
        *,
        kind: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Sample ``latent`` with this sampler and plot the observation.

        Parameters
        ----------
        latent : Latent
            The ground truth to sample.
        seed : int, SeedSequence or None, default=0
            Seed for the random stream; the default keeps plots repeatable.
        kind : str, optional
            How to draw it: ``"scatter"`` (the default) or ``"heatmap"``. Points are
            averaged into bins for a heatmap.
        **kwargs
            Passed to the plot function (for plotly, ``channel`` and
            ``interactive``, then ``fig.update_layout``).

        Returns
        -------
        plotly.graph_objects.Figure
            The figure of the sampled observation.
        """
        from ..viz import plot

        return plot(self, latent, seed=seed, kind=kind, **kwargs)
