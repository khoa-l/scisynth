"""Stage base class."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from scipy import sparse

from ..core.component import Component
from ..core.observation import Observation
from ..core.rng import SeedLike, spawn_rngs


class Stage(Component, register=False):
    """Base class for stages: transform an Observation into another.

    Subclasses implement :meth:`apply`, which receives the already-resolved
    parameters and must not mutate its input. Calling a stage, ``stage(obs, rng)``,
    resolves the parameters, applies, and appends a record to ``meta``. Defects and
    transforms are both stages; the folder split is only for documentation.
    """

    plot_subjects: ClassVar[tuple[str, ...]] = ("stage",)
    plot_kinds: ClassVar[tuple[str, ...]] = ("scatter", "heatmap")

    def apply(
        self, obs: Observation, rng: np.random.Generator, params: Mapping[str, Any]
    ) -> Observation:
        """Transform ``obs`` (implemented by subclasses).

        Parameters
        ----------
        obs : Observation
            The input; must not be mutated.
        rng : numpy.random.Generator
            Source of randomness for this call.
        params : mapping
            The resolved parameters.

        Returns
        -------
        Observation
            The transformed observation, before this stage's ``meta`` record is
            appended.
        """
        raise NotImplementedError

    def links_from(
        self, before: Observation, after: Observation, params: Mapping[str, Any]
    ) -> sparse.csr_matrix | None:
        """Return the links from ``before`` to ``after``, or None to match by ids.

        Override in stages that merge locations. The default, None, means each
        location of ``after`` comes from the location of ``before`` with its id (see
        :func:`~scisynth.observer.links.links_by_id`).

        Parameters
        ----------
        before : Observation
            The input.
        after : Observation
            The output of this stage, with this stage's record last in ``meta``.
        params : mapping
            The resolved parameters of the call that produced ``after``.

        Returns
        -------
        scipy.sparse.csr_matrix or None
            The links, with shape ``(after.size, before.size)`` (see
            :mod:`scisynth.observer.links`), or None to link by ids.
        """
        return None

    def __call__(self, obs: Observation, rng: np.random.Generator) -> Observation:
        return self.call_with_params(obs, rng)[0]

    def call_with_params(
        self, obs: Observation, rng: np.random.Generator
    ) -> tuple[Observation, dict[str, Any]]:
        """Call the stage and also return the parameters it resolved.

        Parameters
        ----------
        obs : Observation
            The observation to transform.
        rng : numpy.random.Generator
            Source of randomness for this call.

        Returns
        -------
        out : Observation
            The same as ``stage(obs, rng)``.
        params : dict
            The resolved parameters, as passed to :meth:`apply` and :meth:`links_from`.
        """
        params = self.resolved_params(rng)
        out = self.apply(obs, rng, params)
        return out.replace(meta=[*obs.meta, self.log_entry(params)]), params

    def run(self, obs: Observation, seed: SeedLike = None) -> Observation:
        """Apply the stage from a seed.

        Parameters
        ----------
        obs : Observation
            The observation to transform.
        seed : int, SeedSequence or None, default=None
            Seed for the random stream; None draws fresh OS entropy.

        Returns
        -------
        Observation
            The transformed observation.

        Notes
        -----
        Uses stream 0 of ``seed``. Inside an
        :class:`~scisynth.observer.observer.Observer` a stage gets stream
        ``i + 1``, so this does not reproduce the stage's behavior in a pipeline.
        """
        return self(obs, spawn_rngs(seed, 1)[0])

    def demo_observation(self) -> Observation:
        """Return the observation :meth:`preview` uses when none is given.

        Override in stages that need a particular input. The default is a smooth,
        signed 2-D grid.

        Returns
        -------
        Observation
            Deterministic, and a 2-D grid.
        """
        from .demo import smooth_grid

        return smooth_grid()

    def preview(
        self,
        obs: Observation | None = None,
        seed: SeedLike = 0,
        *,
        kind: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot this stage's effect: before and after, side by side.

        Parameters
        ----------
        obs : Observation, optional
            Defaults to :meth:`demo_observation`.
        seed : int, SeedSequence or None, default=0
            Makes the preview repeatable.
        kind : str, optional
            How to draw it: ``"scatter"`` (the default) or ``"heatmap"``. Points are
            averaged into bins for a heatmap.
        **kwargs
            Passed to the plot function (for plotly, ``channel`` and
            ``interactive``, then ``fig.update_layout``).

        Returns
        -------
        plotly.graph_objects.Figure
            Two panels: before and after.
        """
        from ..viz import plot

        if obs is None:
            obs = self.demo_observation()
        return plot(self, obs, seed=seed, kind=kind, **kwargs)
