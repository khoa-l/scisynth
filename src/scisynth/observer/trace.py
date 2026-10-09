"""The record of one observer run: every intermediate observation and what changed."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar, overload

import numpy as np
from scipy import sparse

from ..core._types import Array, BoolArray
from ..core.observation import Observation
from .links import links_by_id
from .named_sequence import NamedSequence


@dataclass(frozen=True)
class StageDiff:
    """What one stage did to the locations of an observation.

    Locations are linked through ``links`` (by default their ``ids``, see
    :mod:`scisynth.observer.links`), so the stage may keep, drop, move, merge or add
    locations, and may change the number of them. Locations that are masked out count
    as absent.

    Parameters
    ----------
    name : str
        The stage's name.
    before, after : Observation
        The observation going into and coming out of the stage.
    links : scipy.sparse.csr_matrix, optional
        Shape ``(after.size, before.size)``; see :mod:`scisynth.observer.links`.
        Defaults to matching by ``ids``.

    Attributes
    ----------
    parents : ndarray of int, shape of ``after``
        How many locations of ``before`` each location of ``after`` depends on.
    children : ndarray of int, shape of ``before``
        How many locations of ``after`` depend on each location of ``before``.
    kept : ndarray of bool, shape of ``before``
        Valid before, and feeding at least one valid location after.
    dropped : ndarray of bool, shape of ``before``
        Valid before, but feeding nothing after.
    moved : ndarray of bool, shape of ``before``
        Locations linked one-to-one to a location with different coordinates.
    added : ndarray of bool, shape of ``after``
        Valid after, but depending on nothing before.
    changed : bool
        Whether the stage dropped, added or moved any location.
    """

    name: str
    before: Observation
    after: Observation
    # Filled in ``__post_init__`` when not given.
    links: sparse.csr_matrix = field(default=None)

    def __post_init__(self) -> None:
        if self.links is None:
            object.__setattr__(self, "links", links_by_id(self.before, self.after))

    @cached_property
    def parents(self) -> Array:
        return np.asarray(self.links.getnnz(axis=1)).reshape(self.after.mask.shape)

    @cached_property
    def children(self) -> Array:
        return np.asarray(self.links.getnnz(axis=0)).reshape(self.before.mask.shape)

    @cached_property
    def kept(self) -> BoolArray:
        kept: BoolArray = self.before.mask & (self.children > 0)
        return kept

    @cached_property
    def dropped(self) -> BoolArray:
        dropped: BoolArray = self.before.mask & (self.children == 0)
        return dropped

    @cached_property
    def added(self) -> BoolArray:
        added: BoolArray = self.after.mask & (self.parents == 0)
        return added

    @cached_property
    def moved(self) -> BoolArray:
        rows, cols = self.links.nonzero()
        single = (self.parents.ravel()[rows] == 1) & (self.children.ravel()[cols] == 1)
        rows, cols = rows[single], cols[single]
        shifted = np.zeros(self.before.size, dtype=np.bool_)
        before, after = self.before.spatial_coords, self.after.spatial_coords
        for name in before.keys() & after.keys():
            shifted[cols] |= before[name].ravel()[cols] != after[name].ravel()[rows]
        return shifted.reshape(self.before.mask.shape)

    @property
    def n_kept(self) -> int:
        return int(self.kept.sum())

    @property
    def n_dropped(self) -> int:
        return int(self.dropped.sum())

    @property
    def n_added(self) -> int:
        return int(self.added.sum())

    @property
    def n_moved(self) -> int:
        return int(self.moved.sum())

    @property
    def changed(self) -> bool:
        return bool(self.n_dropped or self.n_added or self.n_moved)


class Trace(NamedSequence[Observation]):
    """The observations after the sampler and after each stage of one run.

    Returned by
    :meth:`Observer.trace <scisynth.observer.observer.Observer.trace>`. It is an
    immutable sequence of observations in run order, so index 0 is the sampler's
    output. Besides integer indexing, a step can be picked by name.

    Parameters
    ----------
    observations : iterable of Observation
        The observations in run order. Each one's last ``meta`` entry names its step.
    links : iterable of scipy.sparse.csr_matrix or None, optional
        One per stage (``len(observations) - 1``): how that stage's output was built
        from its input, or None to match by ids. See :mod:`scisynth.observer.links`.

    Attributes
    ----------
    names : list of str
        The step names, sampler first.
    diffs : tuple of StageDiff
        One per stage (``len(trace) - 1``), comparing each observation with the one
        before it. Computed on first use.

    Raises
    ------
    ValueError
        If the number of links is wrong, or two steps have the same name.

    Notes
    -----
    ``trace[i]`` is the ``i``-th step and ``trace["name"]`` the step with that name.
    A slice returns a ``Trace``; slice bounds are positions, not names. Each
    :class:`StageDiff` follows locations through the stage's links, by default their
    ``Observation.ids``. A slice made with a step other than 1 loses the links and
    falls back to ids.

    Examples
    --------
    >>> from scisynth.core import Domain
    >>> from scisynth.latent import AnalyticField
    >>> from scisynth.observer import Observer
    >>> from scisynth.samplers import GridSampler
    >>> from scisynth.stages import GaussianNoise, RandomDropout
    >>> domain = Domain.from_extents([(0, 1), (0, 1)])
    >>> latent = AnalyticField(lambda x, y: x + y, domain)
    >>> observer = Observer(GridSampler(32), [GaussianNoise(0.1), RandomDropout(0.1)])
    >>> trace = observer.trace(latent, seed=0)
    >>> trace.names
    ['sampler', 'gaussian_noise', 'random_dropout']
    >>> int(trace["random_dropout"].mask.sum())
    933
    >>> trace.diffs[-1].n_dropped
    91
    """

    plot_subjects: ClassVar[tuple[str, ...]] = ("trace",)
    plot_kinds: ClassVar[tuple[str, ...]] = ("layers", "scatter", "heatmap")
    _kind = "step"

    def __init__(
        self,
        observations: Iterable[Observation],
        links: Iterable[sparse.csr_matrix | None] | None = None,
    ) -> None:
        self._observations = tuple(observations)
        n_stages = max(len(self._observations) - 1, 0)
        self._links = (None,) * n_stages if links is None else tuple(links)
        if len(self._links) != n_stages:
            raise ValueError(
                f"expected {n_stages} links for {len(self._observations)} "
                f"observations, got {len(self._links)}"
            )
        self._set_names(
            [
                obs.meta[-1]["name"] if obs.meta else str(i)
                for i, obs in enumerate(self._observations)
            ]
        )

    @cached_property
    def diffs(self) -> tuple[StageDiff, ...]:
        return tuple(
            StageDiff(name, before, after, links)
            for name, before, after, links in zip(
                self._names[1:],
                self._observations[:-1],
                self._observations[1:],
                self._links,
                strict=True,
            )
        )

    def links_between(self, start: int = 0, stop: int = -1) -> sparse.csr_matrix:
        """Return how step ``stop`` depends on step ``start``.

        Parameters
        ----------
        start, stop : int, default=0 and -1
            Step indices, with ``start`` not after ``stop``; the defaults span the
            whole run.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape ``(trace[stop].size, trace[start].size)`` over flat locations: the
            product of the stages' links, so a nonzero ``[j, i]`` means location ``j``
            of step ``stop`` traces back to location ``i`` of step ``start``.

        Raises
        ------
        ValueError
            If ``start`` is after ``stop``.
        """
        first, last = range(len(self))[start], range(len(self))[stop]
        if first > last:
            raise ValueError(f"start ({start}) is after stop ({stop})")
        out = sparse.identity(self[first].size, format="csr")
        for diff in self.diffs[first:last]:
            out = diff.links @ out
        return out

    def __len__(self) -> int:
        return len(self._observations)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self._observations)

    @overload
    def __getitem__(self, key: int | str) -> Observation: ...

    @overload
    def __getitem__(self, key: slice) -> Trace: ...

    def __getitem__(self, key: int | str | slice) -> Observation | Trace:
        index = self._locate(key)
        if not isinstance(index, range):
            return self._observations[index]
        links = None
        if index.step == 1:
            links = self._links[index.start : index.start + max(len(index) - 1, 0)]
        return Trace([self._observations[i] for i in index], links)

    def __repr__(self) -> str:
        return f"Trace({self.names})"

    def plot(
        self,
        *,
        kind: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the run: 3-D layers of points by default, or one panel per step.

        Parameters
        ----------
        kind : str, optional
            How to draw it: ``"layers"`` (the default, the 3-D plot), or ``"scatter"``
            or ``"heatmap"`` for one panel per step.
        **kwargs
            Passed to the plot function: ``max_points``, ``axes`` and
            ``frame_tolerance`` for ``"layers"``, or ``channel``, ``ncols`` and
            ``interactive`` for the panels.

        Returns
        -------
        plotly.graph_objects.Figure
            One layer of points per step. With ``kind="scatter"`` or ``"heatmap"``,
            one panel per step instead, titled with the step names.
        """
        from ..viz import plot

        return plot(self, kind=kind, **kwargs)
