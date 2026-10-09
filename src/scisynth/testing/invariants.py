"""Reusable invariant checks for observations and pipelines."""

from __future__ import annotations

import numpy as np

from ..core._coords import has_position
from ..core.observation import Observation
from ..core.rng import SeedLike
from ..latent.base import Latent
from ..observer.observer import Observer
from ..stages.base import Stage


class InvariantError(AssertionError):
    """An invariant of an Observation or pipeline was violated."""


def check_observation(obs: Observation) -> None:
    """Check that an observation is well formed.

    ``values`` is ``(*spatial, d)``; ``mask`` must have the spatial shape and
    ``truth`` the shape of ``values``. ``ids`` must be unique integers with the
    spatial shape. Coordinates must match a grid or points layout, masked-out
    values must be NaN, valid locations must have finite coordinates, and the
    ``meta`` records must be well formed.

    Parameters
    ----------
    obs : Observation
        The observation to check.

    Raises
    ------
    InvariantError
        If any of these fails.
    """
    _check_arrays(obs)
    _check_ids(obs)
    _check_coords(obs)
    if not np.all(np.isnan(obs.values[~obs.mask])):
        raise InvariantError("masked-out values must be NaN")
    for entry in obs.meta:
        if set(entry) != {"name", "type", "params"}:
            raise InvariantError(f"malformed meta entry: {entry!r}")


def _check_arrays(obs: Observation) -> None:
    """Check that ``values``, ``mask`` and ``truth`` have matching shapes and types."""
    shape = obs.values.shape
    spatial = shape[:-1]
    if obs.mask.shape != spatial:
        raise InvariantError(
            f"mask shape {obs.mask.shape} != spatial shape {spatial} of values {shape}"
        )
    if obs.mask.dtype != np.bool_:
        raise InvariantError(f"mask dtype must be bool, got {obs.mask.dtype}")
    if obs.truth is not None and obs.truth.shape != shape:
        raise InvariantError(f"truth shape {obs.truth.shape} != values shape {shape}")


def _check_ids(obs: Observation) -> None:
    """Check that the ids are unique integers with the spatial shape."""
    spatial = obs.values.shape[:-1]
    if obs.ids.shape != spatial or not np.issubdtype(obs.ids.dtype, np.integer):
        raise InvariantError(f"ids must be integers of shape {spatial}")
    if np.unique(obs.ids).size != obs.ids.size:
        raise InvariantError("ids must be unique")


def _check_coords(obs: Observation) -> None:
    """Check that the coordinates fit a layout and valid locations have a position."""
    spatial = obs.values.shape[:-1]
    coords = list(obs.coords.values())
    if obs.layout == "points" and any(c.shape != spatial for c in coords):
        raise InvariantError(
            f"coords {[c.shape for c in coords]} match neither a grid nor points "
            f"layout for values of shape {obs.values.shape}"
        )
    if np.any(obs.mask & ~has_position(obs)):
        raise InvariantError("valid locations must have finite coordinates")


def observations_equal(a: Observation, b: Observation) -> bool:
    """Return whether two observations are exactly equal.

    Compares values, mask, ids, coordinates, truth and meta, treating NaN as equal to
    NaN.

    Parameters
    ----------
    a, b : Observation
        The observations to compare.

    Returns
    -------
    bool
        True if they are exactly equal.
    """
    if not (
        np.array_equal(a.values, b.values, equal_nan=True)
        and np.array_equal(a.mask, b.mask)
        and np.array_equal(a.ids, b.ids)
        and a.coords.keys() == b.coords.keys()
        and all(np.array_equal(a.coords[k], b.coords[k]) for k in a.coords)
        and a.meta == b.meta
    ):
        return False
    if a.truth is None or b.truth is None:
        return a.truth is b.truth
    return np.array_equal(a.truth, b.truth, equal_nan=True)


def assert_reproducible(observer: Observer, latent: Latent, seed: SeedLike) -> None:
    """Check that two runs with the same seed give identical observations.

    Parameters
    ----------
    observer : Observer
        The pipeline to run.
    latent : Latent
        The ground truth to observe.
    seed : int, SeedSequence or None
        Seed for both runs.

    Raises
    ------
    InvariantError
        If the two runs differ.
    """
    if not observations_equal(observer.run(latent, seed), observer.run(latent, seed)):
        raise InvariantError("same seed gave different observations")


def assert_prefix_stable(observer: Observer, latent: Latent, seed: SeedLike) -> None:
    """Check that removing trailing stages leaves earlier output unchanged.

    Parameters
    ----------
    observer : Observer
        The pipeline to run.
    latent : Latent
        The ground truth to observe.
    seed : int, SeedSequence or None
        Seed for all runs.

    Raises
    ------
    InvariantError
        If the observer built from the first ``k`` stages differs from the full run
        at step ``k``.
    """
    full = observer.trace(latent, seed)
    for k in range(len(observer.stages) + 1):
        head = observer.with_stages(observer.stages[:k])
        if not observations_equal(head.run(latent, seed), full[k]):
            raise InvariantError(
                f"the first {k} stage(s) differ from the full run at step {k}"
            )


def assert_append_stable(
    observer: Observer, extra: Stage, latent: Latent, seed: SeedLike
) -> None:
    """Check that appending a stage leaves every existing step's output unchanged.

    Parameters
    ----------
    observer : Observer
        The pipeline to run.
    extra : Stage
        The stage to append.
    latent : Latent
        The ground truth to observe.
    seed : int, SeedSequence or None
        Seed for both runs.

    Raises
    ------
    InvariantError
        If any step of the run with ``extra`` differs from the run without it.
    """
    before = observer.trace(latent, seed)
    taken = set(observer.names)
    n = 1
    while f"appended_{n}" in taken:
        n += 1
    after = observer.with_stages(
        [*observer.stages.items(), (f"appended_{n}", extra)]
    ).trace(latent, seed)
    for k, obs in enumerate(before):
        if not observations_equal(obs, after[k]):
            raise InvariantError(f"appending {type(extra).__name__} changed step {k}")
