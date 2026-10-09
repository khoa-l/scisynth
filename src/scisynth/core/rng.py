"""Independent random streams from a single seed."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SeedLike = int | Sequence[int] | np.random.SeedSequence | None


def _root(seed: SeedLike) -> np.random.SeedSequence:
    if isinstance(seed, np.random.Generator):
        raise TypeError(
            "Pass an int or SeedSequence as the seed, not a Generator: "
            "a shared Generator would couple otherwise independent streams."
        )
    if isinstance(seed, np.random.SeedSequence):
        # ``spawn`` is stateful; rebuild so repeated calls give identical children.
        return np.random.SeedSequence(
            seed.entropy, spawn_key=seed.spawn_key, pool_size=seed.pool_size
        )
    return np.random.SeedSequence(seed)


def spawn_rngs(seed: SeedLike, n: int) -> list[np.random.Generator]:
    """Spawn independent random generators from one seed.

    Parameters
    ----------
    seed : int, sequence of int, SeedSequence or None
        None draws fresh OS entropy. A ``Generator`` is rejected, since sharing one
        would couple otherwise independent streams. A ``SeedSequence`` is never
        consumed: repeated calls give identical children.
    n : int
        Number of generators; must be non-negative.

    Returns
    -------
    list of numpy.random.Generator
        Child ``i`` depends only on ``(seed, i)``, never on ``n``, so asking for
        more generators does not change the earlier ones.

    Raises
    ------
    TypeError
        If ``seed`` is a ``Generator``.
    ValueError
        If ``n`` is negative.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return [np.random.default_rng(child) for child in _root(seed).spawn(n)]
