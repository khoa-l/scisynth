"""An ordered, named collection of stages."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from typing import overload

from ..stages.base import Stage
from ._names import check_name
from .named_sequence import NamedSequence

StageSpec = Stage | tuple[str, Stage]


def _snake(name: str) -> str:
    """Return ``name`` in snake_case, e.g. ``PoissonNoise`` -> ``poisson_noise``."""
    # a lowercase letter or digit followed by a capital
    _BEFORE_CAPITAL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

    # the last capital of an acronym, as in HTTP|Server
    _AFTER_ACRONYM = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")

    name = _BEFORE_CAPITAL.sub("_", name)
    name = _AFTER_ACRONYM.sub("_", name)
    return name.lower()


def _autoname(
    stages: Sequence[Stage], explicit: Sequence[str | None], taken: set[str]
) -> list[str]:
    """Return names for ``stages``: explicit ones as given, the rest from their class.

    A class that appears once among the bare stages keeps its snake_case name;
    otherwise (or if the name is taken) its stages are numbered ``_1``, ``_2``, ...
    """
    bases = [_snake(type(s).__name__) for s in stages]
    counts = Counter(b for b, n in zip(bases, explicit, strict=True) if n is None)
    used, counters = set(taken), Counter[str]()
    names: list[str] = []
    for base, name in zip(bases, explicit, strict=True):
        if name is None:
            if counts[base] == 1 and base not in used:
                name = base
            else:
                while f"{base}_{counters[base] + 1}" in used:
                    counters[base] += 1
                counters[base] += 1
                name = f"{base}_{counters[base]}"
        used.add(name)
        names.append(name)
    return names


class StageList(NamedSequence[Stage]):
    """An ordered sequence of stages, each with a unique name.

    This is what :attr:`Observer.stages <scisynth.observer.observer.Observer.stages>`
    holds. Iterating gives the stages; use :meth:`items` for ``(name, stage)``
    pairs.

    Parameters
    ----------
    stages : iterable of Stage or (str, Stage), default=()
        A bare stage is named after its class in snake_case (``GaussianNoise`` ->
        ``"gaussian_noise"``); if a class appears more than once, or the name is
        taken, they are numbered ``"gaussian_noise_1"``, ``"gaussian_noise_2"``, ...
        A ``StageList`` keeps its names.
    reserved : iterable of str, default=()
        Names the stages may not use, e.g. the sampler's.

    Attributes
    ----------
    names : list of str
        The stage names, in order.

    Raises
    ------
    TypeError
        If an element is not a Stage.
    ValueError
        If a name is empty, contains ``"__"`` (reserved) or is repeated.

    Notes
    -----
    ``stages[i]`` is the ``i``-th stage (negative counts from the end) and
    ``stages["name"]`` the stage with that name. A slice returns a ``StageList``
    with the names kept; slice bounds are positions, not names.

    Examples
    --------
    >>> from scisynth.observer import StageList
    >>> from scisynth.stages import GaussianNoise, RandomDropout
    >>> stages = StageList([GaussianNoise(0.1), ("drop", RandomDropout(0.1))])
    >>> stages.names
    ['gaussian_noise', 'drop']
    >>> stages[1:].names
    ['drop']
    """

    _kind = "stage"

    def __init__(
        self,
        stages: Iterable[StageSpec] = (),
        *,
        reserved: Iterable[str] = (),
    ) -> None:
        specs = stages.items() if isinstance(stages, StageList) else stages
        explicit: list[str | None] = []
        parsed: list[Stage] = []
        for spec in specs:
            name: str | None = None
            if isinstance(spec, tuple):
                name, spec = spec
                check_name(name)
            if not isinstance(spec, Stage):
                raise TypeError(
                    f"stages must be Stage instances, got {type(spec).__name__}"
                )
            explicit.append(name)
            parsed.append(spec)

        taken = [*reserved, *(n for n in explicit if n is not None)]
        duplicates = sorted(n for n, c in Counter(taken).items() if c > 1)
        if duplicates:
            raise ValueError(f"Names must be unique; repeated: {duplicates}")
        self._stages = tuple(parsed)
        self._set_names(_autoname(parsed, explicit, set(taken)))

    def items(self) -> list[tuple[str, Stage]]:
        """Return the ``(name, stage)`` pairs, in order.

        Returns
        -------
        list of (str, Stage)
            Can be passed back to :class:`StageList` or ``Observer``, which keep
            the names.
        """
        return list(zip(self._names, self._stages, strict=True))

    def __len__(self) -> int:
        return len(self._stages)

    def __iter__(self) -> Iterator[Stage]:
        return iter(self._stages)

    @overload
    def __getitem__(self, key: int | str) -> Stage: ...

    @overload
    def __getitem__(self, key: slice) -> StageList: ...

    def __getitem__(self, key: int | str | slice) -> Stage | StageList:
        index = self._locate(key)
        if isinstance(index, range):
            return StageList([(self._names[i], self._stages[i]) for i in index])
        return self._stages[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StageList):
            return NotImplemented
        return self._names == other._names and self._stages == other._stages

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"StageList({self.items()!r})"
