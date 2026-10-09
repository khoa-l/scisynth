"""Base for immutable sequences whose items can also be picked by name."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, ClassVar, TypeVar

T = TypeVar("T")


class NamedSequence(Sequence[T]):
    """An immutable sequence with one unique name per item.

    Subclasses set the names with :meth:`_set_names`, which rejects repeated names,
    so names are unique in every subclass. They also set ``_kind`` (what an item
    is called in error messages) and write ``__getitem__`` on top of
    :meth:`_locate`.

    Attributes
    ----------
    names : list of str
        The item names, in order.
    """

    _kind: ClassVar[str] = "item"
    _names: list[str]

    def _set_names(self, names: Sequence[str]) -> None:
        """Store the item names, raising ``ValueError`` if any is repeated."""
        repeated = sorted(n for n, count in Counter(names).items() if count > 1)
        if repeated:
            raise ValueError(f"Names must be unique; repeated: {repeated}")
        self._names = list(names)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def _locate(self, key: Any) -> int | range:
        """Turn a position, name or slice into a non-negative index or a range.

        Slice bounds are positions; names are not accepted as bounds.
        """
        n = len(self)
        if isinstance(key, int):
            if not -n <= key < n:
                kind = self._kind
                raise IndexError(
                    f"{kind.capitalize()} index {key} out of range for {n} {kind}(s)"
                )
            return key % n
        if isinstance(key, str):
            try:
                return self._names.index(key)
            except ValueError:
                raise KeyError(
                    f"No {self._kind} named {key!r}. Names: {self.names}"
                ) from None
        if not isinstance(key, slice):
            raise TypeError(
                f"{type(self).__name__} supports indices, names and slices, "
                f"not {type(key).__name__}"
            )
        if any(isinstance(bound, str) for bound in (key.start, key.stop)):
            raise TypeError("Slice bounds must be integers; names are not supported")
        return range(n)[key]
