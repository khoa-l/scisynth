"""Domain description: axes, extents, units, value kind."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from .component import Component

_DEFAULT_NAMES = ("x", "y", "z")


@dataclass
class Axis(Component):
    """One coordinate axis.

    Parameters
    ----------
    name : str
        Coordinate name, e.g. ``"x"``.
    extent : (float, float)
        ``(lo, hi)`` with ``lo < hi``.
    unit : str, default=""
        Unit label.
    """

    name: str
    extent: tuple[float, float]
    unit: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        lo, hi = self.extent
        self.extent = (float(lo), float(hi))
        if not self.extent[0] < self.extent[1]:
            raise ValueError(f"Axis {self.name!r}: need lo < hi, got {self.extent}")


@dataclass
class Domain(Component):
    """The space a latent lives in: its axes plus what its values mean.

    Parameters
    ----------
    axes : sequence of Axis
        Axes with unique names; stored as a tuple.
    value_unit : str, default=""
        Unit label of the measured values.
    value_kind : str, default="continuous"
        Free-form description of the values.

    Attributes
    ----------
    ndim : int
        Number of axes.
    names : tuple of str
        Axis names.
    extents : tuple of (float, float)
        Axis extents.

    See Also
    --------
    Domain.from_extents : Build a domain from extents alone.
    """

    axes: Sequence[Axis]
    value_unit: str = ""
    value_kind: str = "continuous"
    plot_subjects: ClassVar[tuple[str, ...]] = ("domain",)
    plot_kinds: ClassVar[tuple[str, ...]] = ("outline",)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.axes = tuple(self.axes)
        names = [a.name for a in self.axes]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate axis names: {names}")

    @classmethod
    def from_extents(
        cls,
        extents: Sequence[tuple[float, float]],
        names: Sequence[str] | None = None,
        units: Sequence[str] | None = None,
        value_unit: str = "",
        value_kind: str = "continuous",
    ) -> Domain:
        """Build a domain from extents, naming the axes automatically.

        Parameters
        ----------
        extents : sequence of (float, float)
            ``(lo, hi)`` per axis.
        names : sequence of str, optional
            Axis names; defaults to ``x, y, z`` for up to three axes, then
            ``x0, x1, ...``.
        units : sequence of str, optional
            Axis units; defaults to empty strings.
        value_unit : str, default=""
            Unit label of the measured values.
        value_kind : str, default="continuous"
            Free-form description of the values.

        Returns
        -------
        Domain
            The new domain.

        Examples
        --------
        >>> from scisynth.core import Domain
        >>> Domain.from_extents([(0, 10), (0, 5)], units=["km", "km"]).names
        ('x', 'y')
        """
        n = len(extents)
        if names is None:
            names = _DEFAULT_NAMES[:n] if n <= 3 else [f"x{i}" for i in range(n)]
        if units is None:
            units = [""] * n
        if not len(names) == len(units) == n:
            raise ValueError("extents, names and units must have the same length")
        axes = tuple(
            Axis(name, extent, unit)
            for name, extent, unit in zip(names, extents, units, strict=True)
        )
        return cls(axes, value_unit=value_unit, value_kind=value_kind)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.axes)

    @property
    def extents(self) -> tuple[tuple[float, float], ...]:
        return tuple(a.extent for a in self.axes)
