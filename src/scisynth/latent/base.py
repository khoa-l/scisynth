"""Latent protocol, kinds, and the generator/realization split."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from ..core._types import FloatArray
from ..core.component import Component
from ..core.domain import Domain
from ..core.rng import SeedLike

if TYPE_CHECKING:
    from .compose import Product, Sum

Kind = Literal["field", "gridded", "catalog", "label", "dataset"]


@runtime_checkable
class Latent(Protocol):
    """A realized ground truth.

    Attributes
    ----------
    kind : {"field", "gridded", "catalog", "label", "dataset"}
        What kind of latent this is, so samplers know what they can ask of it.
    domain : Domain
        The space the latent lives in.
    """

    @property
    def kind(self) -> Kind: ...

    @property
    def domain(self) -> Domain: ...


@runtime_checkable
class FieldLatent(Latent, Protocol):
    """A continuous field that can be queried anywhere in its domain.

    Channels are part of the shape of ``evaluate``'s answer, not of ``kind``: a
    multi-channel field is still a ``"field"``.
    """

    def evaluate(self, points: FloatArray) -> FloatArray:
        """Evaluate the field.

        Parameters
        ----------
        points : ndarray of shape (n, ndim)
            Query points, one row per point.

        Returns
        -------
        ndarray of shape (n, d)
            ``d`` is the number of channels (1 for a scalar field).
        """
        ...


class LatentGenerator(Component, register=False):
    """Configuration for a random latent: ``realize(seed)`` draws one instance."""

    def realize(self, seed: SeedLike = None) -> Latent:
        """Draw one latent.

        Parameters
        ----------
        seed : int, SeedSequence or None, default=None
            Seed for the random streams; None draws fresh OS entropy.

        Returns
        -------
        Latent
            A realized latent.
        """
        raise NotImplementedError

    def plot(
        self,
        seed: SeedLike = 0,
        *,
        kind: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Realize with ``seed`` and plot the result.

        Parameters
        ----------
        seed : int, SeedSequence or None, default=0
            Seed for the realization; the default keeps plots repeatable.
        kind : str, optional
            How to draw it, one of the kinds the object declares; by default the
            first that has a plot (see :func:`scisynth.viz.kinds_of`).
        **kwargs
            Passed to the plot function.

        Returns
        -------
        plotly.graph_objects.Figure
            The figure of the realized latent.
        """
        from ..viz import plot_latent

        return plot_latent(self.realize(seed), kind=kind, **kwargs)


class LatentPlotMixin:
    """Add a ``plot()`` method to realized latents.

    The drawing is in :mod:`scisynth.viz`. This is not part of the ``Latent``
    protocol, so custom latents need not provide it.
    """

    def plot(
        self,
        *,
        kind: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from ..viz import plot_latent

        return plot_latent(
            self,  # type: ignore[arg-type]
            kind=kind,
            **kwargs,
        )


def _is_latent_like(other: object) -> bool:
    """Return whether ``other`` could be a latent or a generator of one.

    Anything else, such as a number, makes the operator return ``NotImplemented``,
    so Python raises its usual ``TypeError``. A latent of the wrong kind, or an
    unrealized generator, is still passed on, so its error can say what is wrong.
    """
    return hasattr(other, "kind") or hasattr(other, "realize")


class FieldOps(LatentPlotMixin):
    """Add the operators ``a + b`` and ``a * b`` to field latents."""

    def __add__(self, other: FieldLatent) -> Sum:
        from .compose import Sum

        if not _is_latent_like(other):
            return NotImplemented
        return Sum(self, other)  # type: ignore[arg-type]

    def __mul__(self, other: FieldLatent) -> Product:
        from .compose import Product

        if not _is_latent_like(other):
            return NotImplemented
        return Product(self, other)  # type: ignore[arg-type]
