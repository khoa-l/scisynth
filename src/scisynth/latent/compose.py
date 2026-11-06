"""Composite latents built from realized latents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from typing import ClassVar

import numpy as np

from ..core._types import FloatArray
from ..core._unimplemented import Unimplemented
from ..core.domain import Domain
from .base import FieldLatent, FieldOps, Kind, LatentPlotMixin


def _check_fields(owner: object, terms: tuple[FieldLatent, ...]) -> None:
    """Check that every term is a field latent, all with the same number of axes."""
    for t in terms:
        kind = getattr(t, "kind", None)
        if kind != "field":
            hint = " (realize generators first)" if hasattr(t, "realize") else ""
            raise TypeError(
                f"{type(owner).__name__} composes realized latents of kind 'field', "
                f"got {type(t).__name__} with kind {kind!r}{hint}"
            )
    if len({t.domain.ndim for t in terms}) != 1:
        raise ValueError("All terms must have the same number of axes")


class _FieldComposite(FieldOps):
    kind: ClassVar[Kind] = "field"

    def __init__(self, *terms: FieldLatent) -> None:
        if len(terms) < 2:
            raise ValueError(f"{type(self).__name__} needs at least two fields")
        _check_fields(self, terms)
        self.terms = tuple(terms)

    @property
    def domain(self) -> Domain:
        """Domain of the first term."""
        return self.terms[0].domain

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.terms))})"


class Sum(_FieldComposite):
    """Pointwise sum of fields.

    Parameters
    ----------
    *terms : FieldLatent
        At least two realized fields with the same number of axes. The domain is
        the first term's. A one-channel term broadcasts against ``d`` channels.

    Notes
    -----
    ``a + b`` on fields builds a ``Sum``.
    """

    def evaluate(self, points: FloatArray) -> FloatArray:
        out: FloatArray = reduce(np.add, (t.evaluate(points) for t in self.terms))
        return out


class Product(_FieldComposite):
    """Pointwise product of fields.

    Parameters
    ----------
    *terms : FieldLatent
        At least two realized fields with the same number of axes. The domain is
        the first term's. A one-channel term broadcasts against ``d`` channels.

    Notes
    -----
    ``a * b`` on fields builds a ``Product``.
    """

    def evaluate(self, points: FloatArray) -> FloatArray:
        out: FloatArray = reduce(np.multiply, (t.evaluate(points) for t in self.terms))
        return out


@dataclass
class Threshold(FieldOps):
    """1 where a field is above ``level``, 0 elsewhere.

    Parameters
    ----------
    field : FieldLatent
        A realized field. A multi-channel field is thresholded channel by channel.
    level : float, default=0.0
        The cut. Values equal to ``level`` give 0.

    Raises
    ------
    TypeError
        If ``field`` is not a realized field.

    Notes
    -----
    A :class:`~scisynth.latent.fields.GaussianField` with mean 0, cut at 0, covers
    about half of the domain; a higher ``level`` gives fewer, smaller blobs.

    Examples
    --------
    >>> import numpy as np
    >>> from scisynth.core import Domain
    >>> from scisynth.latent import AnalyticField, Threshold
    >>> domain = Domain.from_extents([(0, 10)])
    >>> mask = Threshold(AnalyticField(lambda x: x, domain), level=5.0)
    >>> mask.evaluate(np.array([[2.0], [8.0]]))
    array([[0.],
           [1.]])
    """

    field: FieldLatent
    level: float = 0.0
    kind: ClassVar[Kind] = "field"

    def __post_init__(self) -> None:
        _check_fields(self, (self.field,))

    @property
    def domain(self) -> Domain:
        """Domain of ``field``."""
        return self.field.domain

    def evaluate(self, points: FloatArray) -> FloatArray:
        """Evaluate at ``points``.

        Parameters
        ----------
        points : ndarray of shape (n, ndim)
            Query points, one row per point.

        Returns
        -------
        ndarray of shape (n, d)
            1.0 where ``field`` is above ``level``, 0.0 elsewhere, with the same
            channels as ``field``.
        """
        return (self.field.evaluate(points) > self.level).astype(np.float64)


class Warp(Unimplemented, LatentPlotMixin):
    """Evaluate a latent through a coordinate transform. Not implemented yet."""


class Multichannel(LatentPlotMixin):
    """Stack realized single-channel fields as the channels of one latent.

    Parameters
    ----------
    *channels : FieldLatent
        One or more fields with one channel each (nesting is not supported) and the
        same number of axes. The domain is the first channel's.
    corr : array-like of shape (d, d), optional
        Correlation matrix: symmetric, unit diagonal, positive definite. The
        channels are mixed with its Cholesky factor, so independent channels of
        equal variance come out with correlation ``corr``. Without it the channels
        are stacked unchanged.

    Attributes
    ----------
    n_channels : int
        Number of channels.
    corr : ndarray of shape (d, d) or None
        The correlation matrix, None if the channels are stacked unchanged.

    Raises
    ------
    ValueError
        If ``corr`` has the wrong shape, is not symmetric with a unit diagonal, or
        is not positive definite.

    Examples
    --------
    >>> import numpy as np
    >>> from scisynth.core import Domain
    >>> from scisynth.latent import GaussianField, Multichannel
    >>> domain = Domain.from_extents([(0, 1), (0, 1)])
    >>> fields = [GaussianField(domain).realize(seed) for seed in (1, 2)]
    >>> latent = Multichannel(*fields, corr=[[1, 0.8], [0.8, 1]])
    >>> points = np.random.default_rng(0).uniform(0, 1, size=(5, 2))
    >>> latent.evaluate(points).shape
    (5, 2)
    """

    kind: ClassVar[Kind] = "field"

    def __init__(
        self,
        *channels: FieldLatent,
        corr: Sequence[Sequence[float]] | FloatArray | None = None,
    ) -> None:
        if not channels:
            raise ValueError("Multichannel needs at least one channel")
        _check_fields(self, channels)
        self.channels = tuple(channels)
        self.corr: FloatArray | None = None
        self._mix: FloatArray | None = None

        if corr is not None:
            matrix = np.asarray(corr, dtype=np.float64)
            d = len(channels)
            if matrix.shape != (d, d):
                raise ValueError(f"corr must be {d}x{d}, got shape {matrix.shape}")
            if not (np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 1)):
                raise ValueError("corr must be symmetric with a unit diagonal")
            try:
                self._mix = np.asarray(np.linalg.cholesky(matrix), dtype=np.float64)
            except np.linalg.LinAlgError:
                raise ValueError("corr must be positive definite") from None
            self.corr = matrix

    @property
    def domain(self) -> Domain:
        """Domain of the first channel.

        Returns
        -------
        Domain
            The first channel's domain.
        """
        return self.channels[0].domain

    @property
    def n_channels(self) -> int:
        """Number of channels.

        Returns
        -------
        int
            How many fields are stacked.
        """
        return len(self.channels)

    def evaluate(self, points: FloatArray) -> FloatArray:
        """Evaluate every channel.

        Parameters
        ----------
        points : ndarray of shape (n, ndim)
            Query points, one row per point.

        Returns
        -------
        ndarray of shape (n, d)
            One column per channel, in the order given.
        """
        columns = [c.evaluate(points) for c in self.channels]
        if any(col.shape[-1] != 1 for col in columns):
            raise ValueError("Each Multichannel channel must have exactly one channel")
        stacked = np.concatenate(columns, axis=1)
        if self._mix is None:
            return stacked
        mixed: FloatArray = stacked @ self._mix.T
        return mixed

    def __repr__(self) -> str:
        return f"Multichannel({', '.join(map(repr, self.channels))})"
