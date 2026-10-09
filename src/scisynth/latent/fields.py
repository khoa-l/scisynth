"""Continuous field latents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import ndimage

from ..core._grid import as_shape, check_cells
from ..core._numbers import as_int
from ..core._types import FloatArray
from ..core.domain import Domain
from ..core.params import Param
from ..core.rng import SeedLike, spawn_rngs
from .base import FieldOps, Kind, LatentGenerator

MAX_FIELD_CELLS = 50_000_000


@dataclass
class AnalyticField(FieldOps):
    """A deterministic field defined by a function of the coordinates.

    Parameters
    ----------
    func : callable
        Called as ``func(x, y)`` for a 2-D domain: one 1-D array per axis. Must
        return an array of the same length.
    domain : Domain
        The space the field lives in.

    Notes
    -----
    The field has one channel; stack several with
    :class:`~scisynth.latent.compose.Multichannel`. Because ``func`` is a callable,
    the field cannot be serialized.

    Examples
    --------
    >>> import numpy as np
    >>> from scisynth.core import Domain
    >>> from scisynth.latent import AnalyticField
    >>> domain = Domain.from_extents([(0, 10), (0, 10)])
    >>> trend = AnalyticField(lambda x, y: 0.5 * x + 0.2 * y, domain)
    >>> trend.evaluate(np.array([[1.0, 2.0]]))
    array([[0.9]])
    """

    func: Callable[..., Any]
    domain: Domain
    kind: ClassVar[Kind] = "field"

    def evaluate(self, points: FloatArray) -> FloatArray:
        """Evaluate at ``points``.

        Parameters
        ----------
        points : ndarray of shape (n, ndim)
            Query points, one row per point.

        Returns
        -------
        ndarray of shape (n, 1)
            The field values.

        Raises
        ------
        ValueError
            If ``func`` does not return shape ``(n,)``.
        """
        out = np.asarray(self.func(*points.T), dtype=np.float64)
        if out.shape != (points.shape[0],):
            raise ValueError(
                f"func must return shape ({points.shape[0]},), got {out.shape}"
            )
        return out[:, None]


@dataclass
class InterpolatedField(FieldOps):
    """A field stored on a regular grid and interpolated between nodes.

    Parameters
    ----------
    values : ndarray of shape (n_0, ..., n_{ndim-1})
        Field values at the nodes. Node ``i`` on an axis with extent ``(lo, hi)``
        and ``n`` nodes sits at ``lo + i * (hi - lo) / n``, so the last node is one
        spacing short of ``hi``.
    domain : Domain
        The space the field lives in.
    order : int, default=1
        Spline order, from 0 (nearest) and 1 (linear) up to 5.
    mode : str, default="nearest"
        A :func:`scipy.ndimage.map_coordinates` boundary mode. With ``"nearest"``,
        queries beyond the outermost nodes (including the last cell next to ``hi``)
        take the edge value; ``"grid-wrap"`` makes the field periodic.

    Raises
    ------
    ValueError
        If ``order`` is outside 0-5 or ``values`` does not have one dimension per
        axis.

    Notes
    -----
    The field has one channel.
    """

    values: FloatArray
    domain: Domain
    order: int = 1
    mode: str = "nearest"
    kind: ClassVar[Kind] = "field"

    def __post_init__(self) -> None:
        if not 0 <= self.order <= 5:
            raise ValueError(f"order must be between 0 and 5, got {self.order}")
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.ndim != self.domain.ndim:
            raise ValueError(
                f"values has {self.values.ndim} dims, domain has {self.domain.ndim}"
            )

    def evaluate(self, points: FloatArray) -> FloatArray:
        """Interpolate at ``points``.

        Parameters
        ----------
        points : ndarray of shape (n, ndim)
            Query points, one row per point.

        Returns
        -------
        ndarray of shape (n, 1)
            The interpolated values.
        """
        idx = np.stack(
            [
                (points[:, i] - lo) / ((hi - lo) / n)
                for i, ((lo, hi), n) in enumerate(
                    zip(self.domain.extents, self.values.shape, strict=True)
                )
            ]
        )
        out: FloatArray = ndimage.map_coordinates(
            self.values, idx, order=self.order, mode=self.mode
        )
        return out[:, None]


@dataclass
class GaussianField(LatentGenerator):
    """Stationary, periodic Gaussian random field with an RBF covariance.

    Parameters
    ----------
    domain : Domain
        The space the field lives in.
    length_scale : float or distribution, default=0.1
        The RBF length scale ``l``; correlation falls to 1/e at ``r = sqrt(2) * l``.
        Must be positive. A distribution gives one draw per realization.
    amplitude : float or distribution, default=1.0
        Standard deviation of the field.
    mean : float or distribution, default=0.0
        Mean of the field.
    resolution : int or sequence of int, default=128
        Grid nodes per axis (an int is repeated for every axis). The field holds no
        detail finer than this grid.
    order : int, default=1
        Interpolation spline order when querying: 1 (linear) shows kinks at the
        nodes, 3 (cubic) is smooth, 0 is nearest-neighbor.
    max_cells : int, default=50_000_000
        Upper limit on ``prod(resolution)``, checked before allocating.

    Notes
    -----
    A realization is white noise filtered in Fourier space by ``exp(-k^2 l^2 / 4)``,
    which gives the covariance ``exp(-r^2 / (2 l^2))``, the same kernel as
    scikit-learn's ``RBF(length_scale=l)``. The result is standardized to zero mean
    and unit variance, then scaled to ``mean + amplitude * field``. The FFT makes it
    periodic, so the realization wraps around the domain.

    The grid has ``resolution ** ndim`` cells, so higher-dimensional domains need a
    small ``resolution``.

    Examples
    --------
    >>> import numpy as np
    >>> from scisynth.core import Domain
    >>> from scisynth.latent import GaussianField
    >>> domain = Domain.from_extents([(0, 10), (0, 10)])
    >>> latent = GaussianField(domain, length_scale=1.5).realize(seed=1)
    >>> latent.evaluate(np.array([[5.0, 5.0]])).shape
    (1, 1)
    """

    domain: Domain
    length_scale: Param = 0.1
    amplitude: Param = 1.0
    mean: Param = 0.0
    resolution: int | Sequence[int] = 128
    order: int = 1
    max_cells: int = MAX_FIELD_CELLS

    def realize(self, seed: SeedLike = None) -> InterpolatedField:
        """Draw one realization.

        Parameters
        ----------
        seed : int, SeedSequence or None, default=None
            Seed for the random streams; None draws fresh OS entropy.

        Returns
        -------
        InterpolatedField
            One channel, periodic (``mode="grid-wrap"``).

        Raises
        ------
        ValueError
            If ``length_scale`` is not positive or the grid exceeds ``max_cells``.
        """
        param_rng, noise_rng = spawn_rngs(seed, 2)
        params = self.resolved_params(param_rng)
        shape = as_shape(params["resolution"], self.domain.ndim, "resolution")
        check_cells(shape, self.max_cells, "resolution")
        length = float(params["length_scale"])
        if length <= 0:
            raise ValueError(f"length_scale must be > 0, got {length}")

        freqs = [
            2 * np.pi * np.fft.fftfreq(n, d=(hi - lo) / n)
            for n, (lo, hi) in zip(shape, self.domain.extents, strict=True)
        ]
        k2 = np.sum([g**2 for g in np.meshgrid(*freqs, indexing="ij")], axis=0)
        white = noise_rng.standard_normal(shape)
        field = np.fft.ifftn(np.fft.fftn(white) * np.exp(-0.25 * length**2 * k2)).real
        field -= field.mean()
        std = field.std()
        if std > 0:
            field /= std
        return InterpolatedField(
            float(params["mean"]) + float(params["amplitude"]) * field,
            self.domain,
            order=as_int(params["order"], "order"),
            mode="grid-wrap",  # the FFT construction is periodic
        )
