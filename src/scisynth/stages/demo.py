"""Demo signals: small deterministic observations used for stage previews.

A stage chooses its own through ``Stage.demo_observation()``; the default is
:func:`smooth_grid`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core._types import FloatArray
from ..core.domain import Domain
from ..core.observation import Observation
from ..latent.fields import AnalyticField
from ..samplers.grid import GridSampler
from ..samplers.points import PointSampler

Shape = tuple[int, int]


def _unit_square() -> Domain:
    return Domain.from_extents([(0.0, 1.0), (0.0, 1.0)])


def _smooth(x: FloatArray, y: FloatArray) -> FloatArray:
    out: FloatArray = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    return out


def _sample_grid(
    func: Callable[[FloatArray, FloatArray], FloatArray], shape: Shape
) -> Observation:
    field = AnalyticField(func, _unit_square())
    return GridSampler(shape)(field, np.random.default_rng(0))


def smooth_grid(shape: Shape = (48, 48)) -> Observation:
    """Return ``sin(2 pi x) cos(2 pi y)`` on the unit square: smooth, in [-1, 1].

    Parameters
    ----------
    shape : (int, int), default=(48, 48)
        Number of samples along each axis.

    Returns
    -------
    Observation
        A fully valid 2-D grid.
    """
    return _sample_grid(_smooth, shape)


def bumps(shape: Shape = (48, 48)) -> Observation:
    """Return two Gaussian bumps (peaks 4 and 2) on a zero background.

    The values are non-negative, so they work as rates for counting noise.

    Parameters
    ----------
    shape : (int, int), default=(48, 48)
        Number of samples along each axis.

    Returns
    -------
    Observation
        A fully valid 2-D grid.
    """

    def f(x: FloatArray, y: FloatArray) -> FloatArray:
        a = 4 * np.exp(-((x - 0.3) ** 2 + (y - 0.35) ** 2) / 0.02)
        b = 2 * np.exp(-((x - 0.7) ** 2 + (y - 0.7) ** 2) / 0.04)
        out: FloatArray = a + b
        return out

    return _sample_grid(f, shape)


def ramp(shape: Shape = (48, 48)) -> Observation:
    """Return a diagonal gradient ``(x + y) / 2`` in [0, 1].

    Quantization shows up as bands.

    Parameters
    ----------
    shape : (int, int), default=(48, 48)
        Number of samples along each axis.

    Returns
    -------
    Observation
        A fully valid 2-D grid.
    """
    return _sample_grid(lambda x, y: (x + y) / 2, shape)


def scatter(n: int = 300) -> Observation:
    """Return ``sin(2 pi x) cos(2 pi y)`` at ``n`` random points of the unit square.

    Parameters
    ----------
    n : int, default=300
        Number of points.

    Returns
    -------
    Observation
        A fully valid points observation.
    """
    field = AnalyticField(_smooth, _unit_square())
    return PointSampler(n)(field, np.random.default_rng(0))
