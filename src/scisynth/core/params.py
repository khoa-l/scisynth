"""Parameter resolution: numbers, arrays, callables and distributions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike


@runtime_checkable
class HasRvs(Protocol):
    """Anything with an ``rvs`` method, e.g. a frozen ``scipy.stats`` distribution."""

    def rvs(self, *args: Any, **kwargs: Any) -> Any: ...


Param = ArrayLike | Callable[[np.random.Generator], Any] | HasRvs


def resolve(param: Any, rng: np.random.Generator) -> Any:
    """Turn a parameter specification into a concrete value.

    Parameters
    ----------
    param : object
        An object with an ``rvs`` method (such as a frozen ``scipy.stats``
        distribution) is drawn once with ``param.rvs(random_state=rng)``. Any other
        callable is called as ``param(rng)``. Everything else (numbers, arrays,
        strings, tuples, None) is returned unchanged.
    rng : numpy.random.Generator
        Source of randomness for distributions and callables.

    Returns
    -------
    object
        The concrete value.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from scisynth.core import resolve
    >>> rng = np.random.default_rng(0)
    >>> resolve(0.5, rng)
    0.5
    >>> round(float(resolve(stats.uniform(0, 1), rng)), 3)  # one draw
    0.637
    """
    if isinstance(param, HasRvs):
        return param.rvs(random_state=rng)
    if callable(param):
        return param(rng)
    return param
