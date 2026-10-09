from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from scisynth import resolve


def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_numbers_arrays_and_others_pass_through() -> None:
    arr = np.arange(3.0)
    assert resolve(2.5, rng()) == 2.5
    assert resolve(arr, rng()) is arr
    assert resolve((1, 2), rng()) == (1, 2)
    assert resolve(None, rng()) is None
    assert resolve("x", rng()) == "x"


def test_callable_receives_rng() -> None:
    assert resolve(lambda r: r.uniform(5, 6), rng()) == rng().uniform(5, 6)


def test_frozen_scipy_distribution_uses_given_rng() -> None:
    dist = stats.norm(loc=3.0, scale=0.5)
    assert resolve(dist, rng()) == dist.rvs(random_state=rng())
    assert resolve(dist, np.random.default_rng(1)) != resolve(dist, rng())


def test_duck_typed_rvs() -> None:
    class Custom:
        def rvs(self, *args: Any, random_state: Any = None, **kwargs: Any) -> float:
            return float(random_state.integers(100))

    assert resolve(Custom(), rng()) == float(rng().integers(100))


def global_state() -> NDArray[Any]:
    """Return the key array of numpy's legacy global generator."""
    state: Any = np.random.get_state()  # noqa: NPY002
    key: NDArray[Any] = state[1].copy()
    return key


def test_does_not_touch_global_state() -> None:
    np.random.seed(123)  # noqa: NPY002
    before = global_state()
    resolve(stats.norm(), rng())
    assert np.array_equal(before, global_state())
