from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scisynth import spawn_rngs


def draws(gen: np.random.Generator) -> NDArray[Any]:
    return gen.random(4)


def test_same_seed_same_streams() -> None:
    a, b = spawn_rngs(7, 3), spawn_rngs(7, 3)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(draws(x), draws(y))


def test_streams_are_distinct() -> None:
    x, y = spawn_rngs(7, 2)
    assert not np.array_equal(draws(x), draws(y))


def test_child_does_not_depend_on_count() -> None:
    small, large = spawn_rngs(7, 2), spawn_rngs(7, 5)
    for i in range(2):
        assert np.array_equal(draws(small[i]), draws(large[i]))


def test_seedsequence_input_is_not_consumed() -> None:
    ss = np.random.SeedSequence(3)
    assert np.array_equal(draws(spawn_rngs(ss, 1)[0]), draws(spawn_rngs(ss, 1)[0]))


def test_generator_seed_rejected() -> None:
    with pytest.raises(TypeError, match="Generator"):
        spawn_rngs(np.random.default_rng(0), 1)  # type: ignore[arg-type]
