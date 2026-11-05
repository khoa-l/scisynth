import numpy as np
import pytest

from scisynth.core._numbers import as_index, as_int


@pytest.mark.parametrize("value", [3, 3.0, np.int64(3), np.float64(3.0), np.array(3)])
def test_as_int_accepts_whole_numbers(value: object) -> None:
    result = as_int(value, "n")
    assert result == 3 and type(result) is int


@pytest.mark.parametrize(
    "value", [2.7, float("nan"), float("inf"), True, "3", None, [1, 2]]
)
def test_as_int_refuses_everything_else(value: object) -> None:
    with pytest.raises(ValueError, match="n must be an integer"):
        as_int(value, "n")


def test_as_index_counts_negative_numbers_from_the_end() -> None:
    assert [as_index(i, 3, "channel") for i in (0, 2, -1, -3)] == [0, 2, 2, 0]
    for bad in (3, -4):
        with pytest.raises(ValueError, match=rf"channel {bad} is out of range for x"):
            as_index(bad, 3, "channel", " for x")
    with pytest.raises(ValueError, match="channel must be an integer"):
        as_index(1.5, 3, "channel")
