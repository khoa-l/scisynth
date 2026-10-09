import numpy as np
import pytest

from scisynth import AnalyticField, Domain, Observation
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


def test_integer_parameters_refuse_fractions() -> None:
    from scisynth import GridSampler, PointSampler, Quantize, RandomInsertion

    obs = Observation(np.zeros((4, 1)), {"x": np.arange(4.0)}, np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="levels must be an integer"):
        Quantize(levels=3.9).run(obs, 0)
    with pytest.raises(ValueError, match="n must be an integer"):
        RandomInsertion(n=2.5).run(obs, 0)
    with pytest.raises(ValueError, match="shape must be an integer"):
        GridSampler((4, 2.5))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n must be an integer"):
        PointSampler(2.7).run(
            AnalyticField(lambda x: x, Domain.from_extents([(0, 1)])), 0
        )


def test_as_index_counts_negative_numbers_from_the_end() -> None:
    assert [as_index(i, 3, "channel") for i in (0, 2, -1, -3)] == [0, 2, 2, 0]
    for bad in (3, -4):
        with pytest.raises(ValueError, match=rf"channel {bad} is out of range for x"):
            as_index(bad, 3, "channel", " for x")
    with pytest.raises(ValueError, match="channel must be an integer"):
        as_index(1.5, 3, "channel")


def test_shape_helpers_name_the_setting_in_their_errors() -> None:
    from scisynth.core._grid import as_shape, check_cells

    assert as_shape(4, 3) == (4, 4, 4) and as_shape([2, 3], 2) == (2, 3)
    with pytest.raises(ValueError, match="resolution must be an integer"):
        as_shape([4, 2.5], 2, "resolution")
    with pytest.raises(ValueError, match="Expected 2 positive sizes for resolution"):
        as_shape([4, 0], 2, "resolution")
    assert check_cells((3, 4), 100, "shape") == 12
    with pytest.raises(ValueError, match=r"resolution .* over the limit of 10"):
        check_cells((4, 4), 10, "resolution")
