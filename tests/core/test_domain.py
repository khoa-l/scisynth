import pytest

from scisynth import Axis, Domain
from scisynth.core.component import Component


def test_from_extents_defaults() -> None:
    d = Domain.from_extents([(0, 1), (0, 2)])
    assert d.names == ("x", "y")
    assert d.extents == ((0.0, 1.0), (0.0, 2.0))
    assert d.ndim == 2


def test_validation() -> None:
    with pytest.raises(ValueError, match="lo < hi"):
        Axis("x", (1, 0))
    with pytest.raises(ValueError, match="Duplicate"):
        Domain((Axis("x", (0, 1)), Axis("x", (0, 1))))


def test_round_trip() -> None:
    d = Domain.from_extents([(0, 1), (0, 2)], units=["m", "s"], value_unit="K")
    assert Component.from_dict(d.to_dict()) == d
