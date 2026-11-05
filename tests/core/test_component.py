import json
from dataclasses import dataclass

import pytest
from scipy import stats

from scisynth import (
    Domain,
    GaussianField,
    GaussianNoise,
    GridSampler,
    PointSampler,
    Sampler,
    Stage,
)
from scisynth.core import registry
from scisynth.core.component import Component


def test_base_classes_are_not_registered() -> None:
    from scisynth import LatentGenerator, Sampler

    for base in (Stage, Sampler, LatentGenerator):
        with pytest.raises(KeyError, match="Unknown component"):
            registry.get(base.__name__)
        with pytest.raises(KeyError, match="Unknown component"):
            base.from_dict({"type": base.__name__})


def test_list_parameters_become_tuples_for_every_component() -> None:
    from scisynth import Domain, GridSampler, Sampler

    @dataclass
    class ListSampler(Sampler):
        sizes: list[int]

    assert ListSampler([1, 2]).sizes == (1, 2)  # type: ignore[comparison-overlap]
    assert GridSampler([4, 5]).shape == (4, 5)
    quantize = Stage.from_dict({"type": "Quantize", "params": {"vmin": [0, 1]}})
    assert quantize.params()["vmin"] == (0, 1)
    axes = [{"type": "Axis", "params": {"name": "x", "extent": [0, 1]}}]
    domain = Domain.from_dict({"type": "Domain", "params": {"axes": axes}})
    assert domain.axes[0].extent == (0.0, 1.0)


def test_register_false_is_not_inherited() -> None:
    class Base(Component, register=False):
        pass

    class Child(Base):
        pass

    with pytest.raises(KeyError):
        registry.get("Base")
    assert registry.get("Child") is Child


def test_duplicate_name_from_other_class_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @dataclass
        class GaussianNoise(Stage):
            pass


def test_every_stage_round_trips_through_json(stage: Stage) -> None:
    assert Stage.from_dict(json.loads(json.dumps(stage.to_dict()))) == stage


@pytest.mark.parametrize(
    "component",
    [
        GridSampler((4, 5)),
        PointSampler(10),
        Domain.from_extents([(0, 1), (0, 2)], units=["m", "s"], value_unit="K"),
        GaussianField(Domain.from_extents([(0, 1)]), 0.2, resolution=16, order=3),
    ],
    ids=lambda component: type(component).__name__,
)
def test_other_components_round_trip_through_json(component: Component) -> None:
    assert Component.from_dict(json.loads(json.dumps(component.to_dict()))) == component


def test_from_dict_checks_base_class() -> None:
    with pytest.raises(TypeError):
        Sampler.from_dict(GaussianNoise().to_dict())


def test_distribution_params_are_not_serializable() -> None:
    with pytest.raises(TypeError, match="Cannot serialize"):
        GaussianNoise(sigma=stats.uniform(0.1, 0.2)).to_dict()
