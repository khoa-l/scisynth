import json

import pytest

from scisynth import (
    AnalyticField,
    GaussianNoise,
    GridSampler,
    Observer,
    Quantize,
    RandomDropout,
)


def named() -> Observer:
    return Observer(
        GridSampler((8, 8)),
        [
            ("noise", GaussianNoise(0.1)),
            ("drop", RandomDropout(0.2)),
            ("quant", Quantize(8)),
        ],
    )


def test_sampler_is_named_sampler_unless_given_a_name() -> None:
    bare = Observer(GridSampler(4), [GaussianNoise(), RandomDropout()])
    assert bare.names == ["sampler", "gaussian_noise", "random_dropout"]
    mixed = Observer(("grid", GridSampler(4)), [("a", Quantize()), GaussianNoise()])
    assert mixed.names == ["grid", "a", "gaussian_noise"]


def test_the_sampler_name_is_taken_by_the_stages() -> None:
    with pytest.raises(ValueError, match=r"unique.*'sampler'"):
        Observer(GridSampler(4), [("sampler", Quantize())])
    Observer(("grid", GridSampler(4)), [("sampler", Quantize())])  # free now
    named_like_a_stage = Observer(("quantize", GridSampler(4)), [Quantize()])
    assert named_like_a_stage.stage_names == ["quantize_1"]


def test_sampler_names_are_validated() -> None:
    with pytest.raises(ValueError, match="reserved"):
        Observer(("a__b", GridSampler(4)))
    with pytest.raises(TypeError, match="Sampler"):
        Observer(("a", Quantize()))  # type: ignore[arg-type]


def test_names_are_recorded_in_meta(field: AnalyticField) -> None:
    observer = named()
    expected = ["sampler", "noise", "drop", "quant"]
    assert [m["name"] for m in observer.run(field, 0).meta] == expected
    assert [t.meta[-1]["name"] for t in observer.trace(field, 0)] == expected


def test_names_survive_a_dict_round_trip_and_unnamed_dicts_are_named() -> None:
    for o in (
        named(),
        Observer(("grid", GridSampler(4)), [GaussianNoise(), GaussianNoise()]),
    ):
        restored = Observer.from_dict(json.loads(json.dumps(o.to_dict())))
        assert restored == o and restored.names == o.names
    data = {
        "sampler": GridSampler(4).to_dict(),
        "stages": [
            GaussianNoise().to_dict(),
            {"name": "q", "component": Quantize().to_dict()},
        ],
    }
    assert Observer.from_dict(data).names == ["sampler", "gaussian_noise", "q"]


def test_equality_and_repr_depend_on_names() -> None:
    a = Observer(GridSampler(4), [("x", Quantize())])
    assert a == Observer(GridSampler(4), [("x", Quantize())])
    assert a != Observer(GridSampler(4), [("y", Quantize())])
    assert a != Observer(("s", GridSampler(4)), [("x", Quantize())])
    assert a != "observer"
    assert "('noise', GaussianNoise(" in repr(named())
