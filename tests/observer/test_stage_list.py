import pytest

from scisynth import (
    AnalyticField,
    GaussianNoise,
    GridSampler,
    Observer,
    PoissonNoise,
    Quantize,
    RandomDropout,
    Stage,
    StageList,
)


def named() -> StageList:
    return StageList(
        [
            ("noise", GaussianNoise(0.1)),
            ("drop", RandomDropout(0.2)),
            ("quant", Quantize()),
        ]
    )


def test_it_is_a_sequence_of_stages() -> None:
    stages = named()
    assert len(stages) == 3 and all(isinstance(s, Stage) for s in stages)
    assert stages.names == ["noise", "drop", "quant"]
    assert stages.items()[1] == ("drop", stages[1])
    assert stages[0] in stages
    assert list(reversed(stages)) == [stages[2], stages[1], stages[0]]


def test_autonaming_matches_the_class_in_snake_case() -> None:
    stages = StageList([GaussianNoise(), RandomDropout(), Quantize()])
    assert stages.names == ["gaussian_noise", "random_dropout", "quantize"]
    twice = StageList([GaussianNoise(), GaussianNoise(), Quantize()])
    assert twice.names == ["gaussian_noise_1", "gaussian_noise_2", "quantize"]
    assert StageList([PoissonNoise()]).names == ["poisson_noise"]


def test_autonames_avoid_explicit_names() -> None:
    stages = StageList([("gaussian_noise", Quantize()), GaussianNoise()])
    assert stages.names == ["gaussian_noise", "gaussian_noise_1"]
    stages = StageList(
        [("gaussian_noise_1", Quantize()), GaussianNoise(), GaussianNoise()]
    )
    assert stages.names == ["gaussian_noise_1", "gaussian_noise_2", "gaussian_noise_3"]


def test_reserved_names_are_refused_and_avoided() -> None:
    with pytest.raises(ValueError, match="repeated"):
        StageList([("sampler", Quantize())], reserved=["sampler"])
    assert StageList([Quantize()], reserved=["quantize"]).names == ["quantize_1"]


def test_validation() -> None:
    with pytest.raises(TypeError, match="Stage"):
        StageList([GridSampler(4)])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Stage"):
        StageList([("a", GridSampler(4))])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="repeated"):
        StageList([("a", Quantize()), ("a", Quantize())])
    for reserved in ("a__b", "__x", "x__", "__"):
        with pytest.raises(ValueError, match="reserved"):
            StageList([(reserved, Quantize())])
    for empty in ("", 3):
        with pytest.raises(ValueError, match="non-empty"):
            StageList([(empty, Quantize())])  # type: ignore[list-item]
    assert StageList([("a_b_c", Quantize())]).names == ["a_b_c"]


def test_index_by_position_and_name() -> None:
    stages = named()
    assert stages[0] is stages["noise"] and isinstance(stages[0], Stage)
    assert stages[2] is stages["quant"] is stages[-1]
    assert stages[-3] is stages["noise"]
    with pytest.raises(KeyError, match="No stage named 'nope'"):
        stages["nope"]
    with pytest.raises(IndexError, match=r"3 .*out of range for 3 stage"):
        stages[3]
    with pytest.raises(IndexError, match="out of range"):
        stages[-4]
    with pytest.raises(IndexError, match="0 stage"):
        StageList()[0]
    with pytest.raises(TypeError, match="indices, names and slices"):
        stages[1.5]  # type: ignore[call-overload]


def test_slices_return_stage_lists_with_names_kept() -> None:
    stages = named()
    assert isinstance(stages[:2], StageList) and stages[:2].names == ["noise", "drop"]
    assert stages[1:].names == ["drop", "quant"]
    assert stages[::2].names == ["noise", "quant"]
    assert stages[-2:].names == ["drop", "quant"]
    assert stages[:0].names == []
    assert stages[:2] == StageList(stages.items()[:2])


def test_slice_bounds_must_be_positions_not_names() -> None:
    stages = named()
    for key in (slice("noise", "drop"), slice("drop", None), slice(None, "drop")):
        with pytest.raises(TypeError, match="Slice bounds must be integers"):
            stages[key]


def test_a_slice_of_autonamed_stages_keeps_the_autonames() -> None:
    stages = StageList([GaussianNoise(), GaussianNoise(), Quantize()])
    assert stages[1:].names == ["gaussian_noise_2", "quantize"]


def test_equality_depends_on_names_and_stages() -> None:
    assert named() == named()
    assert named() != named()[:2]
    assert named() != [s for s in named()]
    assert "('drop', RandomDropout(" in repr(named())


def test_observer_holds_a_stage_list_and_accepts_one() -> None:
    o = Observer(GridSampler(4), named())
    assert isinstance(o.stages, StageList) and o.stage_names == named().names
    assert o.names == ["sampler", "noise", "drop", "quant"]
    rebuilt = Observer(GridSampler(4), o.stages[1:])
    assert rebuilt.stage_names == ["drop", "quant"]  # names survive a slice


def test_extending_with_items_keeps_existing_names() -> None:
    o = Observer(GridSampler(4), named())
    longer = o.with_stages([*o.stages.items(), Quantize(4)])
    assert longer.stage_names == ["noise", "drop", "quant", "quantize"]
    assert o.stage_names == ["noise", "drop", "quant"]  # the original is unchanged


def test_stage_position_matches_the_trace_offset(field: AnalyticField) -> None:
    o = Observer(GridSampler(6), named())
    trace = o.trace(field, 3)
    for i, (name, stage) in enumerate(o.stages.items()):
        assert trace[i + 1].meta[-1]["name"] == name
        assert type(stage).__name__ == trace[i + 1].meta[-1]["type"]
