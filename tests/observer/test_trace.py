from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scisynth import (
    AnalyticField,
    CoordinateShift,
    Gain,
    GaussianNoise,
    GridSampler,
    Observation,
    Observer,
    PointSampler,
    RandomDropout,
    StageDiff,
    Trace,
)


def observer() -> Observer:
    return Observer(
        GridSampler((8, 8)),
        [
            ("noise", GaussianNoise(0.1)),
            ("drop", RandomDropout(0.3)),
            ("gain", Gain(2.0)),
        ],
    )


def test_observer_trace_is_a_trace_of_every_step(field: AnalyticField) -> None:
    trace = observer().trace(field, 1)
    assert isinstance(trace, Trace) and len(trace) == 4
    assert trace.names == ["sampler", "noise", "drop", "gain"]
    assert trace[0].mask.all() and trace[-1] is trace[3]
    assert "sampler" in repr(trace)


def test_a_trace_refuses_repeated_step_names(field: AnalyticField) -> None:
    obs = GridSampler(4).run(field, 0)
    with pytest.raises(ValueError, match=r"unique.*GridSampler"):
        Trace([obs, obs])
    renamed = obs.replace(meta=[{**obs.meta[-1], "name": "again"}])
    assert Trace([obs, renamed]).names == ["GridSampler", "again"]


def test_it_iterates_and_slices_like_a_sequence(field: AnalyticField) -> None:
    trace = observer().trace(field, 1)
    assert [o.meta[-1]["name"] for o in trace] == trace.names
    sub = trace[1:3]
    assert isinstance(sub, Trace) and sub.names == ["noise", "drop"]
    assert trace[::2].names == ["sampler", "drop"]


def test_empty_slices_give_empty_traces(field: AnalyticField) -> None:
    trace = observer().trace(field, 1)
    for key in (slice(0, 0), slice(2, 2), slice(3, 1)):
        assert len(trace[key]) == 0
    assert len(trace[1:2].diffs) == 0


def test_steps_can_be_picked_by_name_but_slices_are_positional(
    field: AnalyticField,
) -> None:
    trace = observer().trace(field, 1)
    assert trace["drop"] is trace[2] and trace["sampler"] is trace[0]
    with pytest.raises(KeyError, match="No step named 'nope'"):
        trace["nope"]
    with pytest.raises(TypeError, match="Slice bounds must be integers"):
        trace[slice("noise", "drop")]


def test_diffs_describe_dropout_and_leave_value_stages_unchanged(
    field: AnalyticField,
) -> None:
    trace = observer().trace(field, 1)
    assert len(trace.diffs) == 3 and all(isinstance(d, StageDiff) for d in trace.diffs)
    noise, drop, gain = trace.diffs
    assert [d.name for d in trace.diffs] == ["noise", "drop", "gain"]
    assert not noise.changed and noise.n_kept == 64 and noise.n_dropped == 0
    assert drop.changed and drop.n_dropped > 0
    assert drop.n_kept + drop.n_dropped == 64 and drop.n_added == drop.n_moved == 0
    assert np.array_equal(drop.kept, trace["drop"].mask)
    assert np.array_equal(drop.dropped, ~trace["drop"].mask)
    assert not gain.changed  # masked locations stay masked
    assert gain.n_kept == drop.n_kept


def test_a_coordinate_shift_moves_every_kept_point(field: AnalyticField) -> None:
    for sampler in (GridSampler((5, 4)), PointSampler(20)):
        o = Observer(sampler, [RandomDropout(0.3), CoordinateShift(0.5)])
        drop, shift = o.trace(field, 2).diffs
        assert shift.changed and shift.n_dropped == 0
        assert np.array_equal(shift.moved, shift.kept)
        assert shift.n_moved == drop.n_kept
        assert drop.n_moved == 0


def test_only_the_changed_axis_counts_as_a_move(field: AnalyticField) -> None:
    o = Observer(
        PointSampler(10), [CoordinateShift([0.0, 0.0]), CoordinateShift([0.1, 0.0])]
    )
    none, some = o.trace(field, 0).diffs
    assert none.n_moved == 0 and some.n_moved == 10


def test_masked_locations_that_come_back_are_added() -> None:
    coords: dict[str, NDArray[Any]] = {"x": np.arange(4.0)}
    before = Observation(np.ones((4, 1)), coords, np.array([True, False, False, True]))
    after = Observation(np.ones((4, 1)), coords, np.array([True, True, False, False]))
    diff = StageDiff("revive", before, after)
    assert diff.kept.tolist() == [True, False, False, False]
    assert diff.dropped.tolist() == [False, False, False, True]
    assert diff.added.tolist() == [False, True, False, False]
    assert (diff.n_kept, diff.n_dropped, diff.n_added) == (1, 1, 1)


def obs_at(ids: list[int], x: list[float], mask: list[bool]) -> Observation:
    return Observation(
        np.ones((len(ids), 1)),
        {"x": np.array(x)},
        np.array(mask),
        ids=np.array(ids),
    )


def test_locations_are_matched_by_id_when_the_number_of_locations_changes() -> None:
    before = obs_at([0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0], [True] * 4)
    after = obs_at([2, 3, 10], [2.0, 3.0, 9.0], [True] * 3)  # 0, 1 removed; 10 new
    diff = StageDiff("crop", before, after)
    assert diff.kept.tolist() == [False, False, True, True]
    assert diff.dropped.tolist() == [True, True, False, False]
    assert diff.added.tolist() == [False, False, True]
    assert (diff.n_kept, diff.n_dropped, diff.n_added, diff.n_moved) == (2, 2, 1, 0)
    assert diff.changed


def test_matching_by_id_ignores_the_order_of_locations() -> None:
    before = obs_at([0, 1, 2], [0.0, 1.0, 2.0], [True] * 3)
    after = obs_at([2, 0, 1], [2.0, 0.5, 1.0], [True] * 3)  # reordered; id 0 moved
    diff = StageDiff("shuffle", before, after)
    assert diff.kept.all() and not diff.dropped.any() and not diff.added.any()
    assert diff.moved.tolist() == [True, False, False]


def test_a_stage_that_changes_the_locations_must_pass_ids() -> None:
    obs = obs_at([0, 1, 2], [0.0, 1.0, 2.0], [True] * 3)
    with pytest.raises(ValueError, match="must also pass ids"):
        obs.replace(
            values=np.ones((2, 1)),
            coords={"x": np.array([0.0, 1.0])},
            mask=np.ones(2, bool),
        )
    cropped = obs.replace(
        values=np.ones((2, 1)),
        coords={"x": np.array([0.0, 1.0])},
        mask=np.ones(2, bool),
        ids=obs.ids[:2],
    )
    assert StageDiff("crop", obs, cropped).n_dropped == 1


def test_an_empty_observation_diffs_without_errors() -> None:
    full = obs_at([0, 1], [0.0, 1.0], [True, True])
    empty = obs_at([], [], [])
    assert StageDiff("clear", full, empty).n_dropped == 2
    assert StageDiff("fill", empty, full).n_added == 2
    assert not StageDiff("noop", empty, empty).changed


def test_a_trace_without_step_names_falls_back_to_positions() -> None:
    obs = Observation(np.ones((2, 1)), {"x": np.arange(2.0)}, np.ones(2, bool))
    assert Trace([obs, obs]).names == ["0", "1"]
    assert Trace([]).diffs == ()


def test_trace_plots_one_panel_per_step(field: AnalyticField) -> None:
    pytest.importorskip("plotly")
    from scisynth.viz import plot, plot_observations

    trace = observer().trace(field, 1)
    fig = trace.plot(kind="scatter")
    assert len(fig.data) == 4
    assert [a.text for a in fig.layout.annotations] == trace.names
    assert len(plot(trace, kind="scatter", ncols=2).data) == 4
    assert len(plot_observations(trace[1:], trace.names[1:]).data) == 3  # also accepted
