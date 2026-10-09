import numpy as np
import pytest

from scisynth import (
    AnalyticField,
    Downsample,
    GaussianNoise,
    GridSampler,
    Observation,
    Observer,
    PointSampler,
    RandomDropout,
    Trace,
)
from scisynth.observer.links import links_by_id


def obs(ids: list[int], mask: list[bool]) -> Observation:
    n = len(ids)
    return Observation(
        np.ones((n, 1)), {"x": np.arange(float(n))}, np.array(mask), ids=np.array(ids)
    )


def test_links_by_id_join_valid_locations_with_the_same_id() -> None:
    before = obs([0, 1, 2, 3], [True, True, False, True])
    after = obs([3, 1, 2, 9], [True, False, True, True])  # reordered, 9 new
    links = links_by_id(before, after)
    assert links.shape == (4, 4)
    assert sorted(zip(*links.nonzero(), strict=True)) == [(0, 3)]  # id 3 only


def test_links_by_id_of_empty_observations() -> None:
    empty = obs([], [])
    assert links_by_id(empty, obs([0], [True])).shape == (1, 0)
    assert links_by_id(obs([0], [True]), empty).shape == (0, 1)


def test_one_to_one_stages_use_the_ids(field: AnalyticField) -> None:
    trace = Observer(PointSampler(30), [GaussianNoise(0.1), RandomDropout(0.4)]).trace(
        field, 0
    )
    assert all(link is None for link in trace._links)
    noise, drop = trace.diffs
    assert np.array_equal(noise.links.toarray(), np.eye(30))
    assert (drop.links.toarray().sum(axis=0) == drop.kept.ravel()).all()


def test_dependency_multiplies_the_links(field: AnalyticField) -> None:
    observer = Observer(
        GridSampler((8, 8)), [Downsample(2), RandomDropout(0.3), Downsample(2)]
    )
    trace = observer.trace(field, 5)
    whole = trace.links_between()
    assert whole.shape == (4, 64)
    middle = trace.links_between(1, 2)
    assert middle.shape == (16, 16)
    assert np.array_equal(trace.links_between(2, 2).toarray(), np.eye(16))
    # each final location traces back to sampler locations inside its own 4x4 block
    for j in range(4):
        cols = whole[j].nonzero()[1]
        rows, cols2d = np.divmod(cols, 8)
        assert (rows // 4 == j // 2).all() and (cols2d // 4 == j % 2).all()
    with pytest.raises(ValueError, match="after stop"):
        trace.links_between(2, 1)


def test_trace_slices_keep_their_links(field: AnalyticField) -> None:
    trace = Observer(GridSampler(8), [GaussianNoise(0.1), Downsample(2)]).trace(
        field, 1
    )
    sliced = trace[1:]
    assert sliced.diffs[0].links is trace.diffs[1].links
    assert trace[::2].diffs[0].links.shape == (16, 64)  # falls back to ids
    assert len(trace[2:2].diffs) == 0


def test_trace_checks_the_number_of_links(field: AnalyticField) -> None:
    trace = Observer(GridSampler(4), [GaussianNoise(0.1)]).trace(field, 0)
    with pytest.raises(ValueError, match="expected 1 links"):
        Trace(list(trace), [None, None])
