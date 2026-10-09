import numpy as np
import pytest
from scipy import stats

from scisynth import (
    AnalyticField,
    Downsample,
    GridSampler,
    Observation,
    Observer,
    PointSampler,
    PositionJitter,
    RandomDropout,
)
from scisynth.testing.invariants import check_observation


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def grid(field: AnalyticField, shape: tuple[int, ...] = (6, 4)) -> Observation:
    return GridSampler(shape)(field, rng())


def test_blocks_are_averaged(field: AnalyticField) -> None:
    obs = grid(field)
    out = Downsample(2)(obs, rng())
    check_observation(out)
    assert out.spatial_shape == (3, 2) and out.layout == "grid"
    blocks = obs.values[..., 0].reshape(3, 2, 2, 2).mean(axis=(1, 3))
    assert np.allclose(out.values[..., 0], blocks)
    assert np.allclose(out.truth[..., 0], blocks)  # type: ignore[index]
    assert np.allclose(out.coords["x"], obs.coords["x"].reshape(3, 2).mean(axis=1))
    assert np.allclose(out.coords["y"], obs.coords["y"].reshape(2, 2).mean(axis=1))


def test_masked_inputs_are_left_out_of_the_mean(field: AnalyticField) -> None:
    obs = RandomDropout(0.6)(grid(field, (8, 8)), rng(1))
    out = Downsample(4)(obs, rng())
    check_observation(out)
    for i in range(2):
        for j in range(2):
            block = (slice(4 * i, 4 * i + 4), slice(4 * j, 4 * j + 4))
            valid = obs.values[block][obs.mask[block]]
            assert out.mask[i, j] == (len(valid) > 0)
            if len(valid):
                assert np.isclose(out.values[i, j, 0], valid.mean())


def test_factor_per_axis_and_uneven_edges(field: AnalyticField) -> None:
    obs = grid(field, (5, 7))
    out = Downsample([2, 3])(obs, rng())
    assert out.spatial_shape == (3, 3)
    assert np.isclose(out.values[2, 2, 0], obs.values[4:, 6:, 0].mean())  # edge block
    assert out.coords["x"][2] == obs.coords["x"][4]
    assert Downsample(1)(obs, rng()).spatial_shape == (5, 7)


def test_ids_are_new_and_links_average_the_block(field: AnalyticField) -> None:
    obs = grid(field)
    trace = Observer(GridSampler((6, 4)), [Downsample(2)]).trace(field, 0)
    out = trace[1]
    assert out.ids.min() == obs.ids.max() + 1 and out.ids.size == 6
    links = trace.diffs[0].links
    assert links.shape == (6, 24)
    assert np.allclose(links.sum(axis=1), 1.0)
    assert (links.getnnz(axis=1) == 4).all()
    # applying the links to the values reproduces the stage
    assert np.allclose(links @ obs.values.reshape(24, 1), out.values.reshape(6, 1))


def test_diff_of_a_downsample(field: AnalyticField) -> None:
    observer = Observer(GridSampler((8, 8)), [RandomDropout(0.5), Downsample(4)])
    drop, down = observer.trace(field, 3).diffs
    assert down.n_dropped == 0 and down.n_moved == 0  # merged, not moved
    assert down.n_added == 0
    assert down.parents.sum() == drop.n_kept
    assert (down.children[drop.after.mask] == 1).all()


def test_links_use_the_factor_that_was_drawn(field: AnalyticField) -> None:
    observer = Observer(GridSampler((6, 6)), [Downsample(stats.randint(2, 4))])
    trace = observer.trace(field, 0)
    factor = trace[-1].meta[-1]["params"]["factor"]
    assert trace.diffs[0].links.shape == (trace[1].size, 36)
    assert trace[1].spatial_shape == (6 // factor,) * 2


def test_distribution_factor_must_be_an_integer(field: AnalyticField) -> None:
    obs = grid(field)
    with pytest.raises(ValueError, match="integer"):
        Downsample(1.5)(obs, rng())
    with pytest.raises(ValueError, match="integer"):
        Downsample(0)(obs, rng())
    with pytest.raises(ValueError, match="one entry per axis"):
        Downsample([2, 2, 2])(obs, rng())
    out = Downsample(stats.randint(2, 3))(obs, rng())  # always 2
    assert out.meta[-1]["params"]["factor"] == 2


def test_points_are_binned(field: AnalyticField) -> None:
    obs = PointSampler(400)(field, rng())  # about a 20 x 20 grid
    out = Downsample(4)(obs, rng())  # about 5 x 5 bins
    check_observation(out)
    assert out.layout == "points" and 15 <= out.size <= 25
    assert out.ids.min() == obs.ids.max() + 1
    weights = Observer(PointSampler(400), [Downsample(4)]).trace(field, 0).diffs[0]
    assert np.allclose(weights.links.sum(axis=1), 1.0)
    assert weights.links.getnnz() == 400  # every point lands in one bin
    # the links reproduce the values and the centroid coordinates
    assert np.allclose(weights.links @ weights.before.values, weights.after.values)
    assert np.allclose(
        weights.links @ weights.before.coords["x"], weights.after.coords["x"]
    )


def test_masked_points_are_left_out_and_empty_bins_dropped(
    field: AnalyticField,
) -> None:
    obs = RandomDropout(0.5)(PointSampler(200)(field, rng()), rng(1))
    out = Downsample(5)(obs, rng())
    check_observation(out)
    assert out.mask.sum() <= out.size
    valid = obs.values[obs.mask]
    assert out.values[out.mask].min() >= valid.min()
    assert out.values[out.mask].max() <= valid.max()


def test_a_grid_that_became_points_is_binned(field: AnalyticField) -> None:
    observer = Observer(GridSampler(16), [PositionJitter(0.02), Downsample(4)])
    trace = observer.trace(field, 0)
    check_observation(trace[-1])
    assert trace[-1].layout == "points" and trace[-1].size < 256 // 4


def test_multichannel_and_one_axis() -> None:
    obs = Observation(
        np.arange(12, dtype=np.float64).reshape(6, 2),
        {"t": np.arange(6.0)},
        np.ones(6, bool),
    )
    out = Downsample(3)(obs, rng())
    assert np.allclose(out.values, [[2.0, 3.0], [8.0, 9.0]])
    assert out.coords["t"].tolist() == [1.0, 4.0]


def test_round_trips_through_a_dict() -> None:
    observer = Observer(GridSampler(8), [Downsample([2, 4])])
    assert Observer.from_dict(observer.to_dict()) == observer


def test_points_without_a_position_are_ignored(field: AnalyticField) -> None:
    obs = PointSampler(200)(field, rng())
    keep = np.arange(obs.size) % 2 == 0
    coords = {k: np.where(keep, c, np.nan) for k, c in obs.coords.items()}
    holey = obs.replace(
        coords=coords, mask=keep, values=np.where(keep[:, None], obs.values, np.nan)
    )
    with np.errstate(all="raise"):
        out = Downsample(2)(holey, rng())
    check_observation(out)
    assert np.isfinite(out.coords["x"]).all() and np.isfinite(out.coords["y"]).all()
    clean = Downsample(2)(
        holey.replace(
            coords={k: c[keep] for k, c in coords.items()},
            mask=keep[keep],
            values=obs.values[keep],
            truth=obs.truth[keep] if obs.truth is not None else None,
            ids=obs.ids[keep],
        ),
        rng(),
    )
    assert np.allclose(out.values, clean.values)
    assert np.allclose(out.coords["x"], clean.coords["x"])


def test_all_points_without_a_position_downsample_to_nothing(
    field: AnalyticField,
) -> None:
    obs = PointSampler(10)(field, rng())
    nowhere = obs.replace(
        coords={k: np.full(10, np.nan) for k in obs.coords},
        mask=np.zeros(10, dtype=bool),
        values=np.full_like(obs.values, np.nan),
    )
    out = Downsample(2)(nowhere, rng())
    assert out.size == 0


def test_a_valid_point_without_a_position_raises(field: AnalyticField) -> None:
    obs = PointSampler(10)(field, rng())
    x = obs.coords["x"].copy()
    x[3] = np.nan
    with pytest.raises(ValueError, match=r"1 valid location.*non-finite"):
        Downsample(2)(obs.replace(coords={**obs.coords, "x": x}), rng())
