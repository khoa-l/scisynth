import numpy as np
import pytest

from scisynth.observed.operators import (
    BoundingBoxSubsample,
    DenseSubsample,
    RadiusSubsample,
    SparseSubsample,
    StratifiedSubsample,
    SystematicSubsample,
    ThresholdSubsample,
    UniformSubsample,
)


def ids_are_subset(result, original):
    """All returned ids must come from the original set."""
    return set(result.ids.tolist()).issubset(set(original.ids.tolist()))


class TestUniformSubsample:
    def test_frac(self, state, rng):
        out = UniformSubsample(frac=0.5).apply(state, rng)
        assert out.X.shape[0] == 250

    def test_n(self, state, rng):
        out = UniformSubsample(n=100).apply(state, rng)
        assert out.X.shape[0] == 100

    def test_ids_preserved(self, state, rng):
        out = UniformSubsample(n=100).apply(state, rng)
        assert ids_are_subset(out, state)

    def test_reproducible(self, state):
        out1 = UniformSubsample(n=100).apply(state, np.random.default_rng(0))
        out2 = UniformSubsample(n=100).apply(state, np.random.default_rng(0))
        np.testing.assert_array_equal(out1.ids, out2.ids)

    def test_n_gte_total_returns_unchanged(self, state, rng):
        out = UniformSubsample(n=9999).apply(state, rng)
        assert out.X.shape[0] == state.X.shape[0]

    def test_requires_exactly_one_param(self):
        with pytest.raises(ValueError):
            UniformSubsample(frac=0.5, n=100)
        with pytest.raises(ValueError):
            UniformSubsample()


class TestSparseSubsample:
    def test_shape(self, state, rng):
        out = SparseSubsample(n=100).apply(state, rng)
        assert out.X.shape[0] == 100

    def test_ids_preserved(self, state, rng):
        out = SparseSubsample(n=100).apply(state, rng)
        assert ids_are_subset(out, state)

    def test_favors_sparse_regions(self, clustered_state, rng):
        # outliers are rows 400–499; sparse sampling should keep more of them
        out = SparseSubsample(n=100, k=10).apply(clustered_state, rng)
        outlier_count = (out.ids >= 400).sum()
        # outliers are 20% of data but should be well over 20% of kept rows
        assert outlier_count > 30


class TestDenseSubsample:
    def test_shape(self, state, rng):
        out = DenseSubsample(n=100).apply(state, rng)
        assert out.X.shape[0] == 100

    def test_ids_preserved(self, state, rng):
        out = DenseSubsample(n=100).apply(state, rng)
        assert ids_are_subset(out, state)

    def test_favors_dense_regions(self, clustered_state, rng):
        # outliers are rows 400–499; dense sampling should keep very few of them
        out = DenseSubsample(n=100, k=10).apply(clustered_state, rng)
        outlier_count = (out.ids >= 400).sum()
        assert outlier_count < 20


class TestBoundingBoxSubsample:
    def test_rows_within_bounds(self, state, rng):
        out = BoundingBoxSubsample(boxes=[[(-1, 1), (-1, 1), (-1, 1)]]).apply(
            state, rng
        )
        assert np.all(out.X[:, 0] >= -1) and np.all(out.X[:, 0] <= 1)

    def test_open_bounds(self, state, rng):
        # only upper bound on column 0
        out = BoundingBoxSubsample(
            boxes=[[(None, 0), (None, None), (None, None)]]
        ).apply(state, rng)
        assert np.all(out.X[:, 0] <= 0)

    def test_union_of_boxes(self, state, rng):
        single_a = BoundingBoxSubsample(
            boxes=[[(-2, -0.5), (None, None), (None, None)]]
        ).apply(state, rng)
        single_b = BoundingBoxSubsample(
            boxes=[[(0.5, 2), (None, None), (None, None)]]
        ).apply(state, rng)
        both = BoundingBoxSubsample(
            boxes=[
                [(-2, -0.5), (None, None), (None, None)],
                [(0.5, 2), (None, None), (None, None)],
            ]
        ).apply(state, rng)
        assert both.X.shape[0] == single_a.X.shape[0] + single_b.X.shape[0]

    def test_ids_preserved(self, state, rng):
        out = BoundingBoxSubsample(boxes=[[(-1, 1), (None, None), (None, None)]]).apply(
            state, rng
        )
        assert ids_are_subset(out, state)


class TestRadiusSubsample:
    def test_rows_within_radius(self, state, rng):
        center = [0.0, 0.0, 0.0]
        radius = 1.0
        out = RadiusSubsample(sites=[{"center": center, "radius": radius}]).apply(
            state, rng
        )
        dists = np.linalg.norm(out.X - np.array(center), axis=1)
        assert np.all(dists <= radius)

    def test_union_of_sites(self, state, rng):
        s1 = RadiusSubsample(sites=[{"center": [-1, 0, 0], "radius": 0.5}]).apply(
            state, rng
        )
        s2 = RadiusSubsample(sites=[{"center": [1, 0, 0], "radius": 0.5}]).apply(
            state, rng
        )
        both = RadiusSubsample(
            sites=[
                {"center": [-1, 0, 0], "radius": 0.5},
                {"center": [1, 0, 0], "radius": 0.5},
            ]
        ).apply(state, rng)
        assert both.X.shape[0] == s1.X.shape[0] + s2.X.shape[0]

    def test_ids_preserved(self, state, rng):
        out = RadiusSubsample(sites=[{"center": [0, 0, 0], "radius": 1.0}]).apply(
            state, rng
        )
        assert ids_are_subset(out, state)


class TestThresholdSubsample:
    def test_above(self, state, rng):
        out = ThresholdSubsample(
            conditions=[{"column": 0, "threshold": 0.0, "direction": "above"}]
        ).apply(state, rng)
        assert np.all(out.X[:, 0] >= 0.0)

    def test_below(self, state, rng):
        out = ThresholdSubsample(
            conditions=[{"column": 0, "threshold": 0.0, "direction": "below"}]
        ).apply(state, rng)
        assert np.all(out.X[:, 0] <= 0.0)

    def test_or_gate_larger_than_either_alone(self, state, rng):
        cond_a = [{"column": 0, "threshold": 1.5, "direction": "above"}]
        cond_b = [{"column": 1, "threshold": -1.5, "direction": "below"}]
        only_a = ThresholdSubsample(conditions=cond_a).apply(state, rng)
        only_b = ThresholdSubsample(conditions=cond_b).apply(state, rng)
        both = ThresholdSubsample(conditions=cond_a + cond_b, mode="any").apply(
            state, rng
        )
        assert both.X.shape[0] >= max(only_a.X.shape[0], only_b.X.shape[0])

    def test_and_gate_smaller_than_either_alone(self, state, rng):
        conditions = [
            {"column": 0, "threshold": 0.5, "direction": "above"},
            {"column": 1, "threshold": 0.5, "direction": "above"},
        ]
        only_0 = ThresholdSubsample(conditions=[conditions[0]]).apply(state, rng)
        only_1 = ThresholdSubsample(conditions=[conditions[1]]).apply(state, rng)
        both = ThresholdSubsample(conditions=conditions, mode="all").apply(state, rng)
        assert both.X.shape[0] <= min(only_0.X.shape[0], only_1.X.shape[0])

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            ThresholdSubsample(conditions=[], mode="invalid")


class TestStratifiedSubsample:
    def test_roughly_equal_strata(self, state, rng):
        out = StratifiedSubsample(n_per_stratum=20, n_strata=4).apply(state, rng)
        assert out.X.shape[0] == 80

    def test_ids_preserved(self, state, rng):
        out = StratifiedSubsample(n_per_stratum=20, n_strata=4).apply(state, rng)
        assert ids_are_subset(out, state)


class TestSystematicSubsample:
    def test_every_nth(self, state, rng):
        out = SystematicSubsample(step=10).apply(state, rng)
        assert out.X.shape[0] == 50  # 500 / 10

    def test_offset(self, state, rng):
        out = SystematicSubsample(step=10, offset=5).apply(state, rng)
        # ids should be 5, 15, 25, ...
        assert out.ids[0] == 5
        assert out.ids[1] == 15

    def test_ids_are_correct_interval(self, state, rng):
        step = 7
        out = SystematicSubsample(step=step).apply(state, rng)
        diffs = np.diff(out.ids)
        assert np.all(diffs == step)
