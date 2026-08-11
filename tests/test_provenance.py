import numpy as np
import pytest

from scisynth import observe
from scisynth.latent import Independent
from scisynth.observed import GaussianNoise, UniformSubsample


@pytest.fixture
def latent():
    return Independent(p=3).rvs(n=200, seed=0)


class TestObservedData:
    def test_shape(self, latent):
        observed, _ = observe(latent, [UniformSubsample(n=80)])
        assert observed.X.shape == (80, 3)
        assert observed.ids.shape == (80,)

    def test_blind_to_latent(self, latent):
        observed, _ = observe(latent, [UniformSubsample(n=80)])
        assert not hasattr(observed, "source")
        assert not hasattr(observed, "latent")

    def test_stores_operator_specs(self, latent):
        observed, _ = observe(
            latent, [UniformSubsample(n=80), GaussianNoise(sigma=0.1)]
        )
        assert len(observed.operators) == 2
        assert observed.operators[0].name == "uniform_subsample"
        assert observed.operators[1].name == "gaussian_noise"

    def test_to_frame(self, latent):
        observed, _ = observe(latent, [UniformSubsample(n=50)])
        df = observed.to_frame()
        assert df.shape == (50, 4)  # id + 3 features
        assert df.columns[0] == "id"


class TestObserve:
    def test_no_operators_is_identity(self, latent):
        observed, prov = observe(latent, [])
        assert observed.X.shape == latent.X.shape
        assert len(prov.observed_steps) == 0

    def test_records_one_step_per_operator(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80), GaussianNoise(sigma=0.1)])
        assert len(prov.observed_steps) == 2

    def test_reproducible(self, latent):
        obs1, _ = observe(latent, [UniformSubsample(n=80)], seed=7)
        obs2, _ = observe(latent, [UniformSubsample(n=80)], seed=7)
        np.testing.assert_array_equal(obs1.ids, obs2.ids)

    def test_different_seeds_differ(self, latent):
        obs1, _ = observe(latent, [UniformSubsample(n=80)], seed=0)
        obs2, _ = observe(latent, [UniformSubsample(n=80)], seed=1)
        assert not np.array_equal(obs1.ids, obs2.ids)


class TestProvenance:
    def test_ids_are_subset_of_latent(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        assert set(prov.ids.tolist()).issubset(set(latent.ids.tolist()))

    def test_ids_with_no_steps_returns_source_ids(self, latent):
        _, prov = observe(latent, [])
        np.testing.assert_array_equal(prov.ids, latent.ids)

    def test_step_layer_is_observed(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        assert prov.observed_steps[0].layer == "observed"

    def test_step_x_shape(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80), GaussianNoise(sigma=0.1)])
        assert prov.observed_steps[0].X.shape == (80, 3)
        assert prov.observed_steps[1].X.shape == (80, 3)

    def test_noise_changes_x_not_ids(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80), GaussianNoise(sigma=1.0)])
        np.testing.assert_array_equal(
            prov.observed_steps[0].ids, prov.observed_steps[1].ids
        )
        assert not np.allclose(prov.observed_steps[0].X, prov.observed_steps[1].X)

    def test_projected_steps_initially_empty(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        assert prov.projected_steps == []


class TestProvenanceStep:
    def test_to_frame_shape(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        df = prov.observed_steps[0].to_frame()
        assert df.shape == (80, 4)  # id + 3 features

    def test_to_frame_observed_columns(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        df = prov.observed_steps[0].to_frame()
        assert df.columns[0] == "id"
        assert df.columns[1] == "x0"

    def test_to_frame_ids_match(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        df = prov.observed_steps[0].to_frame()
        np.testing.assert_array_equal(df["id"].to_numpy(), prov.observed_steps[0].ids)


class TestProvenanceToFrame:
    def test_columns(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        df = prov.to_frame()
        assert set(df.columns) == {"layer", "step", "operator", "n_in", "n_out"}

    def test_row_count(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80), GaussianNoise(sigma=0.1)])
        df = prov.to_frame()
        assert len(df) == 3  # 1 latent + 2 observed steps

    def test_n_out_matches_n_in_next_row(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80), GaussianNoise(sigma=0.1)])
        df = prov.to_frame()
        assert df["n_out"][0] == df["n_in"][1]
        assert df["n_out"][1] == df["n_in"][2]

    def test_no_observed_steps(self, latent):
        _, prov = observe(latent, [])
        df = prov.to_frame()
        assert len(df) == 1
        assert df["layer"][0] == "latent"


class TestMappingFrame:
    def test_lo_shape(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        df = prov.mapping_frame()
        assert df.shape == (80, 3)  # observed_idx, latent_id, coverage

    def test_lo_columns(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        df = prov.mapping_frame()
        assert df.columns == ["latent_id", "observed_idx", "coverage"]

    def test_observed_idx_is_sequential(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        df = prov.mapping_frame()
        np.testing.assert_array_equal(df["observed_idx"].to_numpy(), np.arange(80))

    def test_latent_ids_are_subset_of_source(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        df = prov.mapping_frame()
        assert set(df["latent_id"].to_list()).issubset(set(latent.ids.tolist()))

    def test_coverage_is_one_for_bijective(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        df = prov.mapping_frame()
        assert (df["coverage"] == 1).all()

    def test_lop_adds_projected_idx_column(self, latent):
        from scisynth import project

        class _DR:
            def fit_transform(self, X):
                return X[:, :2]

        observed, prov = observe(latent, [UniformSubsample(n=80)])
        _, prov = project(observed, _DR(), provenance=prov)
        df = prov.mapping_frame()
        assert "projected_idx" in df.columns
        assert df.shape == (80, 4)

    def test_projected_idx_is_sequential(self, latent):
        from scisynth import project

        class _DR:
            def fit_transform(self, X):
                return X[:, :2]

        observed, prov = observe(latent, [UniformSubsample(n=80)])
        _, prov = project(observed, _DR(), provenance=prov)
        df = prov.mapping_frame()
        np.testing.assert_array_equal(df["projected_idx"].to_numpy(), np.arange(80))
