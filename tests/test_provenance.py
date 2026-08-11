import numpy as np
import pytest

from scisynth import Independent, ObservedData, Provenance, observe
from scisynth.observed.operators import GaussianNoise, UniformSubsample


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
        observed, _ = observe(latent, [UniformSubsample(n=80), GaussianNoise(sigma=0.1)])
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

    def test_state_after(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80), GaussianNoise(sigma=0.1)])
        X, ids = prov.state_after(0)
        assert X.shape == (80, 3)
        assert ids.shape == (80,)

    def test_projected_steps_initially_empty(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        assert prov.projected_steps == []


class TestProvenanceMatrix:
    def test_latent_to_observed_shape(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        W = prov.latent_to_observed()
        assert W.shape == (80, 200)

    def test_each_observed_row_maps_to_one_latent(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        W = prov.latent_to_observed()
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0)

    def test_coverage_sums_to_observed_count(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        assert prov.coverage().sum() == 80

    def test_coverage_shape(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        assert prov.coverage().shape == (200,)

    def test_unobserved_latent_points_have_zero_coverage(self, latent):
        _, prov = observe(latent, [UniformSubsample(n=80)])
        cov = prov.coverage()
        assert (cov == 0).sum() == 120  # 200 - 80 not observed
