import numpy as np
import pytest

from scisynth import observe, project
from scisynth.latent import Independent
from scisynth.observed import UniformSubsample


class _IdentityDR:
    def fit_transform(self, X):
        return X[:, :2]


class _SeedCapture:
    def __init__(self):
        self.random_state = None

    def set_params(self, **kw):
        self.random_state = kw.get("random_state")
        return self

    def get_params(self):
        return {"random_state": self.random_state}

    def fit_transform(self, X):
        return X[:, :2]


@pytest.fixture
def latent():
    return Independent(p=5).rvs(n=200, seed=0)


@pytest.fixture
def observed(latent):
    obs, _ = observe(latent, [UniformSubsample(n=100)], seed=0)
    return obs


@pytest.fixture
def prov(latent):
    _, prov = observe(latent, [UniformSubsample(n=100)], seed=0)
    return prov


class TestProjectedData:
    def test_shape(self, observed):
        result, _ = project(observed, _IdentityDR())
        assert result.Z.shape == (100, 2)
        assert result.ids.shape == (100,)

    def test_ids_preserved(self, observed):
        result, _ = project(observed, _IdentityDR())
        np.testing.assert_array_equal(result.ids, observed.ids)

    def test_method_name_recorded(self, observed):
        result, _ = project(observed, _IdentityDR())
        assert result.method == "_IdentityDR"

    def test_to_frame(self, observed):
        result, _ = project(observed, _IdentityDR())
        df = result.to_frame()
        assert df.shape == (100, 3)  # id + z0 + z1
        assert df.columns[0] == "id"
        assert df.columns[1] == "z0"


class TestProjectSeed:
    def test_random_state_injected(self, observed):
        cap = _SeedCapture()
        project(observed, cap, seed=99)
        assert cap.random_state == 99

    def test_params_recorded(self, observed):
        cap = _SeedCapture()
        result, _ = project(observed, cap, seed=7)
        assert result.params["random_state"] == 7

    def test_missing_set_params_does_not_raise(self, observed):
        result, _ = project(observed, _IdentityDR(), seed=42)
        assert result.Z.shape[0] == 100


class TestProjectProvenance:
    def test_no_provenance_returns_none(self, observed):
        _, prov = project(observed, _IdentityDR())
        assert prov is None

    def test_provenance_records_projected_step(self, observed, prov):
        _, updated = project(observed, _IdentityDR(), provenance=prov)
        assert len(updated.projected_steps) == 1

    def test_projected_step_stores_z(self, observed, prov):
        _, updated = project(observed, _IdentityDR(), provenance=prov)
        assert updated.projected_steps[0].X.shape == (100, 2)

    def test_projected_step_stores_ids(self, observed, prov):
        _, updated = project(observed, _IdentityDR(), provenance=prov)
        np.testing.assert_array_equal(updated.projected_steps[0].ids, observed.ids)

    def test_projected_step_records_method_name(self, observed, prov):
        _, updated = project(observed, _IdentityDR(), provenance=prov)
        assert updated.projected_steps[0].operator.name == "_IdentityDR"

    def test_original_provenance_not_mutated(self, observed, prov):
        project(observed, _IdentityDR(), provenance=prov)
        assert len(prov.projected_steps) == 0

    def test_to_frame_includes_projected_row(self, observed, prov):
        _, updated = project(observed, _IdentityDR(), provenance=prov)
        df = updated.to_frame()
        assert "projected" in df["layer"].to_list()
