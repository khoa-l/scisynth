import numpy as np
import pytest

from scisynth.latent import Independent
from scisynth.observed import ObservationState


@pytest.fixture
def state():
    """500 rows, 3 features — reused across observed operator tests."""
    data = Independent(p=3).rvs(n=500, seed=0)
    return ObservationState(X=data.X, ids=data.ids)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def clustered_state():
    """Tight cluster (rows 0–399) + sparse outliers (rows 400–499)."""
    rng = np.random.default_rng(0)
    cluster = rng.normal(loc=[0, 0], scale=0.2, size=(400, 2))
    outliers = rng.uniform(-5, 5, size=(100, 2))
    X = np.vstack([cluster, outliers])
    return ObservationState(X=X, ids=np.arange(len(X), dtype=np.int64))
