import numpy as np
from scipy.stats import norm, poisson, randint

from scisynth import Tabular, generate

COLUMNS = [norm(0, 1), poisson(mu=5), randint(0, 4)]


def test_basic_shape():
    data = Tabular(columns=COLUMNS).rvs(n=100, seed=0)
    assert data.X.shape == (100, 3)
    assert data.Z.shape == (100, 3)
    assert data.ids.shape == (100,)


def test_reproducible():
    d1 = Tabular(columns=COLUMNS).rvs(n=50, seed=42)
    d2 = Tabular(columns=COLUMNS).rvs(n=50, seed=42)
    np.testing.assert_array_equal(d1.X, d2.X)


def test_different_seeds_differ():
    d1 = Tabular(columns=COLUMNS).rvs(n=50, seed=0)
    d2 = Tabular(columns=COLUMNS).rvs(n=50, seed=1)
    assert not np.array_equal(d1.X, d2.X)


def test_marginals_respected():
    data = Tabular(columns=COLUMNS).rvs(n=1000, seed=0)

    poisson_col = data.X[:, 1]
    assert np.all(poisson_col >= 0)
    assert np.all(poisson_col == np.floor(poisson_col))

    cat_col = data.X[:, 2]
    assert set(cat_col.astype(int)).issubset({0, 1, 2, 3})


def test_correlated_columns():
    corr = [[1.0, 0.9], [0.9, 1.0]]
    data = Tabular(columns=[norm(0, 1), norm(0, 1)], corr=corr).rvs(n=2000, seed=0)
    empirical = np.corrcoef(data.X[:, 0], data.X[:, 1])[0, 1]
    assert abs(empirical - 0.9) < 0.05


def test_zero_correlation():
    corr = [[1.0, 0.0], [0.0, 1.0]]
    data = Tabular(columns=[norm(0, 1), norm(0, 1)], corr=corr).rvs(n=2000, seed=0)
    empirical = np.corrcoef(data.X[:, 0], data.X[:, 1])[0, 1]
    assert abs(empirical) < 0.05


def test_spec_roundtrip():
    spec = Tabular(columns=COLUMNS).to_spec()
    data = generate(spec)
    assert data.X.shape == (1000, 3)
    assert data.family == "tabular"


def test_to_frame():
    data = Tabular(columns=COLUMNS).rvs(n=10, seed=0)
    df = data.to_frame()
    assert df.shape == (10, 4)  # id + 3 features
    assert df.columns[0] == "id"
