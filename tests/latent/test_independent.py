import numpy as np
import pytest

from scisynth import Spec, generate
from scisynth.latent import Independent


def test_basic_shape():
    data = Independent(p=5).rvs(n=100, seed=0)
    assert data.X.shape == (100, 5)
    assert data.Z.shape == (100, 5)
    assert data.ids.shape == (100,)


def test_z_equals_x():
    # Independent has no hidden structure so intrinsic == ambient.
    data = Independent(p=4).rvs(n=50, seed=0)
    np.testing.assert_array_equal(data.Z, data.X)


def test_reproducible():
    d1 = Independent(p=3).rvs(n=50, seed=42)
    d2 = Independent(p=3).rvs(n=50, seed=42)
    np.testing.assert_array_equal(d1.X, d2.X)


def test_different_seeds_differ():
    d1 = Independent(p=3).rvs(n=50, seed=0)
    d2 = Independent(p=3).rvs(n=50, seed=1)
    assert not np.array_equal(d1.X, d2.X)


def test_correlated_path():
    data = Independent(p=4, correlated=True).rvs(n=500, seed=0)
    assert data.X.shape == (500, 4)


def test_spec_roundtrip():
    dist = Independent(p=5)
    data = generate(dist.to_spec())
    assert data.X.shape == (1000, 5)
    assert data.family == "independent_features"


def test_generate_via_spec():
    spec = Spec(family="independent_features", n=100, params={"p": 5})
    data = generate(spec)
    assert data.X.shape == (100, 5)


def test_unknown_family_raises():
    with pytest.raises(KeyError, match="Unknown latent family"):
        generate(Spec(family="does_not_exist"))


def test_to_frame():
    data = Independent(p=3).rvs(n=10, seed=0)
    df = data.to_frame()
    assert df.shape == (10, 4)  # id + 3 features
    assert df.columns[0] == "id"
