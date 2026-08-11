import numpy as np
import pytest

from scisynth import Spec, generate


def test_basic_shape():
    spec = Spec(family="independent_features", n=100, params={"p": 5})
    d = generate(spec)
    assert d.X.shape == (100, 5)
    assert d.Z.shape == (100, 5)
    assert d.ids.shape == (100,)


def test_reproducible():
    spec = Spec(family="independent_features", n=50, seed=42)
    d1 = generate(spec)
    d2 = generate(spec)
    np.testing.assert_array_equal(d1.X, d2.X)


def test_different_seeds_differ():
    d1 = generate(Spec(family="independent_features", n=50, seed=0))
    d2 = generate(Spec(family="independent_features", n=50, seed=1))
    assert not np.array_equal(d1.X, d2.X)


def test_unknown_family_raises():
    with pytest.raises(KeyError, match="Unknown latent family"):
        generate(Spec(family="does_not_exist"))
