import pytest

from scisynth import (
    GaussianNoise,
    GridSampler,
    IncompatibleLatentError,
    Observer,
    PointSampler,
    Sampler,
)


@pytest.mark.parametrize("sampler", [GridSampler(4), PointSampler(4)])
def test_kind_mismatch_raises_clear_error(sampler: Sampler, catalog: object) -> None:
    with pytest.raises(IncompatibleLatentError) as exc:
        Observer(sampler, [GaussianNoise()]).run(catalog, seed=0)  # type: ignore[arg-type]
    msg = str(exc.value)
    assert type(sampler).__name__ in msg
    assert "['field']" in msg and "'catalog'" in msg and "FakeCatalog" in msg


def test_error_is_a_type_error(catalog: object) -> None:
    with pytest.raises(TypeError):
        GridSampler(4).run(catalog, seed=0)  # type: ignore[arg-type]
