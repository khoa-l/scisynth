import numpy as np
import pytest

from scisynth import AnalyticField, Domain, GaussianField, Product, Sum
from scisynth.latent.compose import Multichannel


def test_sum_and_product_and_their_operators(domain: Domain) -> None:
    a = AnalyticField(lambda x, y: x, domain)
    b = AnalyticField(lambda x, y: y + 1, domain)
    pts = np.array([[0.5, 1.0], [0.25, 0.0]])
    assert np.allclose(Sum(a, b).evaluate(pts), [[2.5], [1.25]])
    assert np.allclose(Product(a, b).evaluate(pts), [[1.0], [0.25]])
    assert Sum(a, b).kind == "field" and Sum(a, b).domain == domain
    assert np.allclose(
        ((a + b) * a).evaluate(pts), [[1.25], [0.3125]]
    )  # operators nest


def test_operators_hand_back_things_that_are_not_latents(
    field: AnalyticField, domain: Domain
) -> None:
    for other in (1, 2.5, "a", None):
        with pytest.raises(TypeError, match="unsupported operand"):
            field + other  # type: ignore[operator]
        with pytest.raises(TypeError, match=r"unsupported operand|can't multiply"):
            field * other  # type: ignore[operator]
    with pytest.raises(TypeError, match="unsupported operand"):
        1 + field  # type: ignore[operator]
    # a generator is still passed on, so its error says how to fix it
    with pytest.raises(TypeError, match="realize generators first"):
        field + GaussianField(domain)  # type: ignore[operator]


def test_rejects_non_fields_and_too_few_terms(
    catalog: object, field: AnalyticField
) -> None:
    with pytest.raises(TypeError, match="kind 'field'"):
        Sum(field, catalog)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two"):
        Sum(field)


def test_rejects_mismatched_dimensions(field: AnalyticField) -> None:
    other = AnalyticField(lambda x: x, Domain.from_extents([(0, 1)]))
    with pytest.raises(ValueError, match="number of axes"):
        Product(field, other)


def test_multichannel_stacks_channels_in_order(domain: Domain) -> None:
    a = AnalyticField(lambda x, y: x, domain)
    b = AnalyticField(lambda x, y: y + 1, domain)
    pts = np.array([[0.5, 1.0], [0.25, 0.0]])
    mc = Multichannel(a, b)
    assert np.allclose(mc.evaluate(pts), [[0.5, 2.0], [0.25, 1.0]])
    assert mc.kind == "field"
    assert mc.n_channels == 2
    assert mc.domain == domain
    assert Multichannel(a).evaluate(pts).shape == (2, 1)


def _corr(values: np.ndarray) -> float:
    return float(np.asarray(np.corrcoef(values.T))[0, 1])


def test_multichannel_corr_is_none_without_a_matrix(domain: Domain) -> None:
    field = GaussianField(domain).realize(0)
    assert Multichannel(field, field).corr is None


def test_multichannel_corr_mixes_independent_channels(domain: Domain) -> None:
    fields = [GaussianField(domain, length_scale=0.1).realize(s) for s in (1, 2)]
    corr = [[1.0, 0.8], [0.8, 1.0]]
    pts = np.random.default_rng(0).uniform(0, 1, size=(20_000, 2))
    plain = Multichannel(*fields).evaluate(pts)
    mixed = Multichannel(*fields, corr=corr).evaluate(pts)
    assert abs(_corr(plain)) < 0.2
    assert _corr(mixed) == pytest.approx(0.8, abs=0.1)
    assert np.allclose(mixed[:, 0], plain[:, 0])  # first channel is unchanged


def test_multichannel_validation(field: AnalyticField, catalog: object) -> None:
    with pytest.raises(ValueError, match="at least one"):
        Multichannel()
    with pytest.raises(TypeError, match="kind 'field'"):
        Multichannel(field, catalog)  # type: ignore[arg-type]
    other = AnalyticField(lambda x: x, Domain.from_extents([(0, 1)]))
    with pytest.raises(ValueError, match="number of axes"):
        Multichannel(field, other)
    with pytest.raises(ValueError, match="2x2"):
        Multichannel(field, field, corr=[[1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        Multichannel(field, field, corr=[[1.0, 0.5], [0.2, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        Multichannel(field, field, corr=[[1.0, 1.5], [1.5, 1.0]])


def test_generators_and_distributions_are_rejected_with_a_clear_error(
    domain: Domain,
) -> None:
    from scipy import stats

    generator = GaussianField(domain)
    with pytest.raises(TypeError, match="realize generators first"):
        Multichannel(generator, generator)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="realized latents"):
        Sum(stats.norm(), stats.norm())
