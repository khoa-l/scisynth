from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from scisynth import AnalyticField, Domain, GaussianField, InterpolatedField
from scisynth.core.component import Component


def test_analytic_field_evaluates_with_per_axis_args(domain: Domain) -> None:
    f = AnalyticField(lambda x, y: x + 10 * y, domain)
    assert f.kind == "field"
    out = f.evaluate(np.array([[0.5, 1.0], [1.0, 2.0]]))
    assert np.allclose(out, [[10.5], [21.0]])  # shape (n, 1)


def test_analytic_field_bad_shape(domain: Domain) -> None:
    f = AnalyticField(lambda x, y: 1.0, domain)
    with pytest.raises(ValueError, match="shape"):
        f.evaluate(np.zeros((3, 2)))


def test_grf_seed_reproducible_and_distinct(domain: Domain) -> None:
    gen = GaussianField(domain, length_scale=0.2, resolution=32)
    a, b, c = gen.realize(1), gen.realize(1), gen.realize(2)
    assert np.array_equal(a.values, b.values)
    assert not np.array_equal(a.values, c.values)


def test_grf_statistics(domain: Domain) -> None:
    f = GaussianField(
        domain, length_scale=0.1, amplitude=2.0, mean=5.0, resolution=64
    ).realize(0)
    assert f.values.shape == (64, 64)
    assert f.values.mean() == pytest.approx(5.0)
    assert f.values.std() == pytest.approx(2.0)


def test_grf_longer_correlation_is_smoother(domain: Domain) -> None:
    def roughness(length: float) -> float:
        v = GaussianField(domain, length, resolution=64).realize(0).values
        return float(np.abs(np.diff(v, axis=0)).mean())

    assert roughness(0.4) < roughness(0.05)


def test_grf_params_accept_distributions(domain: Domain) -> None:
    gen = GaussianField(
        domain,
        length_scale=stats.uniform(0.1, 0.2),
        amplitude=stats.uniform(1, 2),
    )
    assert np.array_equal(gen.realize(3).values, gen.realize(3).values)
    assert gen.realize(3).values.std() != gen.realize(4).values.std()


def test_grf_rejects_bad_length(domain: Domain) -> None:
    with pytest.raises(ValueError, match="length_scale"):
        GaussianField(domain, length_scale=0.0).realize(0)


def test_gaussian_field_realization_is_exact_at_the_nodes_and_wraps(
    domain: Domain,
) -> None:
    f = GaussianField(domain, 0.2, resolution=16).realize(0)
    assert isinstance(f, InterpolatedField)
    xs, ys = np.arange(16) * (1.0 / 16), np.arange(16) * (2.0 / 16)
    pts = np.array([[xs[3], ys[5]], [xs[3] + 1.0, ys[5] + 2.0]])
    assert np.allclose(f.evaluate(pts), f.values[3, 5])


def line_domain() -> Domain:
    return Domain.from_extents([(0.0, 1.0)])


def fine_points(n: int = 1600) -> NDArray[Any]:
    return np.linspace(0.0, 1.0, n, endpoint=False)[:, None]


def max_second_difference(f: InterpolatedField) -> float:
    return float(np.abs(np.diff(f.evaluate(fine_points())[:, 0], 2)).max())


def test_gaussian_field_exposes_order_and_always_wraps(domain: Domain) -> None:
    default = GaussianField(domain, 0.2, resolution=16).realize(0)
    cubic = GaussianField(domain, 0.2, resolution=16, order=3).realize(0)
    assert default.order == 1 and cubic.order == 3
    assert default.mode == cubic.mode == "grid-wrap"
    assert np.array_equal(default.values, cubic.values)  # order only changes querying


def test_order_changes_values_between_nodes_but_not_at_nodes() -> None:
    gen = GaussianField(line_domain(), 0.1, resolution=16)
    linear, cubic = (
        gen.realize(1),
        GaussianField(line_domain(), 0.1, resolution=16, order=3).realize(1),
    )
    nodes = (np.arange(16) / 16)[:, None]
    assert np.allclose(linear.evaluate(nodes)[:, 0], linear.values)
    assert np.allclose(
        cubic.evaluate(nodes)[:, 0], cubic.values
    )  # interpolating splines
    between = fine_points()
    assert not np.allclose(linear.evaluate(between), cubic.evaluate(between))


def test_cubic_is_smoother_than_linear_and_order_zero_is_nearest() -> None:
    def make(order: int) -> InterpolatedField:
        return GaussianField(line_domain(), 0.1, resolution=16, order=order).realize(2)

    assert max_second_difference(make(3)) < max_second_difference(make(1))
    nearest = make(0)
    out = nearest.evaluate(fine_points())[:, 0]
    assert set(np.round(out, 12)) <= set(np.round(nearest.values, 12))


@pytest.mark.parametrize("bad", [-1, 6])
def test_order_is_validated(bad: int, domain: Domain) -> None:
    with pytest.raises(ValueError, match="order must be between 0 and 5"):
        GaussianField(domain, 0.2, resolution=8, order=bad).realize(0)
    with pytest.raises(ValueError, match="order must be between 0 and 5"):
        InterpolatedField(np.zeros(4), line_domain(), order=bad)


def test_generator_with_an_order_round_trips(domain: Domain) -> None:
    gen = GaussianField(domain, 0.2, resolution=16, order=3)
    assert Component.from_dict(gen.to_dict()) == gen


def test_interpolated_field_holds_its_edge_values_unless_asked_to_wrap() -> None:
    values = np.array([10.0, 20.0, 30.0, 40.0])  # nodes at 0, 0.25, 0.5, 0.75
    f = InterpolatedField(values, line_domain())
    assert f.mode == "nearest" and f.order == 1
    outside = np.array([[-0.5], [-0.01], [0.0], [0.75], [0.9], [1.0], [1.5]])
    assert np.allclose(f.evaluate(outside)[:, 0], [10, 10, 10, 40, 40, 40, 40])
    assert np.allclose(f.evaluate(np.array([[0.125], [0.375]]))[:, 0], [15, 25])
    wrapped = InterpolatedField(values, line_domain(), mode="grid-wrap")
    edge = np.array([[0.875], [1.0], [1.25]])
    assert np.allclose(wrapped.evaluate(edge)[:, 0], [25, 10, 20])


def test_gaussian_field_refuses_an_oversized_grid_before_allocating() -> None:
    five_d = Domain.from_extents([(0.0, 1.0)] * 5)
    with pytest.raises(ValueError, match=r"resolution \(128, 128, 128, 128, 128\)"):
        GaussianField(five_d).realize(0)  # 128**5 cells would exhaust memory
    with pytest.raises(ValueError, match="over the limit of 100"):
        GaussianField(five_d, resolution=3, max_cells=100).realize(0)
    f = GaussianField(five_d, length_scale=0.3, resolution=6).realize(0)
    assert f.evaluate(np.full((2, 5), 0.5)).shape == (2, 1)
