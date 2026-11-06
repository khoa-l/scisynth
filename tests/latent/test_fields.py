from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from scisynth import (
    AnalyticField,
    BumpField,
    Domain,
    GaussianBumps,
    GaussianField,
    InterpolatedField,
    RegionField,
    VoronoiRegions,
)


def test_analytic_field_evaluates_with_per_axis_args(domain: Domain) -> None:
    f = AnalyticField(lambda x, y: x + 10 * y, domain)
    assert f.kind == "field"
    out = f.evaluate(np.array([[0.5, 1.0], [1.0, 2.0]]))
    assert np.allclose(out, [[10.5], [21.0]])  # shape (n, 1)


def test_analytic_field_bad_shape(domain: Domain) -> None:
    f = AnalyticField(lambda x, y: 1.0, domain)
    with pytest.raises(ValueError, match="shape"):
        f.evaluate(np.zeros((3, 2)))


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


def inside(points: NDArray[Any], domain: Domain) -> bool:
    lo, hi = np.array(domain.extents).T
    return bool(np.all((points >= lo) & (points <= hi)))


def test_bump_field_peaks_at_its_centers_and_falls_off_by_width() -> None:
    f = BumpField(
        np.array([[0.2], [0.8]]),
        np.array([0.05, 0.1]),
        np.array([1.0, 3.0]),
        line_domain(),
    )
    assert f.kind == "field"
    out = f.evaluate(np.array([[0.2], [0.8], [0.9], [0.5]]))[:, 0]
    assert np.allclose(out[:3], [1.0, 3.0, 3.0 * np.exp(-0.5)])
    assert out[3] < 0.1


def test_bump_field_with_no_bumps_is_zero(domain: Domain) -> None:
    f = BumpField(np.empty((0, 2)), np.empty(0), np.empty(0), domain)
    assert np.array_equal(f.evaluate(np.zeros((3, 2))), np.zeros((3, 1)))


def test_bump_field_validates_its_arrays(domain: Domain) -> None:
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        BumpField(np.zeros((2, 2)), np.full(1, 0.1), np.ones(2), domain)
    with pytest.raises(ValueError, match="widths must be > 0"):
        BumpField(np.zeros((1, 2)), np.zeros(1), np.ones(1), domain)


def test_gaussian_bumps_places_count_bumps_inside_the_domain(domain: Domain) -> None:
    f = GaussianBumps(domain, count=50).realize(0)
    assert isinstance(f, BumpField)
    assert f.centers.shape == (50, 2)
    assert inside(f.centers, domain)
    assert np.all(f.widths == 0.1) and np.all(f.heights == 1.0)


def test_gaussian_bumps_draws_width_and_height_per_bump(domain: Domain) -> None:
    gen = GaussianBumps(
        domain,
        count=stats.poisson(20),
        width=stats.uniform(0.05, 0.1),
        height=stats.norm(),
    )
    f = gen.realize(0)
    assert len(set(f.widths)) == len(f.widths) > 1
    assert len(set(f.heights)) == len(f.heights)
    assert np.all((f.widths >= 0.05) & (f.widths <= 0.15))


def test_gaussian_bumps_is_reproducible(domain: Domain) -> None:
    gen = GaussianBumps(domain, width=stats.uniform(0.05, 0.1))
    a, b, c = gen.realize(3), gen.realize(3), gen.realize(4)
    assert np.array_equal(a.centers, b.centers) and np.array_equal(a.widths, b.widths)
    assert not np.array_equal(a.centers, c.centers)


def test_gaussian_bumps_rejects_a_negative_count(domain: Domain) -> None:
    with pytest.raises(ValueError, match="count must be >= 0"):
        GaussianBumps(domain, count=-1).realize(0)


def test_region_field_takes_the_value_of_the_nearest_site() -> None:
    f = RegionField(
        np.array([[0.1], [0.5], [0.9]]), np.array([10.0, 20.0, 30.0]), line_domain()
    )
    assert f.kind == "field"
    pts = np.array([[0.0], [0.29], [0.31], [0.69], [0.71], [1.0]])
    assert np.array_equal(f.region(pts), [0, 0, 1, 1, 2, 2])
    assert np.array_equal(f.evaluate(pts)[:, 0], [10, 10, 20, 20, 30, 30])


def test_region_field_validates_its_arrays(domain: Domain) -> None:
    with pytest.raises(ValueError, match="at least one site"):
        RegionField(np.empty((0, 2)), np.empty(0), domain)
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        RegionField(np.zeros((2, 2)), np.ones(1), domain)


def test_voronoi_regions_default_to_a_label_map(domain: Domain) -> None:
    f = VoronoiRegions(domain, count=5).realize(0)
    assert isinstance(f, RegionField)
    assert f.sites.shape == (5, 2)
    assert inside(f.sites, domain)
    assert np.array_equal(f.values, np.arange(5))
    assert np.array_equal(f.evaluate(f.sites)[:, 0], np.arange(5))


def test_voronoi_regions_draw_a_value_per_region(domain: Domain) -> None:
    f = VoronoiRegions(
        domain, count=stats.poisson(8, loc=1), value=stats.norm()
    ).realize(1)
    assert len(set(f.values)) == len(f.values) > 1


def test_voronoi_regions_is_reproducible(domain: Domain) -> None:
    gen = VoronoiRegions(domain, value=stats.norm())
    a, b, c = gen.realize(3), gen.realize(3), gen.realize(4)
    assert np.array_equal(a.sites, b.sites) and np.array_equal(a.values, b.values)
    assert not np.array_equal(a.sites, c.sites)


def test_voronoi_regions_need_at_least_one_region(domain: Domain) -> None:
    with pytest.raises(ValueError, match="count must be >= 1"):
        VoronoiRegions(domain, count=0).realize(0)
