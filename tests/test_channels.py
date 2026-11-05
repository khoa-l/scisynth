"""Multi-channel observations: samplers and stages treat channels uniformly."""

import numpy as np
import pytest

from scisynth import (
    AnalyticField,
    Clip,
    Domain,
    Drift,
    Gain,
    GaussianNoise,
    GridSampler,
    PointSampler,
    PoissonNoise,
    Quantize,
    RandomDropout,
    Saturate,
    ValueOffset,
)
from scisynth.latent.compose import Multichannel
from scisynth.testing.invariants import check_observation


@pytest.fixture
def three(domain: Domain) -> Multichannel:
    return Multichannel(
        AnalyticField(lambda x, y: x, domain),
        AnalyticField(lambda x, y: 10 * y, domain),
        AnalyticField(lambda x, y: x + y, domain),
    )


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_samplers_keep_every_channel(three: Multichannel) -> None:
    grid = GridSampler((5, 4))(three, rng())
    check_observation(grid)
    assert grid.values.shape == (5, 4, 3) and grid.mask.shape == (5, 4)
    assert grid.truth is not None and grid.truth.shape == grid.values.shape
    x, y = np.meshgrid(grid.coords["x"], grid.coords["y"], indexing="ij")
    assert np.allclose(grid.values[..., 1], 10 * y)
    assert np.allclose(grid.values[..., 2], x + y)

    pts = PointSampler(20)(three, rng())
    check_observation(pts)
    assert pts.values.shape == (20, 3) and pts.layout == "points"
    assert np.allclose(pts.values[:, 0], pts.coords["x"])


def test_elementwise_stages_accept_per_channel_parameters(three: Multichannel) -> None:
    obs = GridSampler((4, 4))(three, rng())
    gained = Gain(np.array([1.0, 2.0, 3.0]))(obs, rng())
    assert np.allclose(gained.values, obs.values * [1.0, 2.0, 3.0])
    shifted = ValueOffset(np.array([0.0, 1.0, 2.0]))(obs, rng())
    assert np.allclose(shifted.values - obs.values, [0.0, 1.0, 2.0])
    clipped = Clip(hi=np.array([0.5, 1.0, 5.0]))(obs, rng())
    assert np.all(clipped.values <= [0.5, 1.0, 5.0])
    assert np.all(Saturate(0.1)(obs, rng()).values <= 0.1)
    drifted = Drift(np.array([0.0, 1.0, 2.0]), axis="x")(obs, rng())
    x = obs.coords["x"][:, None, None]
    assert np.allclose(drifted.values - obs.values, (x - x.min()) * [0.0, 1.0, 2.0])


def test_noise_is_drawn_per_channel(three: Multichannel) -> None:
    obs = GridSampler((30, 30))(three, rng())
    out = GaussianNoise(sigma=np.array([0.0, 1.0, 5.0]))(obs, rng())
    resid = (out.values - obs.values).reshape(-1, 3)
    assert np.allclose(resid[:, 0], 0.0)
    assert resid[:, 1].std() == pytest.approx(1.0, rel=0.1)
    assert resid[:, 2].std() == pytest.approx(5.0, rel=0.1)
    check_observation(out)
    scale = np.array([1.0, 10.0, 100.0])
    shot = PoissonNoise(scale)(obs, rng())
    variance = ((shot.values - obs.values) ** 2).reshape(-1, 3).mean(axis=0)
    expected = obs.values.reshape(-1, 3).mean(axis=0) / scale
    assert variance == pytest.approx(expected, rel=0.15)


def test_dropout_masks_every_channel_of_a_location(three: Multichannel) -> None:
    obs = PointSampler(500)(three, rng())
    out = RandomDropout(0.4)(obs, rng(1))
    check_observation(out)
    assert out.mask.shape == (500,)
    assert np.isnan(out.values[~out.mask]).all()  # all channels of a dropped point
    assert not np.isnan(out.values[out.mask]).any()


def test_quantize_defaults_to_each_channels_own_range(three: Multichannel) -> None:
    obs = GridSampler((20, 20))(three, rng())
    out = Quantize(levels=3)(obs, rng())
    for c, hi in enumerate([1.0, 20.0, 3.0]):
        assert np.allclose(np.unique(out.values[..., c]), [0.0, hi / 2, hi])
    explicit = Quantize(levels=2, vmin=np.array([0, 0, 0]), vmax=np.array([1, 1, 1]))
    assert set(np.unique(explicit(obs, rng()).values)) == {0.0, 1.0}
