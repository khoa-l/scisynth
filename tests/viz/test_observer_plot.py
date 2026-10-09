from typing import Any

import numpy as np
import pytest

from scisynth import (
    AnalyticField,
    Clip,
    GaussianNoise,
    GridSampler,
    Multichannel,
    Observer,
    PointSampler,
    Quantize,
    RandomDropout,
)
from scisynth.viz import plot

pytest.importorskip("plotly")


def titles(fig: Any) -> list[str]:
    return [a.text for a in fig.layout.annotations]


def test_one_panel_per_step_led_by_the_latent(field: AnalyticField) -> None:
    observer = Observer(GridSampler(12), [GaussianNoise(0.1), Quantize(8)])
    fig = observer.plot(field)
    assert titles(fig) == ["latent", "sampler", "gaussian_noise", "quantize"]
    assert len(fig.data) == 4


def test_latent_panel_is_optional_and_named_steps_are_used(
    field: AnalyticField,
) -> None:
    observer = Observer(("camera", GridSampler(8)), [("noise", GaussianNoise(0.1))])
    fig = plot(observer, field, show_latent=False)
    assert titles(fig) == ["camera", "noise"]


def test_panels_are_the_traced_observations(field: AnalyticField) -> None:
    observer = Observer(GridSampler(10), [GaussianNoise(0.5), RandomDropout(0.3)])
    fig = observer.plot(field, seed=4, show_latent=False)
    trace = observer.trace(field, 4)
    for panel, obs in zip(fig.data, trace, strict=True):
        assert np.allclose(panel.marker.color, obs.values[..., 0][obs.mask])


def test_seed_makes_the_plot_repeatable(field: AnalyticField) -> None:
    observer = Observer(PointSampler(80), [GaussianNoise(0.3)])
    a, b = observer.plot(field), observer.plot(field)
    c = observer.plot(field, seed=1)
    assert np.array_equal(a.data[2].marker.color, b.data[2].marker.color)
    assert not np.array_equal(a.data[2].marker.color, c.data[2].marker.color)


def test_many_stages_wrap_into_rows_of_four(field: AnalyticField) -> None:
    observer = Observer(GridSampler(6), [GaussianNoise(0.1)] * 10)
    fig = observer.plot(field)  # latent + sampler + 10 stages = 12 panels
    assert len(fig.data) == 12
    lefts = {
        round(fig.layout["xaxis" + ("" if i == 1 else str(i))].domain[0], 3)
        for i in range(1, 13)
    }
    assert len(lefts) == 4


def test_channel_ncols_and_latent_shape_are_passed_through(
    field: AnalyticField,
) -> None:
    three = Multichannel(field, field * field, field + field)
    observer = Observer(GridSampler(6), [Clip(hi=1.0)])
    fig = observer.plot(three, channel=2, ncols=2, shape=9)
    assert len(fig.data) == 3
    assert len(fig.data[0].x) == 81  # the latent evaluated on a 9x9 grid
    assert fig.data[0].marker.color.max() == pytest.approx(
        two_channel_max(three), rel=1e-6
    )


def two_channel_max(latent: Multichannel) -> float:
    obs = GridSampler(9).run(latent)
    return float(obs.values[..., 2].max())
