from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

import scisynth as ok
from scisynth.testing.invariants import check_observation


class PCA:
    """The smallest estimator with scikit-learn's fit / transform protocol."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def get_params(self) -> dict[str, int]:
        return {"n_components": self.n_components}

    def fit(self, x: np.ndarray) -> PCA:
        self.mean_ = x.mean(axis=0)
        _, _, vt = np.linalg.svd(x - self.mean_, full_matrices=False)
        self.components_ = vt[: self.n_components]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) @ self.components_.T


class FitTransformOnly:
    def fit(self, x: np.ndarray) -> FitTransformOnly:
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return x


@pytest.fixture
def trace() -> ok.Trace:
    # four channels in two strongly correlated pairs
    domain = ok.Domain.from_extents([(0, 1), (0, 1)])
    fields = [ok.GaussianField(domain, length_scale=0.3).realize(s) for s in range(4)]
    block = [[1, 0.9, 0, 0], [0.9, 1, 0, 0], [0, 0, 1, 0.9], [0, 0, 0.9, 1]]
    mixed = ok.Multichannel(*fields, corr=block)
    observer = ok.Observer(
        ok.GridSampler(20),
        [ok.GaussianNoise(0.1), ok.RandomDropout(0.2), ok.Downsample(2)],
    )
    return observer.trace(mixed, seed=0)


def test_embeds_valid_locations_as_points(trace: ok.Trace) -> None:
    proj = ok.Projection(PCA(2)).fit(trace[0])
    out = proj.transform(trace[1])
    check_observation(out)
    assert out.layout == "points"
    assert list(out.coords) == ["c0", "c1"]
    assert out.values.shape == (400, 4)
    assert np.array_equal(out.ids, trace[1].ids.ravel())
    assert np.array_equal(out.values, trace[1].values.reshape(400, 4), equal_nan=True)
    expected = proj.estimator.transform(trace[1].values[trace[1].mask])
    assert np.allclose(np.column_stack(list(out.coords.values()))[out.mask], expected)


def test_masked_locations_have_no_position(trace: ok.Trace) -> None:
    out = ok.Projection(PCA(2)).fit(trace[0]).transform(trace["random_dropout"])
    assert not out.mask.all()
    assert all(np.isnan(c[~out.mask]).all() for c in out.coords.values())
    assert not any(np.isnan(c[out.mask]).any() for c in out.coords.values())


def test_extend_adds_the_embedding_as_a_linked_step(trace: ok.Trace) -> None:
    proj = ok.Projection(PCA(2)).fit(trace[0])
    out = proj.extend(trace, "embedding")
    assert out.names == [*trace.names, "embedding"]
    assert all(a is b for a, b in zip(trace, out[:-1], strict=True))
    check_observation(out[-1])
    assert out[-1].meta[-1]["name"] == "embedding"
    assert out[-1].meta[-1]["type"] == "PCA"
    # one to one by id: every valid location of the last step has a position
    last = out.diffs[-1]
    assert last.n_dropped == 0 and last.n_added == 0 and last.n_moved == 0
    # so any earlier location traces to its position, through Downsample's merges
    links = out.links_between(0, -1)
    assert links.shape == (out[-1].size, trace[0].size)
    assert links.nnz > 0 and np.all(out[-1].mask[links.nonzero()[0]])


def test_extend_refuses_a_taken_name(trace: ok.Trace) -> None:
    with pytest.raises(ValueError, match="already"):
        ok.Projection(PCA(2)).fit(trace[0]).extend(trace, "downsample")


def test_requires_fit(trace: ok.Trace) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        ok.Projection(PCA(2)).transform(trace[0])


def test_estimator_must_be_able_to_transform_new_data() -> None:
    with pytest.raises(TypeError, match="transform"):
        ok.Projection(FitTransformOnly())
    with pytest.raises(TypeError, match="fit"):
        ok.Projection(object())


def test_dict_round_trip_names_the_public_class_and_is_unfitted(
    trace: ok.Trace, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Estimator(PCA):
        pass

    # like sklearn.decomposition.PCA, defined in a private module of a public package
    Estimator.__module__, Estimator.__qualname__ = "fake_dr._impl", "Estimator"
    monkeypatch.setitem(
        sys.modules, "fake_dr", types.SimpleNamespace(Estimator=Estimator)
    )
    proj = ok.Projection(Estimator(3)).fit(trace[0])
    data = proj.to_dict()
    assert data["kind"] == "projection"
    assert data["estimator"]["class"] == "fake_dr.Estimator"
    again = ok.Projection.from_dict(data)
    assert isinstance(again.estimator, Estimator) and again.estimator.n_components == 3
    with pytest.raises(RuntimeError):
        again.transform(trace[0])


def test_dict_is_not_a_component_and_refuses_other_dicts(trace: ok.Trace) -> None:
    data = ok.Projection(PCA(2)).to_dict()
    with pytest.raises(KeyError, match="type"):
        ok.Stage.from_dict(data)
    with pytest.raises(ValueError, match="Not a serialized Projection"):
        ok.Projection.from_dict(ok.GaussianNoise(0.1).to_dict())


def test_dict_refuses_parameters_that_cannot_be_serialized() -> None:
    class Estimator(PCA):
        def get_params(self, deep: bool = True) -> dict[str, Any]:
            return {"n_components": 2, "helper": len}

    with pytest.raises(TypeError, match="Cannot serialize"):
        ok.Projection(Estimator(2)).to_dict()


def test_transform_refuses_a_different_number_of_channels(trace: ok.Trace) -> None:
    proj = ok.Projection(PCA(2)).fit(trace[0])
    one_channel = trace[0].replace(
        values=trace[0].values[..., :1],
        truth=None,
    )
    with pytest.raises(ValueError, match=r"fitted on 4 channel.*has 1"):
        proj.transform(one_channel)
    with pytest.raises(ValueError, match="fitted on 4 channel"):
        proj.extend(
            ok.Trace(
                [
                    one_channel,
                    one_channel.replace(
                        meta=[{**one_channel.meta[-1], "name": "next"}]
                    ),
                ]
            )
        )


def test_extended_trace_plots_the_embedding_in_its_own_frame(trace: ok.Trace) -> None:
    pytest.importorskip("plotly")
    out = ok.Projection(PCA(2)).fit(trace[0]).extend(trace)
    fig = out.plot()
    frames = [t for t in fig.data if t.name == "axes"]
    assert [(t.z[0], list(t.meta)) for t in frames] == [
        (0.0, ["x", "y"]),
        (4.0, ["c0", "c1"]),
    ]
    by_name = {t.name: t for t in fig.data}
    assert by_name["project line"].line.color != by_name["stayed line"].line.color


def test_plot_samples_across_the_range_of_the_embedding(trace: ok.Trace) -> None:
    pytest.importorskip("plotly")
    out = ok.Projection(PCA(2)).fit(trace[0]).extend(trace)
    last = out[-1]
    full = np.ptp(last.coords["c0"][last.mask])
    kept = {t.name: t for t in out.plot(max_points=12).data}["kept"]
    drawn = kept.customdata[kept.z == len(out) - 1][:, 0]
    assert np.isclose(np.ptp(drawn), full)  # the extremes are drawn


def test_extend_needs_no_get_params(trace: ok.Trace) -> None:
    class Bare:
        def fit(self, x: np.ndarray) -> Bare:
            return self

        def transform(self, x: np.ndarray) -> np.ndarray:
            return x[:, :2]

    out = ok.Projection(Bare()).fit(trace[0]).extend(trace)
    assert out[-1].meta[-1]["params"] == {} and out[-1].meta[-1]["type"] == "Bare"


def test_a_projected_trace_plots_in_value_space_too(trace: ok.Trace) -> None:
    pytest.importorskip("plotly")
    out = ok.Projection(PCA(2)).fit(trace[0]).extend(trace)
    names = {t.name for t in out.plot().data}
    assert "project line" in names  # the embedding has other axes than the layers below
    fig = out.plot(axes=(0, 1))  # channels exist in every layer, the embedding too
    assert "project line" not in {t.name for t in fig.data}
    first = next(t for t in fig.data if t.name == "axes")
    assert (first.z[0], list(first.meta)) == (0.0, ["ch0", "ch1"])


def test_per_step_axes_chain_channels_into_the_embedding(trace: ok.Trace) -> None:
    pytest.importorskip("plotly")
    out = ok.Projection(PCA(2)).fit(trace[0]).extend(trace)
    fig = out.plot(axes=[(0, 1)] * (len(out) - 1) + [None])
    frames = [(t.z[0], list(t.meta)) for t in fig.data if t.name == "axes"]
    assert frames[0] == (0.0, ["ch0", "ch1"])
    assert frames[-1] == (float(len(out) - 1), ["c0", "c1"])
    assert "project line" in {t.name for t in fig.data}
    with pytest.raises(ValueError, match="one entry per step"):
        out.plot(axes=[(0, 1), None])
