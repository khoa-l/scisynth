import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scisynth import (
    AnalyticField,
    Domain,
    GaussianNoise,
    GridSampler,
    Observation,
    Observer,
    PointSampler,
)
from scisynth.viz import (
    NoPlotError,
    get_plot,
    kinds_of,
    plot,
    plot_latent,
    register_plot,
    registered_plots,
    resolve_plot,
    subjects_of,
)
from scisynth.viz._data import require

SRC = Path(__file__).resolve().parents[2] / "src"


def run_isolated(code: str) -> None:
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )


def grid_obs(field: AnalyticField) -> Observation:
    return GridSampler((6, 6))(field, np.random.default_rng(0))


def test_importing_viz_and_loading_every_module_needs_no_plot_library() -> None:
    run_isolated(
        "import sys\n"
        "import scisynth\n"
        "assert 'scisynth.viz' not in sys.modules\n"
        "import scisynth.viz as viz\n"
        "viz.registered_plots()\n"
        "assert 'plotly' not in sys.modules"
    )


def test_expected_registrations() -> None:
    keys = set(registered_plots())
    for key in [
        ("observation", "scatter"),
        ("observation", "heatmap"),
        ("trace", "layers"),
        ("observer", "layers"),
        ("domain", "outline"),
    ]:
        assert key in keys
    # these work for any kind and receive it
    for subject in ["panels", "sampler", "stage", "observer", "trace", "field"]:
        assert (subject, "*") in keys


def test_subjects_and_kinds_of(field: AnalyticField, domain: Domain) -> None:
    obs = grid_obs(field)
    points = PointSampler(5)(field, np.random.default_rng(0))
    both = ("scatter", "heatmap")
    for obj, subject, kinds in [
        (obs, "observation", both),
        (points, "observation", both),
        ([obs, obs], "panels", both),
        (domain, "domain", ("outline",)),
        (GridSampler(3), "sampler", both),
        (GaussianNoise(), "stage", both),
        (Observer(GridSampler(3)), "observer", ("scatter", "heatmap", "layers")),
        (field, "field", both),  # a latent's own kind is its subject
    ]:
        assert subjects_of(obj) == (subject,)
        assert kinds_of(obj) == kinds
    with pytest.raises(TypeError, match="plot_subjects"):
        subjects_of(object())
    with pytest.raises(ValueError, match="nothing"):
        kinds_of([])


def test_every_declared_subject_and_kind_has_a_plot(
    field: AnalyticField, domain: Domain
) -> None:
    objects: list[object] = [
        GridSampler(3),
        GaussianNoise(),
        domain,
        Observer(GridSampler(3)),
        grid_obs(field),
        PointSampler(3)(field, np.random.default_rng(0)),
        [grid_obs(field)],
        field,
    ]
    for obj in objects:
        for kind in kinds_of(obj):
            assert callable(resolve_plot(obj, kind)), (type(obj), kind)


def test_unregistered_subject(catalog: object) -> None:
    with pytest.raises(NoPlotError, match="catalog"):
        plot(catalog)
    with pytest.raises(NoPlotError, match="nothing"):
        get_plot("nothing", "scatter")


def test_a_plot_for_any_kind_receives_the_kind() -> None:
    fn = get_plot("stage", "heatmap")
    assert fn.keywords == {"kind": "heatmap"}  # type: ignore[attr-defined]
    # an exact registration is returned as it is
    assert get_plot("observation", "heatmap").__name__ == "plotly_heatmap"


def test_register_rejects_duplicates() -> None:
    def other(*args: Any) -> None:
        return None

    with pytest.raises(ValueError, match="already registered"):
        register_plot("observation", "scatter")(other)


def test_subclass_declares_specific_subject_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, field: AnalyticField
) -> None:
    from dataclasses import dataclass
    from typing import ClassVar

    from scisynth import Stage
    from scisynth.viz import registry

    monkeypatch.setattr(registry, "_PLOTS", dict(registered_plots()))

    @dataclass
    class FancyStage(GaussianNoise):
        plot_subjects: ClassVar[tuple[str, ...]] = ("fancy_stage", "stage")

    @dataclass
    class PlainStage(Stage):
        pass

    assert PlainStage.plot_subjects == ("stage",)  # inherited
    # Without a plot for "fancy_stage" it falls back to the generic stage plot.
    assert resolve_plot(FancyStage()).keywords == {"kind": "scatter"}  # type: ignore[attr-defined]

    @register_plot("fancy_stage", "scatter")
    def fancy(stage: Stage, obs: Observation) -> str:
        return "fancy"

    assert resolve_plot(FancyStage()) is fancy
    # Only for the kinds it was registered for, and not for other stages.
    assert resolve_plot(FancyStage(), "heatmap").keywords == {"kind": "heatmap"}  # type: ignore[attr-defined]
    assert resolve_plot(GaussianNoise()) is not fancy


def test_unimplemented_plots_are_stubs(field: AnalyticField, domain: Domain) -> None:
    calls: list[Callable[[], Any]] = [
        lambda: plot(domain),
        lambda: plot(field),
        lambda: plot_latent(field),
        lambda: field.plot(),
    ]
    for call in calls:
        with pytest.raises(NotImplementedError) as exc:
            call()
        assert not isinstance(exc.value, NoPlotError)  # found, just not implemented


def test_require_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    def fail(name: str, *a: Any) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(ImportError, match=r"scisynth\[viz\]"):
        require("plotly.graph_objects")


def test_viz_extra_declared() -> None:
    import tomllib

    data = tomllib.loads((SRC.parent / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert {d.split(">")[0] for d in extras["viz"]} == {"plotly"}
    assert not any("plotly" in d for d in data["project"]["dependencies"])


def test_kind_picks_one_of_the_declared_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import ClassVar

    from scisynth.viz import registry

    monkeypatch.setattr(registry, "_PLOTS", dict(registered_plots()))

    class Thing:
        plot_subjects: ClassVar[tuple[str, ...]] = ("thing",)
        plot_kinds: ClassVar[tuple[str, ...]] = ("first", "second", "unplotted")

    @register_plot("thing", "first")
    def first(thing: Thing) -> str:
        return "first"

    @register_plot("thing", "second")
    def second(thing: Thing) -> str:
        return "second"

    thing = Thing()
    assert plot(thing) == "first"  # the default is the first with a plot
    assert plot(thing, kind="first") == "first"
    assert plot(thing, kind="second") == "second"
    assert resolve_plot(thing, kind="second") is second
    with pytest.raises(ValueError, match=r"cannot be plotted as 'nope'.*'first'"):
        plot(thing, kind="nope")
    with pytest.raises(NoPlotError):  # declared, but nothing registered
        plot(thing, kind="unplotted")


def test_kind_is_forwarded_by_every_plot_method(
    monkeypatch: pytest.MonkeyPatch, field: AnalyticField, domain: Domain
) -> None:
    from dataclasses import dataclass
    from typing import ClassVar

    from scisynth import Trace
    from scisynth.viz import registry

    monkeypatch.setattr(registry, "_PLOTS", dict(registered_plots()))
    both = ("scatter", "alt")

    @dataclass
    class AltField(AnalyticField):
        plot_kinds: ClassVar[tuple[str, ...]] = both

    @dataclass
    class AltSampler(GridSampler):
        plot_kinds: ClassVar[tuple[str, ...]] = both

    @dataclass
    class AltStage(GaussianNoise):
        plot_kinds: ClassVar[tuple[str, ...]] = both

    class AltObserver(Observer):
        plot_kinds: ClassVar[tuple[str, ...]] = both

    class AltTrace(Trace):
        plot_kinds: ClassVar[tuple[str, ...]] = both

    def alt(*args: Any, **kwargs: Any) -> str:
        return "alt"

    for subject in ["field", "sampler", "stage", "observer", "trace"]:
        register_plot(subject, "alt")(alt)

    sampler, stage = AltSampler(4), AltStage(0.1)
    observer = AltObserver(sampler, [stage])
    alt_field = AltField(field.func, domain)
    assert sampler.plot(alt_field, kind="alt") == "alt"
    assert stage.preview(sampler.run(alt_field, 0), kind="alt") == "alt"
    assert observer.plot(alt_field, kind="alt") == "alt"
    assert AltTrace(observer.trace(alt_field, 0)).plot(kind="alt") == "alt"
    assert alt_field.plot(kind="alt") == "alt"
    assert plot_latent(alt_field, kind="alt") == "alt"
    calls: list[Callable[[], Any]] = [
        lambda: sampler.plot(alt_field, kind="nope"),
        lambda: stage.preview(sampler.run(alt_field, 0), kind="nope"),
        lambda: observer.plot(alt_field, kind="nope"),
        lambda: alt_field.plot(kind="nope"),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="cannot be plotted as 'nope'"):
            call()
