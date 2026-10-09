"""Plot registry: ``(subject, kind)`` -> plot function.

A *subject* says what is being plotted (``"sampler"``, ``"observation"``,
``"panels"``) and a *kind* says how it is drawn (``"scatter"``, ``"heatmap"``).
An object lists both: ``plot_subjects`` and ``plot_kinds``, most specific first
(``Sampler``: subject ``"sampler"``, kinds ``("scatter", "heatmap")``). A list of
observations has the subject ``"panels"``, and a latent uses its own ``kind`` as
its subject. :func:`resolve_plot` uses the first kind that has a plot for one of
the subjects, so a subclass can declare ``("fancy_stage", "stage")`` and fall
back to the generic plot.

A function that draws one kind registers for ``(subject, kind)``. A function that
works for any kind registers for ``(subject, "*")`` and receives ``kind=``. Such a
function runs something (a sampler, a stage) and passes the result on to the plot
of another subject with the same kind, so a new kind needs one new plot for
``"observation"``, not one for every subject.

The only drawing library today is plotly. Each plot function imports it itself,
when called.

The modules in this package (``latent``, ``observation``, ``components``,
``observer``, ``trace``) register plots when imported; they are all imported on
the first lookup. That is cheap: plotly is not imported until a plot runs.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from importlib import import_module
from typing import Any

from ..core.observation import Observation
from ..latent.base import Latent

PlotFn = Callable[..., Any]

MODULES = ("latent", "observation", "components", "observer", "trace")
ANY_KIND = "*"
DEFAULT_KINDS = ("scatter", "heatmap")  # for a list of observations or a latent

_PLOTS: dict[tuple[str, str], PlotFn] = {}


class NoPlotError(NotImplementedError):
    """No plot is registered for the requested subject(s) and kind(s)."""


def register_plot(subject: str, kind: str) -> Callable[[PlotFn], PlotFn]:
    """Register a plot function for ``(subject, kind)``, as a decorator.

    Parameters
    ----------
    subject : str
        What the function plots, e.g. ``"observation"``.
    kind : str
        How it draws, e.g. ``"scatter"``, or ``"*"`` for a function that works for
        any kind (it then receives ``kind=``). ``(subject, kind)`` is unique.

    Returns
    -------
    callable
        A decorator returning the function unchanged.

    Raises
    ------
    ValueError
        If ``(subject, kind)`` is already registered by a different function.
    """

    def decorator(fn: PlotFn) -> PlotFn:
        key = (subject, kind)
        existing = _PLOTS.get(key)
        if existing is not None and (existing.__module__, existing.__qualname__) != (
            fn.__module__,
            fn.__qualname__,
        ):
            raise ValueError(
                f"A plot for {key} is already registered: "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        _PLOTS[key] = fn
        return fn

    return decorator


def _load_all() -> None:
    for module in MODULES:
        import_module(f".{module}", __package__)


def _lookup(subject: str, kind: str) -> PlotFn | None:
    fn = _PLOTS.get((subject, kind))
    if fn is not None:
        return fn
    generic = _PLOTS.get((subject, ANY_KIND))
    return functools.partial(generic, kind=kind) if generic is not None else None


def _no_plot(subjects: tuple[str, ...], kinds: tuple[str, ...]) -> NoPlotError:
    known = sorted(_PLOTS)
    return NoPlotError(
        f"No plot registered for subject(s) {list(subjects)} with kind(s) "
        f"{list(kinds)}. Registered as (subject, kind): {known}"
    )


def get_plot(subject: str, kind: str) -> PlotFn:
    """Return the plot registered for ``(subject, kind)``, else for ``(subject, "*")``.

    Parameters
    ----------
    subject : str
        What the plot draws, e.g. ``"observation"``.
    kind : str
        How it draws, e.g. ``"scatter"``.

    Returns
    -------
    callable
        The plot function. A function registered for any kind comes with ``kind``
        already filled in.

    Raises
    ------
    NoPlotError
        If none is registered.
    """
    _load_all()
    fn = _lookup(subject, kind)
    if fn is None:
        raise _no_plot((subject,), (kind,))
    return fn


def registered_plots() -> dict[tuple[str, str], PlotFn]:
    """Return a copy of the registry as ``{(subject, kind): function}``.

    Does not need plotly.

    Returns
    -------
    dict
        Plot functions keyed by ``(subject, kind)``.
    """
    _load_all()
    return dict(_PLOTS)


def subjects_of(obj: object) -> tuple[str, ...]:
    """Return the plot subjects ``obj`` declares, most specific first.

    Parameters
    ----------
    obj : object
        A list of observations has the subject ``"panels"``; otherwise its
        ``plot_subjects``, or a latent's ``kind``.

    Returns
    -------
    tuple of str
        The declared subjects.

    Raises
    ------
    TypeError
        If ``obj`` declares nothing (generators must be realized first).
    ValueError
        If ``obj`` is an empty list.
    """
    if isinstance(obj, list | tuple):
        if not obj:
            raise ValueError("nothing to plot")
        if all(isinstance(o, Observation) for o in obj):
            return ("panels",)
    declared = getattr(obj, "plot_subjects", None)
    if declared:
        return tuple(declared)
    if isinstance(obj, Latent):
        return (obj.kind,)
    raise TypeError(
        f"Don't know how to plot {type(obj).__name__}: it declares no plot_subjects. "
        "(Generators must be realized first: plot(generator.realize(seed)).)"
    )


def kinds_of(obj: object) -> tuple[str, ...]:
    """Return the plot kinds ``obj`` declares, most specific first.

    Parameters
    ----------
    obj : object
        A list of observations or a latent offers ``("scatter", "heatmap")``;
        otherwise its ``plot_kinds``.

    Returns
    -------
    tuple of str
        The declared kinds. The first is the default.

    Raises
    ------
    TypeError
        If ``obj`` declares nothing (generators must be realized first).
    ValueError
        If ``obj`` is an empty list.
    """
    subjects_of(obj)  # fails for anything that cannot be plotted
    declared = getattr(obj, "plot_kinds", None)
    return tuple(declared) if declared else DEFAULT_KINDS


def resolve_plot(obj: object, kind: str | None = None) -> PlotFn:
    """Return the plot for ``obj``: the first of its kinds that has one.

    Parameters
    ----------
    obj : object
        What to plot; see :func:`subjects_of` and :func:`kinds_of`.
    kind : str, optional
        Pick this one of the declared kinds instead of the first that has a plot.

    Returns
    -------
    callable
        The plot function.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the kinds ``obj`` declares.
    NoPlotError
        If no declared (or requested) kind has a plot for any subject.
    """
    subjects, kinds = subjects_of(obj), kinds_of(obj)
    if kind is not None:
        if kind not in kinds:
            raise ValueError(
                f"{type(obj).__name__} cannot be plotted as {kind!r}; "
                f"it declares {list(kinds)}"
            )
        kinds = (kind,)
    _load_all()
    for candidate in kinds:
        for subject in subjects:
            fn = _lookup(subject, candidate)
            if fn is not None:
                return fn
    raise _no_plot(subjects, kinds)


def plot(
    obj: object,
    /,
    *args: Any,
    kind: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot ``obj`` with the function registered for its subject and kind.

    Parameters
    ----------
    obj : object
        A realized latent, an Observation, a list of observations, a Domain, a
        Sampler, a Stage, an Observer or a Trace.
    *args
        What the plot needs besides ``obj``: ``plot(sampler, latent)``,
        ``plot(stage, obs)``, ``plot(observer, latent)``.
    kind : str, optional
        How to draw it, one of the kinds the object declares (``"scatter"``,
        ``"heatmap"``, ...). By default the first that has a plot (see
        :func:`kinds_of`).
    **kwargs
        Passed to the plot function, e.g. ``seed`` or ``channel``; the rest go to
        ``fig.update_layout`` for plotly.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure for ``obj``.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the kinds ``obj`` declares.
    NoPlotError
        If no plot is registered (a ``NotImplementedError``).

    Examples
    --------
    >>> from scisynth.core import Domain
    >>> from scisynth.latent import AnalyticField
    >>> from scisynth.samplers import GridSampler
    >>> from scisynth.stages import GaussianNoise
    >>> from scisynth.viz import plot
    >>> domain = Domain.from_extents([(0, 1), (0, 1)])
    >>> latent = AnalyticField(lambda x, y: x + y, domain)
    >>> sampler = GridSampler(16)
    >>> fig = plot(sampler, latent)
    >>> fig = plot(sampler, latent, kind="heatmap")
    >>> fig = plot(GaussianNoise(0.1), sampler.run(latent, seed=0))  # before, after
    """
    return resolve_plot(obj, kind)(obj, *args, **kwargs)
