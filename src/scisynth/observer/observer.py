"""Observer: one sampler followed by a chain of named stages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from ..core.observation import Observation
from ..core.rng import SeedLike, spawn_rngs
from ..latent.base import Latent
from ..samplers.base import Sampler
from ..stages.base import Stage
from ._names import check_name
from .stage_list import StageList, StageSpec
from .trace import Trace

SAMPLER_NAME = "sampler"

SamplerSpec = Sampler | tuple[str, Sampler]


class Observer:
    """A sampler followed by a chain of named stages: a simulated instrument.

    Parameters
    ----------
    sampler : Sampler or (str, Sampler)
        Queries the latent. Named ``"sampler"`` unless given as ``(name, sampler)``.
    stages : iterable of Stage or (str, Stage), or StageList, default=()
        Run in order on the sampler's output. Bare stages are named after their
        class (see :class:`~scisynth.observer.stage_list.StageList`).

    Attributes
    ----------
    sampler : Sampler
        The sampler.
    stages : StageList
        The stages, in run order. Index or slice it by position or name:
        ``observer.stages[0]``, ``observer.stages["quantize"]``,
        ``observer.stages[:2]``.
    sampler_name : str
        The sampler's name.
    stage_names : list of str
        The stages' names, in run order.
    names : list of str
        All step names in run order, the sampler first.

    Raises
    ------
    TypeError
        If ``sampler`` is not a Sampler or a stage is not a Stage.
    ValueError
        If a name is empty, contains ``"__"`` (reserved) or is repeated.

    Notes
    -----
    Names are recorded in each ``Observation.meta`` entry.

    The sampler gets random stream 0 and stage ``i`` gets stream ``i + 1``, all
    derived from the run seed. An observer built from the first ``k`` stages
    reproduces those steps exactly; one that skips earlier stages shifts the streams
    of the stages it keeps. Use :meth:`with_stages` to build one from a slice.

    Examples
    --------
    >>> from scisynth.core import Domain
    >>> from scisynth.latent import AnalyticField
    >>> from scisynth.observer import Observer
    >>> from scisynth.samplers import GridSampler
    >>> from scisynth.stages import GaussianNoise, Quantize
    >>> domain = Domain.from_extents([(0, 1), (0, 1)])
    >>> latent = AnalyticField(lambda x, y: x + y, domain)
    >>> observer = Observer(GridSampler(8), [GaussianNoise(0.1), Quantize(16)])
    >>> observer.names
    ['sampler', 'gaussian_noise', 'quantize']
    >>> observer.stages["quantize"]
    Quantize(levels=16, vmin=None, vmax=None)
    >>> observer.with_stages(observer.stages[:1]).names
    ['sampler', 'gaussian_noise']
    >>> observer.run(latent, seed=0).values.shape
    (8, 8, 1)
    """

    plot_subjects: ClassVar[tuple[str, ...]] = ("observer",)
    plot_kinds: ClassVar[tuple[str, ...]] = ("scatter", "heatmap", "layers")

    sampler: Sampler
    stages: StageList
    sampler_name: str

    def __init__(
        self, sampler: SamplerSpec, stages: Iterable[StageSpec] | StageList = ()
    ) -> None:
        sampler_name: str | None = None
        if isinstance(sampler, tuple):
            sampler_name, sampler = sampler
            check_name(sampler_name)
        if not isinstance(sampler, Sampler):
            raise TypeError(f"sampler must be a Sampler, got {type(sampler).__name__}")
        self.sampler = sampler
        self.sampler_name = sampler_name or SAMPLER_NAME
        self.stages = StageList(stages, reserved=[self.sampler_name])

    @property
    def stage_names(self) -> list[str]:
        return self.stages.names

    @property
    def names(self) -> list[str]:
        return [self.sampler_name, *self.stage_names]

    def with_stages(self, stages: Iterable[StageSpec] | StageList) -> Observer:
        """Return an observer with the same sampler and different stages.

        Parameters
        ----------
        stages : iterable of Stage or (str, Stage), or StageList
            The new stages. A slice of ``observer.stages`` keeps its names; to extend
            an observer use ``[*observer.stages.items(), new_stage]``.

        Returns
        -------
        Observer
            A new observer; this one is unchanged.
        """
        return Observer((self.sampler_name, self.sampler), stages)

    def trace(self, latent: Latent, seed: SeedLike = None) -> Trace:
        """Run the pipeline and keep every intermediate observation.

        Parameters
        ----------
        latent : Latent
            The ground truth to observe.
        seed : int, SeedSequence or None, default=None
            Seed for all random streams; None draws fresh OS entropy.

        Returns
        -------
        Trace
            The output after the sampler and after each stage, so
            ``len(stages) + 1`` observations. Each one's last ``meta`` entry carries
            that step's name.
        """
        rngs = spawn_rngs(seed, 1 + len(self.stages))
        obs = _named(self.sampler(latent, rngs[0]), self.sampler_name)
        out, links = [obs], []
        for (name, stage), rng in zip(self.stages.items(), rngs[1:], strict=True):
            after, params = stage.call_with_params(obs, rng)
            after = _named(after, name)
            links.append(stage.links_from(obs, after, params))
            out.append(obs := after)
        return Trace(out, links)

    def run(self, latent: Latent, seed: SeedLike = None) -> Observation:
        """Run the pipeline and return the final observation.

        Parameters
        ----------
        latent : Latent
            The ground truth to observe.
        seed : int, SeedSequence or None, default=None
            Seed for all random streams; None draws fresh OS entropy.

        Returns
        -------
        Observation
            The output of the last stage.
        """
        return self.trace(latent, seed)[-1]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Observer):
            return NotImplemented
        return (
            self.sampler == other.sampler
            and self.sampler_name == other.sampler_name
            and self.stages == other.stages
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        steps = ", ".join(f"({n!r}, {s!r})" for n, s in self.stages.items())
        return f"Observer(({self.sampler_name!r}, {self.sampler!r}), [{steps}])"

    def plot(
        self,
        latent: Latent,
        seed: SeedLike = 0,
        *,
        kind: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run on ``latent`` and plot every step's output.

        Parameters
        ----------
        latent : Latent
            The ground truth to observe.
        seed : int, SeedSequence or None, default=0
            Seed for the random streams; the default keeps plots repeatable.
        kind : str, optional
            How to draw it: ``"scatter"`` (the default) or ``"heatmap"`` for one
            panel per step, or ``"layers"`` for the 3-D plot of the trace.
        **kwargs
            Passed to the plot function (for plotly, ``fig.update_layout``).

        Returns
        -------
        plotly.graph_objects.Figure
            One panel per step; with ``kind="layers"``, the 3-D plot of the trace
            (see :meth:`Trace.plot <scisynth.observer.trace.Trace.plot>`).
        """
        from ..viz import plot

        return plot(self, latent, seed=seed, kind=kind, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the sampler and stages (not the latent) with their names.

        Returns
        -------
        dict
            ``{"sampler": {...}, "stages": [...]}``, each entry holding a ``name`` and
            a ``component`` dict.
        """
        return {
            "sampler": {"name": self.sampler_name, "component": self.sampler.to_dict()},
            "stages": [
                {"name": n, "component": s.to_dict()} for n, s in self.stages.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Observer:
        """Rebuild an observer from :meth:`to_dict` output.

        Parameters
        ----------
        data : mapping
            Names are optional; unnamed steps are named as in the constructor.
            Bare component dicts are accepted in place of ``{"name", "component"}``.

        Returns
        -------
        Observer
            The rebuilt observer.
        """

        def entry(raw: Mapping[str, Any]) -> tuple[str | None, Mapping[str, Any]]:
            if "component" in raw:
                return raw.get("name"), raw["component"]
            return None, raw  # a bare component dict

        sampler_name, sampler_data = entry(data["sampler"])
        sampler = Sampler.from_dict(sampler_data)
        stages: list[StageSpec] = []
        for raw in data.get("stages", []):
            name, stage_data = entry(raw)
            stage = Stage.from_dict(stage_data)
            stages.append(stage if name is None else (name, stage))
        return cls(sampler if sampler_name is None else (sampler_name, sampler), stages)


def _named(obs: Observation, name: str) -> Observation:
    *earlier, last = obs.meta
    return obs.replace(meta=[*earlier, {**last, "name": name}])
