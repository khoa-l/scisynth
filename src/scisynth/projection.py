"""Dimensionality reduction of observations, through any scikit-learn style estimator.

A projection is a view of the data, not a pipeline step, so it lives outside the
``Observer``. The estimator (``sklearn.decomposition.PCA``, ``umap.UMAP``, ...) does the
reduction. :class:`Projection` only passes values in and embedding coordinates out.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from .core._plain import to_plain
from .core._types import FloatArray
from .core.observation import Observation
from .observer.trace import Trace


class Projection:
    """Place the locations of observations in a low-dimensional embedding.

    Each valid location is one sample whose features are its ``d`` channel values.
    The estimator is fitted once, on a reference observation, and then applied to any
    observation with the same channels.

    Parameters
    ----------
    estimator : object
        Anything with scikit-learn's ``fit(X)`` and ``transform(X)``, where ``X`` has
        shape ``(n_samples, d)``. Set its ``random_state`` yourself: the estimator
        owns its randomness.

    Raises
    ------
    TypeError
        If the estimator has no ``fit`` or no ``transform``. Estimators that only
        offer ``fit_transform``, like ``sklearn.manifold.TSNE``, cannot be applied to
        other observations and are not supported.

    Notes
    -----
    The output is a points observation: coordinates ``c0, c1, ...`` hold the
    embedding, while values, ``truth``, mask, ids and ``meta`` are unchanged. Masked
    locations have no position, so their coordinates are NaN. :meth:`transform` adds
    no ``meta`` record; :meth:`extend`, which adds a step to a trace, does.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> from scisynth.core import Observation
    >>> from scisynth.projection import Projection
    >>> rng = np.random.default_rng(0)
    >>> obs = Observation(
    ...     rng.normal(size=(50, 4)), {"x": np.arange(50.0)}, np.ones(50, dtype=bool)
    ... )
    >>> embedded = Projection(PCA(n_components=2)).fit(obs).transform(obs)
    >>> list(embedded.coords), embedded.values.shape
    (['c0', 'c1'], (50, 4))
    """

    def __init__(self, estimator: Any) -> None:
        for method in ("fit", "transform"):
            if not callable(getattr(estimator, method, None)):
                raise TypeError(
                    f"{type(estimator).__name__} has no {method}(); a projection "
                    "needs an estimator that can embed observations it was not "
                    "fitted on"
                )
        self.estimator = estimator
        self._n_channels: int | None = None  # set by fit

    def fit(self, obs: Observation) -> Projection:
        """Fit the estimator on the valid locations of ``obs``.

        Parameters
        ----------
        obs : Observation
            The reference observation. Only ``values`` and ``mask`` are used.

        Returns
        -------
        Projection
            ``self``, fitted in place.
        """
        self.estimator.fit(_features(obs))
        self._n_channels = obs.n_channels
        return self

    def transform(self, obs: Observation) -> Observation:
        """Embed the locations of ``obs``.

        Parameters
        ----------
        obs : Observation
            Any layout (a grid is flattened to points), with as many channels as the
            observation the projection was fitted on.

        Returns
        -------
        Observation
            Points with coordinates ``c0, c1, ...``; see the class notes.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If ``obs`` has a different number of channels than the observation the
            projection was fitted on.
        """
        if self._n_channels is None:
            raise RuntimeError("Projection is not fitted; call fit(obs) first")
        if obs.n_channels != self._n_channels:
            raise ValueError(
                f"This projection was fitted on {self._n_channels} channel(s), but "
                f"the observation has {obs.n_channels}"
            )
        obs = obs.to_points()
        features = _features(obs)
        embedded = np.asarray(self.estimator.transform(features), dtype=np.float64)
        out: FloatArray = np.full((obs.size, embedded.shape[1]), np.nan)
        out[obs.mask] = embedded
        return obs.replace(coords={f"c{i}": col for i, col in enumerate(out.T)})

    def extend(self, trace: Trace, name: str = "pca") -> Trace:
        """Add the embedding of the last step to a trace, as one more step.

        Parameters
        ----------
        trace : Trace
            The run to extend; its last step is embedded.
        name : str, default="pca"
            The new step's name.

        Returns
        -------
        Trace
            ``trace`` plus the embedded last step, linked to it by ids (one to one;
            locations without a position are not linked). So
            :meth:`Trace.links_between <scisynth.observer.trace.Trace.links_between>`
            says where every location of an earlier step ended up in the embedding, and
            the 3-D trace plot draws a line to it.

        Raises
        ------
        ValueError
            If ``name`` is already a step of ``trace``, or the last step has a different
            number of channels than the observation the projection was fitted on.
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if name in trace.names:
            raise ValueError(f"trace already has a step named {name!r}")
        last = self.transform(trace[-1])
        record = {
            "name": name,
            "type": type(self.estimator).__name__,
            "params": to_plain(_params(self.estimator), strict=False),
        }
        return Trace(
            [*trace, last.replace(meta=[*last.meta, record])],
            [*(diff.links for diff in trace.diffs), None],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the estimator's class and parameters, not its fitted state.

        Returns
        -------
        dict
            ``{"kind": "projection", "estimator": {"class": dotted path, "params":
            ...}}``. The path leaves out private modules, so it is the one you import
            from (``sklearn.decomposition.PCA``). The shape differs from
            :meth:`Component.to_dict <scisynth.core.component.Component.to_dict>` on
            purpose: a projection is not a component and cannot be loaded as one.

        Raises
        ------
        TypeError
            If an estimator parameter cannot be serialized (a callable or another
            estimator, say).
        """
        cls = type(self.estimator)
        public = [m for m in cls.__module__.split(".") if not m.startswith("_")]
        spec = {
            "class": ".".join([*public, cls.__qualname__]),
            "params": to_plain(_params(self.estimator), strict=True),
        }
        return {"kind": "projection", "estimator": spec}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Projection:
        """Rebuild an unfitted projection from :meth:`to_dict` output.

        The named class is imported and instantiated, so only load trusted data.

        Parameters
        ----------
        data : mapping
            The output of :meth:`to_dict`.

        Returns
        -------
        Projection
            A new projection around a fresh estimator.

        Raises
        ------
        ValueError
            If ``data`` was not made by :meth:`to_dict`.
        """
        if data.get("kind") != "projection":
            raise ValueError(
                f"Not a serialized Projection (expected kind 'projection'): {data!r}"
            )
        spec = data["estimator"]
        module, _, name = spec["class"].rpartition(".")
        estimator = getattr(importlib.import_module(module), name)(**spec["params"])
        return cls(estimator)

    def __repr__(self) -> str:
        return f"Projection({self.estimator!r})"


def _params(estimator: Any) -> dict[str, Any]:
    """Return the estimator's ``get_params()``, or nothing if it has none."""
    get = getattr(estimator, "get_params", None)
    return dict(get()) if callable(get) else {}


def _features(obs: Observation) -> FloatArray:
    """Return the ``(n_valid, d)`` channel values of the valid locations."""
    return obs.values[obs.mask]
