"""Shared base for samplers, stages, latent generators and domains."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, TypeVar

import numpy as np

from . import registry
from ._plain import to_plain
from .params import resolve

C = TypeVar("C", bound="Component")


class Component:
    """Base for everything configurable: samplers, stages, generators, domains.

    Subclasses are expected to be dataclasses; their init fields are the parameters.
    Each subclass registers itself by class name when defined, so that
    :meth:`from_dict` can rebuild it (its module must have been imported). A base
    class that is never built directly opts out with ``register=False`` in its
    class statement; the keyword is not inherited by its subclasses.
    """

    def __init_subclass__(cls, register: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if register:
            registry.register(cls)

    def __post_init__(self) -> None:
        for name, value in self.params().items():
            if isinstance(value, list):  # sequences are kept as tuples
                setattr(self, name, tuple(value))

    def params(self) -> dict[str, Any]:
        """Return the raw (unresolved) parameters, in field order.

        Returns
        -------
        dict
            Parameter names and their values as given.
        """
        if not dataclasses.is_dataclass(self):
            return {}
        return {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.init
        }

    def resolved_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Resolve every parameter to a concrete value, drawing from ``rng``.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of randomness for distributions and callables.

        Returns
        -------
        dict
            Parameters after :func:`~scisynth.core.params.resolve`; nested components
            are left as they are.
        """
        return {
            k: v if isinstance(v, Component) else resolve(v, rng)
            for k, v in self.params().items()
        }

    def log_entry(self, resolved: Mapping[str, Any]) -> dict[str, Any]:
        """Return the provenance record appended to ``Observation.meta``.

        Parameters
        ----------
        resolved : mapping
            The resolved parameters used for this call.

        Returns
        -------
        dict
            ``{"name", "type", "params"}``. ``name`` is the class name here; an
            :class:`~scisynth.observer.observer.Observer` replaces it with the
            step name.
        """
        cls = type(self).__name__
        return {"name": cls, "type": cls, "params": to_plain(resolved, strict=False)}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain Python.

        Returns
        -------
        dict
            ``{"type": class name, "params": {...}}``.

        Raises
        ------
        TypeError
            If a parameter is a callable or distribution; only numbers, strings,
            arrays, sequences and components are supported.
        """
        return {
            "type": type(self).__name__,
            "params": {k: to_plain(v, strict=True) for k, v in self.params().items()},
        }

    @classmethod
    def from_dict(cls: type[C], data: Mapping[str, Any]) -> C:
        """Rebuild a component from :meth:`to_dict` output.

        Parameters
        ----------
        data : mapping
            ``{"type": class name, "params": {...}}``.

        Returns
        -------
        Component
            An instance of the registered class, which must be a subclass of ``cls``.

        Raises
        ------
        KeyError
            If the class name is not registered.
        TypeError
            If the class is not a subclass of ``cls``.
        """
        target = registry.get(data["type"])
        if not issubclass(target, cls):
            raise TypeError(f"{data['type']} is not a {cls.__name__}")
        params = {k: _decode(v) for k, v in data.get("params", {}).items()}
        return target(**params)


def _decode(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"type", "params"}:
            return Component.from_dict(value)
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value
