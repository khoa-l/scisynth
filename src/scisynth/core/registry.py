"""Registry of component classes by name, used by ``to_dict`` and ``from_dict``.

``Component`` subclasses register themselves automatically (by class name) when
they are defined, so a module must be imported before its classes can be
deserialized.
"""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T", bound=type)

_REGISTRY: dict[str, type[Any]] = {}


def register(cls: T) -> T:
    """Register ``cls`` under its class name.

    Parameters
    ----------
    cls : type
        The class to register.

    Returns
    -------
    type
        ``cls`` unchanged, so this can be used as a decorator.

    Raises
    ------
    ValueError
        If the class name is already registered by a different class.
    """
    key = cls.__name__
    existing = _REGISTRY.get(key)
    if existing is not None and (existing.__module__, existing.__qualname__) != (
        cls.__module__,
        cls.__qualname__,
    ):
        raise ValueError(
            f"Component name {key!r} is already registered by "
            f"{existing.__module__}.{existing.__qualname__}"
        )
    _REGISTRY[key] = cls
    return cls


def get(name: str) -> type[Any]:
    """Return the class registered under ``name``.

    Parameters
    ----------
    name : str
        The registered class name.

    Returns
    -------
    type
        The registered class.

    Raises
    ------
    KeyError
        If nothing is registered under ``name``.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown component {name!r}. Is the module that defines it imported? "
            f"Known: {sorted(_REGISTRY)}"
        ) from None
