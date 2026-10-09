"""Config-file helpers for observers, in YAML or JSON (stubs).

They will wrap ``Observer.to_dict`` and ``Observer.from_dict``.
"""

from __future__ import annotations

from pathlib import Path

from ..observer.observer import Observer


def dump_config(observer: Observer, path: str | Path) -> None:
    """Write an observer's config to ``path``. Not implemented yet."""
    raise NotImplementedError


def load_config(path: str | Path) -> Observer:
    """Read an observer from a config file. Not implemented yet."""
    raise NotImplementedError
