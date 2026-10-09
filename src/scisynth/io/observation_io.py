"""Save and load Observations as ``.npz`` (stubs)."""

from __future__ import annotations

from pathlib import Path

from ..core.observation import Observation


def save_observation(obs: Observation, path: str | Path) -> None:
    """Write ``obs`` to an ``.npz`` file. Not implemented yet."""
    raise NotImplementedError


def load_observation(path: str | Path) -> Observation:
    """Read an Observation written by :func:`save_observation`. Not implemented yet."""
    raise NotImplementedError
