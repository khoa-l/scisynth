"""Core building blocks: observations, domains, components, RNG, parameters.

This package depends on nothing else in scisynth, and everything else depends on
it, so it must stay that way: importing it never needs another subpackage.

Its public names are importable from the package itself, for example
``from scisynth.core import Domain, Observation``.
"""

from .domain import Axis, Domain
from .observation import Observation
from .params import resolve
from .rng import spawn_rngs

__all__ = [
    "Observation",
    "Domain",
    "Axis",
    "resolve",
    "spawn_rngs",
]
