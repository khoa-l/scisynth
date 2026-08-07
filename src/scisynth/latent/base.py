from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scisynth.spec import Spec


@dataclass
class LatentData:
    Z: np.ndarray  # (n, d) intrinsic coordinates
    X: np.ndarray  # (n, p) clean ambient realization, ground truth for O
    ids: np.ndarray  # (n,) int64, opaque row identifiers
    family: str
    params: dict[str, Any] = field(default_factory=dict)


FamilyFn = Callable[..., LatentData]
FAMILY_REGISTRY: dict[str, FamilyFn] = {}


def register_family(name: str) -> Callable[[FamilyFn], FamilyFn]:
    def deco(fn: FamilyFn) -> FamilyFn:
        FAMILY_REGISTRY[name] = fn
        return fn

    return deco


def generate_latent(spec: Spec) -> LatentData:
    if spec.family not in FAMILY_REGISTRY:
        known = ", ".join(sorted(FAMILY_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"Unknown latent family '{spec.family}'. Known families: {known}"
        )
    fn = FAMILY_REGISTRY[spec.family]
    return fn(n=spec.n, seed=spec.seed, **spec.family_params)
