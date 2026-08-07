from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from scisynth.spec import OperatorSpec


@dataclass
class ObservationState:
    X: np.ndarray
    ids: np.ndarray


class Operator(ABC):
    name: str = "operator"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def apply(
        self, state: ObservationState, rng: np.random.Generator
    ) -> ObservationState: ...

    def to_spec(self) -> OperatorSpec:
        return OperatorSpec(name=self.name, params=dict(self.params))


OPERATOR_REGISTRY: dict[str, type[Operator]] = {}


def register_operator(cls: type[Operator]) -> type[Operator]:
    OPERATOR_REGISTRY[cls.name] = cls
    return cls
