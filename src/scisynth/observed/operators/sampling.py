from __future__ import annotations

import numpy as np

from scisynth.observed.operators.base import (
    ObservationState,
    Operator,
    register_operator,
)


@register_operator
class RandomSubsample(Operator):
    name = "random_subsample"

    def __init__(self, frac: float = 0.8) -> None:
        super().__init__(frac=frac)

    def apply(
        self, state: ObservationState, rng: np.random.Generator
    ) -> ObservationState:
        n = state.X.shape[0]
        k = round(n * self.params["frac"])
        idx = np.sort(rng.choice(n, size=k, replace=False))
        return ObservationState(X=state.X[idx], ids=state.ids[idx])
