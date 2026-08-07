from __future__ import annotations

import numpy as np

from scisynth.observed.operators.base import (
    ObservationState,
    Operator,
    register_operator,
)


@register_operator
class MCARMissingness(Operator):
    name = "mcar_missingness"

    def __init__(self, rate: float = 0.1) -> None:
        super().__init__(rate=rate)

    def apply(
        self, state: ObservationState, rng: np.random.Generator
    ) -> ObservationState:
        X = state.X.copy()
        mask = rng.random(X.shape) < self.params["rate"]
        X[mask] = np.nan
        return ObservationState(X=X, ids=state.ids)
