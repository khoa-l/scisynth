from __future__ import annotations

import numpy as np

from scisynth.observed.operators.base import (
    ObservationState,
    Operator,
    register_operator,
)


@register_operator
class GaussianNoise(Operator):
    name = "gaussian_noise"

    def __init__(self, sigma: float = 0.1) -> None:
        super().__init__(sigma=sigma)

    def apply(
        self, state: ObservationState, rng: np.random.Generator
    ) -> ObservationState:
        X = state.X + rng.normal(0.0, self.params["sigma"], size=state.X.shape)
        return ObservationState(X=X, ids=state.ids)
