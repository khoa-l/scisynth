from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scisynth.observed.base import ObservedData
from scisynth.observed.operators.base import ObservationState, Operator
from scisynth.provenance import Provenance, ProvenanceStep

if TYPE_CHECKING:
    from scisynth.latent.base import LatentData


def observe(
    latent: LatentData,
    operators: list[Operator],
    seed: int = 0,
) -> tuple[ObservedData, Provenance]:
    rng = np.random.default_rng(seed)
    state = ObservationState(X=latent.X.copy(), ids=latent.ids.copy())
    steps: list[ProvenanceStep] = []

    for op in operators:
        state = op.apply(state, rng)
        steps.append(
            ProvenanceStep(
                layer="observed",
                operator=op.to_spec(),
                ids=state.ids.copy(),
                X=state.X.copy(),
            )
        )

    observed = ObservedData(
        X=state.X,
        ids=state.ids,
        operators=[op.to_spec() for op in operators],
    )
    provenance = Provenance(source=latent, observed_steps=steps)
    return observed, provenance
