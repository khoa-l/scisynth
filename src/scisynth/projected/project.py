from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from scisynth.projected.base import ProjectedData
from scisynth.provenance import Provenance, ProvenanceStep
from scisynth.spec import OperatorSpec

if TYPE_CHECKING:
    from scisynth.observed.base import ObservedData


# method must implement fit_transform(X) -> ndarray (sklearn API).
# random_state is injected automatically when supported.
def project(
    observed: ObservedData,
    method: Any,
    provenance: Provenance | None = None,
    seed: int = 0,
) -> tuple[ProjectedData, Provenance | None]:
    if hasattr(method, "set_params"):
        try:
            method.set_params(random_state=seed)
        except (ValueError, TypeError):
            pass

    Z = method.fit_transform(observed.X)

    params: dict = {}
    if hasattr(method, "get_params"):
        params = method.get_params()

    projected = ProjectedData(
        Z=Z,
        ids=observed.ids.copy(),
        method=type(method).__name__,
        params=params,
    )

    if provenance is not None:
        step = ProvenanceStep(
            layer="projected",
            operator=OperatorSpec(name=type(method).__name__, params=params),
            ids=observed.ids.copy(),
            X=Z,
        )
        provenance = dataclasses.replace(
            provenance,
            projected_steps=[*provenance.projected_steps, step],
        )

    return projected, provenance
