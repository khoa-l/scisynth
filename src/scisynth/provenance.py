from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from scisynth.latent.base import LatentData
    from scisynth.spec import OperatorSpec


@dataclass
class ProvenanceStep:
    layer: str  # "observed" or "projected"
    operator: OperatorSpec
    ids: np.ndarray  # row indices into the previous layer's data
    X: np.ndarray  # data state after this step

    def to_frame(self) -> pl.DataFrame:
        prefix = "z" if self.layer == "projected" else "x"
        p = self.X.shape[1]
        df = pl.DataFrame(self.X, schema=[f"{prefix}{i}" for i in range(p)])
        return df.with_columns(pl.Series("id", self.ids)).select(["id", *df.columns])


@dataclass
class Provenance:
    source: LatentData
    observed_steps: list[ProvenanceStep] = field(default_factory=list)
    projected_steps: list[ProvenanceStep] = field(default_factory=list)

    @property
    def ids(self) -> np.ndarray:
        if self.observed_steps:
            return self.observed_steps[-1].ids
        return self.source.ids

    def to_frame(self) -> pl.DataFrame:
        rows: list[dict] = []
        n_in = len(self.source.ids)
        rows.append(
            {
                "layer": "latent",
                "step": -1,
                "operator": self.source.family,
                "n_in": n_in,
                "n_out": n_in,
            }
        )
        for i, step in enumerate(self.observed_steps):
            n_out = len(step.ids)
            rows.append(
                {
                    "layer": "observed",
                    "step": i,
                    "operator": step.operator.name,
                    "n_in": n_in,
                    "n_out": n_out,
                }
            )
            n_in = n_out
        for i, step in enumerate(self.projected_steps):
            n_out = len(step.ids)
            rows.append(
                {
                    "layer": "projected",
                    "step": i,
                    "operator": step.operator.name,
                    "n_in": n_in,
                    "n_out": n_out,
                }
            )
            n_in = n_out
        return pl.DataFrame(
            rows,
            schema={
                "layer": pl.String,
                "step": pl.Int32,
                "operator": pl.String,
                "n_in": pl.Int64,
                "n_out": pl.Int64,
            },
        )

    def mapping_frame(self) -> pl.DataFrame:
        m = len(self.ids)
        n = len(self.source.ids)
        counts = np.zeros(n, dtype=np.int64)
        np.add.at(counts, self.ids, 1)

        data: dict = {
            "latent_id": self.ids,
            "observed_idx": np.arange(m, dtype=np.int64),
            "coverage": counts[self.ids],
        }

        if self.projected_steps:
            # Projection is currently bijective and order-preserving:
            # observed row i → projected row i.
            data["projected_idx"] = np.arange(m, dtype=np.int64)

        return pl.DataFrame(data)
