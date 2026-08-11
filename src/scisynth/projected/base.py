from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl


@dataclass
class ProjectedData:
    Z: np.ndarray        # (n, d) low-dimensional coordinates
    ids: np.ndarray      # same ids as the ObservedData that was projected
    method: str          # name of the DR method
    params: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pl.DataFrame:
        d = self.Z.shape[1]
        df = pl.DataFrame(self.Z, schema=[f"z{i}" for i in range(d)])
        return df.with_columns(pl.Series("id", self.ids)).select(["id", *df.columns])
