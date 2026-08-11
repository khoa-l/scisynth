from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from scisynth.spec import OperatorSpec


@dataclass
class ObservedData:
    X: np.ndarray
    ids: np.ndarray
    operators: list[OperatorSpec] = field(default_factory=list)

    def to_frame(self) -> pl.DataFrame:
        p = self.X.shape[1]
        df = pl.DataFrame(self.X, schema=[f"x{i}" for i in range(p)])
        return df.with_columns(pl.Series("id", self.ids)).select(["id", *df.columns])
