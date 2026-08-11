from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse

if TYPE_CHECKING:
    from scisynth.latent.base import LatentData
    from scisynth.spec import OperatorSpec


@dataclass
class ProvenanceStep:
    operator: OperatorSpec
    ids: np.ndarray  # which latent rows remain after this step
    X: np.ndarray  # data state after this operator was applied


@dataclass
class Provenance:
    source: LatentData
    observed_steps: list[ProvenanceStep] = field(default_factory=list)
    projected_steps: list[ProvenanceStep] = field(default_factory=list)

    @property
    def ids(self) -> np.ndarray:
        """Final observed ids (latent row indices) after all operators."""
        if self.observed_steps:
            return self.observed_steps[-1].ids
        return self.source.ids

    def state_after(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """(X, ids) after the nth observed operator (0-indexed)."""
        s = self.observed_steps[step]
        return s.X, s.ids

    def latent_to_observed(self) -> scipy.sparse.csr_matrix:
        """Sparse (m, n) matrix: W[i, j] = weight of observed row i from latent row j."""
        final_ids = self.ids
        m = len(final_ids)
        n = len(self.source.ids)
        data = np.ones(m, dtype=np.float32)
        row = np.arange(m)
        col = final_ids
        return scipy.sparse.csr_matrix((data, (row, col)), shape=(m, n))

    def coverage(self) -> np.ndarray:
        """Number of times each latent point was observed. Shape (n,)."""
        n = len(self.source.ids)
        counts = np.zeros(n, dtype=np.int64)
        np.add.at(counts, self.ids, 1)
        return counts
