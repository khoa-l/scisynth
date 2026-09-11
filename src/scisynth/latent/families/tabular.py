from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import norm as std_norm

from scisynth.latent.base import LatentData, LatentDistribution, register_family
from scisynth.spec import Spec


def _serialize_col(col: Any) -> dict:
    # Store enough to reconstruct the frozen distribution from scipy.stats.
    return {"dist": col.dist.name, "args": list(col.args), "kwds": dict(col.kwds)}


def _deserialize_col(d: dict) -> Any:
    return getattr(stats, d["dist"])(*d["args"], **d["kwds"])


class Tabular(LatentDistribution):
    def __init__(
        self,
        columns: list,  #  using scipy.stats distributions, one per column
        corr: list[list[float]] | None = None,
    ) -> None:
        self.columns = columns
        self.corr = corr

    def rvs(self, n: int = 1000, seed: int = 0) -> LatentData:
        return _generate(n=n, seed=seed, columns=self.columns, corr=self.corr)

    def to_spec(self) -> Spec:
        return Spec(
            family="tabular",
            params={
                "columns": [_serialize_col(c) for c in self.columns],
                "corr": self.corr,
            },
        )


# Might be a good idea to use the Synthetic Data Vault in the future?
# Found out about it while figuring out how to do this, lol
#
# Supports mixed continuous / categorical data
@register_family("tabular")
def _generate(
    n: int,
    seed: int,
    columns: list,  # frozen scipy distributions or serialized dicts
    corr: list[list[float]]
    | None = None,  # instead of inferring from real data, we are defining the correlations directly
) -> LatentData:
    rng = np.random.default_rng(seed)
    p = len(columns)

    # Convert columns back if they arrive from Spec as plain dicts
    cols = [_deserialize_col(c) if isinstance(c, dict) else c for c in columns]

    if corr is not None:
        # We want columns with different distributions (e.g. Gaussian,
        # Poisson) to be correlated. We can't sample them jointly directly, so we
        # use a copular, specifically a Guassian copula, to generate data
        #
        # Generate correlated Gaussian columns.
        # Use Cholesky decomp to split the correlation matrix into a mixing matrix L.
        # We multiply the independent random columns by L.T to blend them together
        # in the right proportion to create the desired correlations.
        corr_arr = np.asarray(corr, dtype=float)
        L = np.linalg.cholesky(corr_arr)
        Z_std = rng.standard_normal((n, p)) @ L.T

        # We then rescale the Gaussian columns to a common [0, 1] scale
        # keeping their rank order and correlations.
        # Clipping avoids edge values that would blow up in step 3.
        U = std_norm.cdf(Z_std).clip(1e-8, 1 - 1e-8)

        # Map each [0, 1] column to its target distribution.
        # Put a uniform [0, 1] value into a distribution's inverse CDF
        # to produce a sample from that distribution. Each column gets its own
        # distribution, so the result is mixed-type with the right correlations.
        X = np.column_stack([col.ppf(U[:, i]) for i, col in enumerate(cols)])

        # Categorical columns have a step function for their PPF so correlations are approximate
        # between cat. cols

    else:
        # If independent then each column is sampled directly from its own distribution
        # with no relationship to the other columns.
        X = np.column_stack([col.rvs(n, random_state=rng) for col in cols])

    ids = np.arange(n, dtype=np.int64)
    params: dict[str, Any] = {
        "n": n,
        "columns": [_serialize_col(c) for c in cols],
        "corr": corr,
        "seed": seed,
    }

    # No hidden structure so the intrinsic space equals ambient space (d == p).
    return LatentData(Z=X.copy(), X=X, ids=ids, family="tabular", params=params)
