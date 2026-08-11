from __future__ import annotations

from typing import Any

import numpy as np

from scisynth.latent.base import LatentData, LatentDistribution, register_family
from scisynth.spec import Spec


# Right now Independent is continuous and Gaussian only
# see Tabular for mixed and categorical datasets
class Independent(LatentDistribution):
    def __init__(
        self,
        p: int = 5,
        loc: float | list[float] = 0.0,
        scale: float = 1.0,
        correlated: bool = False,
        cov: list[list[float]] | None = None,
    ) -> None:
        self.p = p
        self.loc = loc
        self.scale = scale
        self.correlated = correlated
        self.cov = cov

    def rvs(self, n: int = 1000, seed: int = 0) -> LatentData:
        return _generate(
            n=n,
            seed=seed,
            p=self.p,
            loc=self.loc,
            scale=self.scale,
            correlated=self.correlated,
            cov=self.cov,
        )

    def to_spec(self) -> Spec:
        return Spec(
            family="independent_features",
            params={
                "p": self.p,
                "loc": self.loc,
                "scale": self.scale,
                "correlated": self.correlated,
                "cov": self.cov,
            },
        )


@register_family("independent_features")
def _generate(
    n: int,
    seed: int,
    p: int = 5,
    correlated: bool = False,
    cov: list[list[float]] | None = None,
    loc: float | list[float] = 0.0,
    scale: float = 1.0,
) -> LatentData:
    # Seeding the RNG ensures the same parameters always produce the same dataset.
    rng = np.random.default_rng(seed)

    # Each feature can have its own mean. If the user passes a single float,
    # we turn it into a length-p vector so every feature shares the same mean.
    loc_arr = np.broadcast_to(np.asarray(loc, dtype=float), (p,)).copy()

    # Correlated, meaning features are not independent of one another
    if correlated:
        # A correlated distribution requires a covariance matrix that describes
        # how the features move together. For example, if A goes up B goes up.
        #
        # A valid covariance matrix must be positive semi-definite (PSD). This means
        # no direction in the data can have negative variance.
        #
        # I think basically, it just means the relationships between the variables
        # make logical, statistical sense.
        #
        # If no matrix is provided we construct a random one: A @ A.T is always PSD
        # for any A, so we use that to simply create a valid random covariance.
        # Dividing by p keeps variance from growing with the number of features.
        if cov is None:
            A = rng.normal(size=(p, p))
            cov_arr = (A @ A.T) / p
        else:
            cov_arr = np.asarray(cov, dtype=float)

        # Draw all p features jointly from a single multivariate normal.
        # Each row is one sample; the covariance matrix controls how correlated
        # the p columns are with each other.
        X = rng.multivariate_normal(loc_arr, cov_arr, size=n)
    else:
        # Without correlation, each feature is drawn independently from its own
        # univariate normal N(loc_i, scale).
        cov_arr = None
        X = rng.normal(loc=loc_arr, scale=scale, size=(n, p))

    ids = np.arange(n, dtype=np.int64)
    params: dict[str, Any] = {
        "n": n,
        "p": p,
        "correlated": correlated,
        "cov": cov_arr.tolist() if cov_arr is not None else None,
        "loc": loc_arr.tolist(),
        "scale": scale,
        "seed": seed,
    }

    # For independent features the intrinsic space Z equals the ambient space X
    # since there is no hidden lower-dimensional structure to recover. d == p.
    return LatentData(
        Z=X.copy(), X=X, ids=ids, family="independent_features", params=params
    )
