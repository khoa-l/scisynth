from __future__ import annotations

from typing import Any

import numpy as np

from scisynth.latent.base import LatentData, LatentDistribution, register_family
from scisynth.spec import Spec


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
    rng = np.random.default_rng(seed)
    loc_arr = np.broadcast_to(np.asarray(loc, dtype=float), (p,)).copy()

    if correlated:
        if cov is None:
            A = rng.normal(size=(p, p))
            cov_arr = (A @ A.T) / p  # guaranteed PSD
        else:
            cov_arr = np.asarray(cov, dtype=float)
        X = rng.multivariate_normal(loc_arr, cov_arr, size=n)
    else:
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
    return LatentData(
        Z=X.copy(), X=X, ids=ids, family="independent_features", params=params
    )
