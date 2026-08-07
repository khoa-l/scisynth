"""independent_features: independent/correlated tabular features. d == p."""

from __future__ import annotations

from typing import Any

import numpy as np

from scisynth.latent.base import LatentData, register_family


@register_family("independent_features")
def generate(
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
