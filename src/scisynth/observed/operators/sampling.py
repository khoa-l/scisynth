from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial import KDTree

from scisynth.observed.operators.base import (
    ObservationState,
    Operator,
    register_operator,
)


def _estimate_density(X: np.ndarray, k: int) -> np.ndarray:
    # For each row, find its k nearest neighbours and take the mean distance.
    # Points in dense regions have small mean distances; sparse regions have large ones.
    # Invert to get density: small distance → high density.
    k = min(k, len(X) - 1)
    tree = KDTree(X)
    distances, _ = tree.query(
        X, k=k + 1
    )  # +1 because the point itself is always returned first
    mean_dist = distances[:, 1:].mean(axis=1)  # drop the self-distance (always 0)
    return 1.0 / (mean_dist + 1e-10)


def _resolve_count(params: dict, total: int) -> int:
    if params["frac"] is not None:
        return round(total * params["frac"])
    return params["n"]


@register_operator
class UniformSubsample(Operator):
    # Random subset of rows / equal chance for every row.
    name = "uniform_subsample"

    def __init__(self, frac: float | None = None, n: int | None = None) -> None:
        if (frac is None) == (n is None):
            raise ValueError("Provide exactly one of frac or n.")
        super().__init__(frac=frac, n=n)

    def apply(
        self, state: ObservationState, rng: np.random.Generator
    ) -> ObservationState:
        total = state.X.shape[0]
        count = _resolve_count(self.params, total)
        if count >= total:
            return state
        idx = np.sort(rng.choice(total, size=count, replace=False))
        return ObservationState(X=state.X[idx], ids=state.ids[idx])


@register_operator
class SparseSubsample(Operator):
    # Preferentially keeps rows from sparse / underrepresented regions.
    name = "sparse_subsample"

    def __init__(
        self, frac: float | None = None, n: int | None = None, k: int = 10
    ) -> None:
        if (frac is None) == (n is None):
            raise ValueError("Provide exactly one of frac or n.")
        super().__init__(frac=frac, n=n, k=k)

    def apply(
        self, state: ObservationState, rng: np.random.Generator
    ) -> ObservationState:
        total = state.X.shape[0]
        count = _resolve_count(self.params, total)
        if count >= total:
            return state
        density = _estimate_density(state.X, self.params["k"])
        weights = 1.0 / density  # low density → high weight
        probs = weights / weights.sum()
        idx = np.sort(rng.choice(total, size=count, replace=False, p=probs))
        return ObservationState(X=state.X[idx], ids=state.ids[idx])


@register_operator
class DenseSubsample(Operator):
    # Preferentially keeps rows from dense / high-signal regions.
    name = "dense_subsample"

    def __init__(
        self, frac: float | None = None, n: int | None = None, k: int = 10
    ) -> None:
        if (frac is None) == (n is None):
            raise ValueError("Provide exactly one of frac or n.")
        super().__init__(frac=frac, n=n, k=k)

    def apply(
        self, state: ObservationState, rng: np.random.Generator
    ) -> ObservationState:
        total = state.X.shape[0]
        count = _resolve_count(self.params, total)
        if count >= total:
            return state
        density = _estimate_density(state.X, self.params["k"])
        probs = density / density.sum()
        idx = np.sort(rng.choice(total, size=count, replace=False, p=probs))
        return ObservationState(X=state.X[idx], ids=state.ids[idx])


@register_operator
class BoundingBoxSubsample(Operator):
    # Keeps rows whose features fall within one or more bounding boxes (union).
    # Each box is a list of (min, max) per feature; use None to leave a side open.
    name = "bounding_box_subsample"

    def __init__(self, boxes: list[list[tuple[float | None, float | None]]]) -> None:
        super().__init__(boxes=boxes)

    def apply(
        self, state: ObservationState, rng: np.random.Generator
    ) -> ObservationState:
        union = np.zeros(state.X.shape[0], dtype=bool)
        for box in self.params["boxes"]:
            # each box independently masks rows; union across all boxes
            mask = np.ones(state.X.shape[0], dtype=bool)
            for i, (lo, hi) in enumerate(box):
                if lo is not None:
                    mask &= state.X[:, i] >= lo
                if hi is not None:
                    mask &= state.X[:, i] <= hi
            union |= mask
        idx = np.where(union)[0]
        return ObservationState(X=state.X[idx], ids=state.ids[idx])


@register_operator
class RadiusSubsample(Operator):
    # Keeps rows within range of one or more center points (union).
    # sites: list of {"center": [...], "radius": float}
    name = "radius_subsample"

    def __init__(self, sites: list[dict]) -> None:
        super().__init__(sites=sites)

    def apply(
        self, state: ObservationState, rng: np.random.Generator
    ) -> ObservationState:
        union = np.zeros(state.X.shape[0], dtype=bool)
        for site in self.params["sites"]:
            center = np.asarray(site["center"], dtype=float)
            dists = np.linalg.norm(state.X - center, axis=1)
            union |= dists <= site["radius"]
        idx = np.where(union)[0]
        return ObservationState(X=state.X[idx], ids=state.ids[idx])


@register_operator
class ThresholdSubsample(Operator):
    # Keeps rows matching one or more threshold conditions.
    # mode="any" keeps rows passing at least one condition (OR-gate).
    # mode="all" keeps rows passing every condition (AND-gate).
    # conditions: list of {"column": int, "threshold": float, "direction": "above"|"below"}
    name = "threshold_subsample"

    def __init__(self, conditions: list[dict], mode: str = "any") -> None:
        if mode not in ("any", "all"):
            raise ValueError("mode must be 'any' or 'all'.")
        super().__init__(conditions=conditions, mode=mode)

    def apply(
        self, state: ObservationState, rng: np.random.Generator
    ) -> ObservationState:
        masks = []
        for cond in self.params["conditions"]:
            col = state.X[:, cond["column"]]
            if cond["direction"] == "above":
                masks.append(col >= cond["threshold"])
            else:
                masks.append(col <= cond["threshold"])

        # combine masks according to mode
        combined = np.stack(masks, axis=0)
        mask = (
            combined.any(axis=0)
            if self.params["mode"] == "any"
            else combined.all(axis=0)
        )
        idx = np.where(mask)[0]
        return ObservationState(X=state.X[idx], ids=state.ids[idx])


@register_operator
class StratifiedSubsample(Operator):
    # Clusters rows and keeps an equal quota from each cluster.
    name = "stratified_subsample"

    def __init__(self, n_per_stratum: int, n_strata: int = 5) -> None:
        super().__init__(n_per_stratum=n_per_stratum, n_strata=n_strata)

    def apply(
        self, state: ObservationState, rng: np.random.Generator
    ) -> ObservationState:
        # Whiten normalises each feature to unit variance so no single feature
        # dominates the clustering due to scale differences.
        whitened = whiten(state.X.astype(float))
        _, labels = kmeans2(
            whitened,
            k=self.params["n_strata"],
            minit="points",
            seed=int(rng.integers(0, 2**31)),
        )
        idx = []
        for stratum in range(self.params["n_strata"]):
            stratum_idx = np.where(labels == stratum)[0]
            k = min(self.params["n_per_stratum"], len(stratum_idx))
            idx.extend(rng.choice(stratum_idx, size=k, replace=False))
        return ObservationState(X=state.X[idx], ids=state.ids[idx])


@register_operator
class SystematicSubsample(Operator):
    # Keeps every Nth row starting from offset.
    name = "systematic_subsample"

    def __init__(self, step: int, offset: int = 0) -> None:
        super().__init__(step=step, offset=offset)

    def apply(
        self, state: ObservationState, rng: np.random.Generator
    ) -> ObservationState:
        idx = np.arange(self.params["offset"], state.X.shape[0], self.params["step"])
        return ObservationState(X=state.X[idx], ids=state.ids[idx])
