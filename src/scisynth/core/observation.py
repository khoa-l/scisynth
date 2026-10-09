"""The data object that flows through an observation pipeline."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from ._types import Array, BoolArray, FloatArray


@dataclass
class Observation:
    """Observed values plus everything needed to interpret them.

    The last axis of ``values`` is always the channel axis, so one measurement per
    location is ``d == 1``. Two layouts exist, told apart by shape: *grid* (one 1-D
    coordinate array per spatial axis) and *points* (every coordinate array has the
    spatial shape).

    Parameters
    ----------
    values : ndarray of shape (*spatial, d)
        Measurements. Invalid locations hold NaN in every channel.
    coords : dict of str to ndarray
        Coordinates by axis name, in axis order.
    mask : ndarray of bool, shape (*spatial)
        True where a location is valid, for all channels at once.
    truth : ndarray of shape (*spatial, d), optional
        The noise-free sample of the latent, kept for evaluation. Stages never
        change it.
    meta : list of dict
        One record per step that ran, in order:
        ``{"name": step name, "type": class name, "params": resolved parameters}``.
    ids : ndarray of int, shape (*spatial), optional
        A unique identifier for each location, which follows the location through
        the pipeline so that steps can be compared even when locations are removed
        or added. Defaults to ``0, 1, 2, ...`` in storage order, which is what the
        samplers produce. A stage that changes the number of locations must pass
        ``ids`` with the new spatial shape (``replace`` raises otherwise), keeping
        the ids of the locations it keeps and giving new ones fresh values.

    Attributes
    ----------
    shape : tuple of int
        Shape of ``values``, including the channel axis.
    spatial_shape : tuple of int
        Shape without the channel axis; also the shape of ``mask``.
    size : int
        Number of locations: the product of ``spatial_shape``.
    n_channels : int
        Length of the channel axis.
    valid : ndarray of bool, shape (*spatial, 1)
        ``mask`` with a trailing axis, to broadcast over channels.
    spatial_coords : dict of str to ndarray
        ``coords`` broadcast to the spatial shape: one array per axis, whatever the
        layout.
    layout : {"grid", "points"}
        ``"grid"`` if the coordinates are one 1-D array per spatial axis, else
        ``"points"``. A 1-D observation fits both and counts as a grid.
    plot_subjects, plot_kinds : tuple of str
        What :func:`scisynth.viz.plot` draws it as: the subject ``"observation"`` in
        the kinds ``"scatter"`` (the default) and ``"heatmap"``. Both are for 2-D
        observations.
    """

    plot_subjects: ClassVar[tuple[str, ...]] = ("observation",)
    plot_kinds: ClassVar[tuple[str, ...]] = ("scatter", "heatmap")

    values: FloatArray
    coords: dict[str, Array]
    mask: BoolArray
    truth: Array | None = None
    meta: list[dict[str, Any]] = field(default_factory=list)
    ids: Array = field(default=None)  # type: ignore[arg-type]  # filled in __post_init__

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        self.mask = np.asarray(self.mask, dtype=np.bool_)
        if self.values.ndim < 1:
            raise ValueError("values needs a trailing channel axis, got a scalar")
        spatial = self.spatial_shape
        if self.ids is None:
            self.ids = np.arange(self.size, dtype=np.int64).reshape(spatial)
        else:
            self.ids = np.asarray(self.ids, dtype=np.int64)
            if self.ids.shape != spatial:
                raise ValueError(
                    f"ids shape {self.ids.shape} != spatial shape {spatial}; a stage "
                    "that changes the number of locations must also pass ids"
                )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape[:-1])

    @property
    def size(self) -> int:
        return math.prod(self.spatial_shape)

    @property
    def n_channels(self) -> int:
        return int(self.values.shape[-1])

    @property
    def valid(self) -> BoolArray:
        return self.mask[..., None]

    @property
    def spatial_coords(self) -> dict[str, Array]:
        arrays = list(self.coords.values())
        if self.layout == "grid":
            arrays = list(np.meshgrid(*arrays, indexing="ij"))
        return dict(zip(self.coords, arrays, strict=True))

    @property
    def layout(self) -> Literal["grid", "points"]:
        coords = list(self.coords.values())
        spatial = self.spatial_shape
        is_grid = len(coords) == len(spatial) and all(
            c.ndim == 1 and c.shape[0] == spatial[i] for i, c in enumerate(coords)
        )
        return "grid" if is_grid else "points"

    def to_points(self) -> Observation:
        """Return the observation as a flat list of points.

        Returns
        -------
        Observation
            One location per cell in storage order, each with its own coordinates;
            ids, mask, values and ``truth`` are flattened the same way, so locations
            keep their ids. Already-flat observations are returned unchanged.
        """
        if len(self.spatial_shape) == 1:
            return self
        d = self.n_channels
        return self.replace(
            values=self.values.reshape(-1, d),
            coords={k: c.ravel() for k, c in self.spatial_coords.items()},
            mask=self.mask.ravel(),
            truth=None if self.truth is None else self.truth.reshape(-1, d),
            ids=self.ids.ravel(),
        )

    def replace(self, **changes: Any) -> Observation:
        """Return a copy with some fields replaced.

        Parameters
        ----------
        **changes
            Field names and their new values. Arrays are not copied.

        Returns
        -------
        Observation
            A new observation with the changes applied.
        """
        return dataclasses.replace(self, **changes)
