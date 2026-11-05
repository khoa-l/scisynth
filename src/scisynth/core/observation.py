"""The data object that flows through an observation pipeline."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from ._types import Array, BoolArray, FloatArray


@dataclass(eq=False)
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
        change its values; a stage that changes the locations moves, merges or
        extends it with them.
    meta : list of dict
        One record per step that ran, in order:
        ``{"name": step name, "type": class name, "params": resolved parameters}``.
    ids : ndarray of int, shape (*spatial), optional
        A unique identifier for each location, which follows the location through
        the pipeline so that steps can be compared even when locations are removed
        or added. Defaults to ``0, 1, 2, ...`` in storage order, which is what the
        samplers produce. A stage that changes the number of locations must pass
        ``ids`` with the new spatial shape, keeping the ids of the locations it
        keeps and giving new ones fresh values.

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
        ``"points"``. A 1-D observation has the same shapes either way, so it is a
        grid only if its coordinate is strictly increasing or decreasing:
        neighbors in storage order must be neighbors in space.
    plot_subjects, plot_kinds : tuple of str
        What :func:`scisynth.viz.plot` draws it as: the subject ``"observation"`` in
        the kinds ``"scatter"`` (the default) and ``"heatmap"``. Both are for 2-D
        observations.

    Raises
    ------
    ValueError
        If the arrays do not have matching shapes (see :func:`check_shapes`).
        Because ``replace`` builds a new observation, a stage that changes the
        number of locations must pass ``values``, ``mask``, ``coords``, ``ids`` and
        ``truth`` together in one ``replace``.

    Notes
    -----
    ``==`` compares identity, not contents; use
    :func:`scisynth.testing.invariants.observations_equal` to compare contents.
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
        if self.ids is None:
            self.ids = np.arange(self.size, dtype=np.int64).reshape(self.spatial_shape)
        else:
            self.ids = np.asarray(self.ids, dtype=np.int64)
        check_shapes(self)

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
        if is_grid and len(spatial) == 1:
            step = np.diff(coords[0])
            is_grid = bool(np.all(step > 0) or np.all(step < 0))
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


def check_shapes(obs: Observation) -> None:
    """Check that the arrays of an observation have matching shapes.

    ``mask`` and ``ids`` must have the spatial shape, ``truth`` (if given) the shape
    of ``values``, and the coordinates must fit a grid or points layout. Only shapes
    are checked, so it is cheap: :class:`Observation` runs it on every construction.

    Parameters
    ----------
    obs : Observation
        The observation to check.

    Raises
    ------
    ValueError
        If a shape does not match.
    """
    spatial = obs.spatial_shape
    if obs.ids.shape != spatial:
        raise ValueError(
            f"ids shape {obs.ids.shape} != spatial shape {spatial}; a stage that "
            "changes the number of locations must also pass ids"
        )
    if obs.mask.shape != spatial:
        raise ValueError(
            f"mask shape {obs.mask.shape} != spatial shape {spatial} of values "
            f"{obs.values.shape}"
        )
    if obs.truth is not None and np.shape(obs.truth) != obs.values.shape:
        raise ValueError(
            f"truth shape {np.shape(obs.truth)} != values shape {obs.values.shape}"
        )
    coords = list(obs.coords.values())
    if obs.layout == "points" and any(np.shape(c) != spatial for c in coords):
        raise ValueError(
            f"coords {[np.shape(c) for c in coords]} match neither a grid nor points "
            f"layout for values of shape {obs.values.shape}"
        )
