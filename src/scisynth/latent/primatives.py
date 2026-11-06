
from collections.abc import Callable, Sequence

import numpy as np

from .fields import InterpolatedField
from .fields import AnalyticField
from .base import FieldLatent
from ..core.domain import Domain
from ..core._grid import as_shape

# Note: current application method assumes single-channel data is being applied
# if a multi-channel latent is passed in, only the first channel will be used.
def apply_mask(
        mask: InterpolatedField,
        inside: FieldLatent,
        outside: FieldLatent
) -> AnalyticField:
    # Maybe naieve way of handling this, but it makes sense to me
    # Hopefully multiple applications of mask won't become too slow.
    def apply(*coords):
          points = np.column_stack(coords)
          m = np.asarray(mask.evaluate(points))[:,0]
          ins = np.asarray(inside.evaluate(points))[:,0]
          outs = np.asarray(outside.evaluate(points))[:,0]
          return m * ins + (1 - m) * outs

    return AnalyticField(apply, mask.domain)

def flatten_masks(
            masks: list[InterpolatedField],
            add: bool = True
) -> AnalyticField:
        if not masks or len(masks) < 2:
            raise ValueError("Cannot flatten mask[s]: Too few mask objects provided.")
        domain = masks[0].domain
        for mask in masks:
              if mask.domain != domain:
                    raise ValueError("Masks must share a common domain.")

        def combine(*coords):
                points = np.column_stack(coords)
                combined = []

                for mask in masks:
                    # If multichannel mask passed, use only first channel
                    combined.append(np.asarray(mask.evaluate(points))[:,0])

                result = combined[0]
                for vals in combined[1:]:
                      result = np.maximum(result, vals) if add else result * (1-vals)
                return result

        return AnalyticField(combine, domain)

# Note: currently hard-coded to 2D fields, will expand to 3- or 4D later.
# Error checking happens external to this function, assume passed parameters are well formed
def binary_shape_field(
       domain: Domain,
       resolution: Sequence[float],
       contains: Callable[[np.ndarray, np.ndarray], np.ndarray],
       order: int =0
) -> InterpolatedField:
    shape = as_shape(resolution, 2, "resolution")
    # This comprehension is confusing to me
    # startPoint + step * (range) / stepSize
    # for {number of requested samples (resolution)}
    # results in 1D array of coordinates
    axes = [
          low + np.arange(n) * (high - low) / n
          for (low, high), n in zip(domain.extents, shape, strict=True)]
    # transform 1D array into 2D array
    # Note: this is another hard-coded 2D process, can generalize later
    x,y = np.meshgrid(*axes, indexing="ij")
    # Finally produce the mask by checking sampled array against the passed comparison 'contains'
    values = contains(x,y).astype(np.float64)

    return InterpolatedField(values, domain, order)

def ellipse_mask(
        domain: Domain,                 # Domain must match dimensions passed
        resolution: Sequence[float],    # to the resolution, center, and radii
        center: Sequence[float],        # of the ellipse mask
        radii: Sequence[float],
        rotation: float =0,             # rotation not implemented yet
        order: int =0                   # assume masks are hard boundries unless specified
) -> InterpolatedField:
    res = np.asarray(resolution, np.float64)
    cent = np.asarray(center, np.float64)
    rad = np.asarray(radii, np.float64)
    if res.shape != (domain.ndim,):
            raise ValueError(f"Resolution must contain {domain.ndim} coordinates, corrisponding with passed domain.")
    if cent.shape != (domain.ndim,):
            raise ValueError(f"Center must contain {domain.ndim} coordinates, corrisponding with passed domain.")
    if rad.shape != (domain.ndim,):
            raise ValueError(f"Radii must contain {domain.ndim} coordinates, corrisponding with passed domain.")
    # Temporary hard-coding to 2D
    if domain.ndim != 2:
        raise ValueError("Unimplemented feature: only working with 2D arrays at the moment.")
    cx, cy = center
    rx, ry = radii
    contains = lambda x, y: ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1

    return binary_shape_field(domain, resolution, contains, order=order)

def rectangle_mask(
            domain: Domain,
            resolution: Sequence[float],
            center: Sequence[float],
            dimensions: Sequence[float],
            order: int =0
) -> InterpolatedField:
    res = np.asarray(resolution, np.float64)
    cent = np.asarray(center, np.float64)
    dims = np.asarray(dimensions, np.float64)
    if res.shape != (domain.ndim,):
            raise ValueError(f"Resolution must contain {domain.ndim} coordinates, corrisponding with passed domain.")
    if cent.shape != (domain.ndim,):
            raise ValueError(f"Center must contain {domain.ndim} coordinates, corrisponding with passed domain.")
    if dims.shape != (domain.ndim,):
                raise ValueError(f"Dimensions must contain {domain.ndim} coordinates, corrisponding with passed domain.")
    # Temporary hard-coding to 2D
    if domain.ndim != 2:
        raise ValueError("Unimplemented feature: only working with 2D arrays at the moment.")
    cx, cy = center
    w, h = dimensions
    contains = lambda x,y: (np.abs(x-cx) <= w/2) & (np.abs(y-cy) <= h/2)

    return binary_shape_field(domain, resolution, contains, order=order)


#TODO: Polygon Mask