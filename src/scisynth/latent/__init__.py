"""Latent ground truths: what exists before any sensor looks at it.

Its public names are importable from the package itself, for example
``from scisynth.latent import AnalyticField, GaussianField``.
"""

from .base import FieldLatent, Latent, LatentGenerator
from .compose import Multichannel, Product, Sum
from .fields import AnalyticField, GaussianField, InterpolatedField

__all__ = [
    "Latent",
    "FieldLatent",
    "LatentGenerator",
    "AnalyticField",
    "InterpolatedField",
    "GaussianField",
    "Sum",
    "Product",
    "Multichannel",
]
