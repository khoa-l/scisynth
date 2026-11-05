"""Latents driven by scipy.stats distributions (stubs).

Optional dependencies must be imported lazily inside methods, not at module level.
"""

from __future__ import annotations

from ..core._unimplemented import Unimplemented
from ..latent.base import LatentGenerator


class DistributionField(Unimplemented, LatentGenerator):
    """Field whose values follow a scipy distribution. Not implemented yet."""


class DistributionCatalog(Unimplemented, LatentGenerator):
    """Catalog whose attributes follow scipy distributions. Not implemented yet."""
