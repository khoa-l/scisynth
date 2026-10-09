"""xarray adapter (stub).

Optional dependency: import xarray lazily inside the function.
"""

from __future__ import annotations

from typing import Any

from ..latent.base import Latent


def from_xarray(data: Any, **kwargs: Any) -> Latent:
    """Wrap an xarray DataArray/Dataset as a latent. Not implemented yet."""
    raise NotImplementedError
