"""polars adapter (stub).

Optional dependency: import polars lazily inside the function.
"""

from __future__ import annotations

from typing import Any

from ..latent.dataset import DatasetLatent


def from_dataframe(df: Any, **kwargs: Any) -> DatasetLatent:
    """Wrap a polars DataFrame as a DatasetLatent. Not implemented yet."""
    raise NotImplementedError
