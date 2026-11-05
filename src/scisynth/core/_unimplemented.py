"""Base for classes that are planned but not implemented yet."""

from __future__ import annotations

from typing import Any


class Unimplemented:
    """Mixin for a planned class: creating an instance raises ``NotImplementedError``.

    List it first among the bases so that its ``__init__`` runs. The class still
    registers as a component, so loading a saved config that names it fails with
    the same error instead of an unknown-name error.

    Parameters
    ----------
    *args, **kwargs
        Accepted and ignored, so any call to the planned class raises.

    Raises
    ------
    NotImplementedError
        Always, naming the class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"{type(self).__name__} is not implemented yet")
