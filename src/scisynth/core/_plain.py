"""Conversion of parameters to plain Python, for serialization and logging."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def to_plain(value: Any, *, strict: bool) -> Any:
    """Convert a parameter value to plain Python (numbers, strings, lists, dicts).

    Parameters
    ----------
    value : object
        A number, string, array, sequence, mapping or ``Component``.
    strict : bool
        True is for serialization: anything unsupported raises. False is for
        logging: unknown objects become their ``repr`` and arrays over 64 elements
        are summarized as a string.

    Returns
    -------
    object
        The plain value. Arrays become lists and tuples become lists.

    Raises
    ------
    TypeError
        If ``strict`` and ``value`` (or something inside it) is not supported, such
        as a callable or a distribution. A nested ``Component`` is converted with
        its own strict ``to_dict``, so it can raise even when ``strict`` is False.
    """
    from .component import Component  # imported here: component imports this module

    if isinstance(value, Component):
        return value.to_dict()
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if strict or value.size <= 64:
            return value.tolist()
        return f"<array shape={value.shape}>"
    if isinstance(value, list | tuple):
        return [to_plain(v, strict=strict) for v in value]
    if isinstance(value, Mapping):
        return {str(k): to_plain(v, strict=strict) for k, v in value.items()}
    if strict:
        raise TypeError(
            f"Cannot serialize parameter of type {type(value).__name__}: only numbers, "
            "strings, arrays, sequences and Components are supported (not callables "
            "or distributions)."
        )
    return repr(value)
