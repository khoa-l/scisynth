"""Rules for step names, shared by the sampler and the stages."""

from __future__ import annotations

# Reserved for addressing, e.g. ``stage__param``; never allowed in names.
RESERVED_SEPARATOR = "__"


def check_name(name: object) -> str:
    """Return ``name`` if it is a valid step name, else raise ``ValueError``.

    Parameters
    ----------
    name : object
        The candidate name.

    Returns
    -------
    str
        ``name`` unchanged.

    Raises
    ------
    ValueError
        If ``name`` is not a non-empty string or contains ``"__"``.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"Names must be non-empty strings, got {name!r}")
    if RESERVED_SEPARATOR in name:
        raise ValueError(
            f"Name {name!r} contains {RESERVED_SEPARATOR!r}, which is reserved"
        )
    return name
