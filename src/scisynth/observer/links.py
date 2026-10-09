"""Links between steps: which locations of one observation fed each of the next.

A step's links are a sparse matrix of shape ``(after.size, before.size)`` over flat
(storage-order) locations. ``links[j, i]`` is nonzero when location ``j`` of the
output depends on location ``i`` of the input, and its value is the weight of that
dependence. Only locations that are valid on both sides are linked.

Most stages keep each location's ``id``, so their links follow from the ids
(:func:`links_by_id`). A stage that merges locations supplies its own, see
:meth:`Stage.links_from <scisynth.stages.base.Stage.links_from>`. Multiplying the
links of consecutive steps gives the dependency of a later step on an earlier one.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..core.observation import Observation


def links_by_id(before: Observation, after: Observation) -> sparse.csr_matrix:
    """Link each valid location of ``after`` to the valid one of ``before`` with its id.

    Parameters
    ----------
    before, after : Observation
        The observation going into and coming out of a step.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(after.size, before.size)`` with a one wherever an id is valid on
        both sides. Ids are unique, so each row and column has at most one entry.
    """
    wanted, flat = after.ids.ravel(), before.ids.ravel()
    shape = (wanted.size, flat.size)
    if not flat.size or not wanted.size:
        return sparse.csr_matrix(shape)
    order = np.argsort(flat)
    at = np.searchsorted(flat[order], wanted).clip(max=flat.size - 1)
    pos = order[at]
    linked = (flat[pos] == wanted) & before.mask.ravel()[pos] & after.mask.ravel()
    rows = np.flatnonzero(linked)
    return sparse.csr_matrix((np.ones(rows.size), (rows, pos[rows])), shape=shape)
