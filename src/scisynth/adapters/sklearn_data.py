"""scikit-learn dataset generators as catalogs (stubs).

Optional dependencies must be imported lazily inside methods, not at module level.
"""

from __future__ import annotations

from ..core._unimplemented import Unimplemented
from ..latent.base import LatentGenerator


class SklearnDataCatalog(Unimplemented, LatentGenerator):
    """Catalog made with a scikit-learn data generator, such as ``make_blobs``.

    Not implemented yet.
    """
