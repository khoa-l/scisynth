"""Shared numpy type aliases."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
