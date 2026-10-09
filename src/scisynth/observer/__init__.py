"""Observers: a sampler and stages run as one instrument, and the record of a run.

Holds the ``Observer``, its ``StageList``, and the ``Trace`` of one run with the
``StageDiff`` of each stage. These names are importable from the package itself,
for example ``from scisynth.observer import Observer, Trace``.
"""

from .observer import Observer
from .stage_list import StageList
from .trace import StageDiff, Trace

__all__ = [
    "Observer",
    "StageList",
    "Trace",
    "StageDiff",
]
