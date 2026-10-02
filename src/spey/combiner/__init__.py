"""
Statistical model combination.

* :class:`~spey.combiner.UncorrelatedStatisticsCombiner` — ``default.uncorrelated_combiner``
  plug-in: independent analyses sharing only their parameter of interest.
* :class:`~spey.combiner.CorrelatedStatisticsCombiner` — ``default.correlated_combiner``
  plug-in: analyses sharing any declared parameters.
* :class:`~spey.combiner.combiner_core.CombinerBase` — common base class of both.
* :class:`~spey.UnCorrStatisticsCombiner` — deprecated wrapper around
  ``default.uncorrelated_combiner``.
"""

from .combiner_core import CombinerBase
from .correlated_statistics_combiner import CorrelatedStatisticsCombiner
from .deprecated_combiner import UnCorrStatisticsCombiner
from .uncorrelated_statistics_combiner import UncorrelatedStatisticsCombiner

__all__ = [
    "CombinerBase",
    "UncorrelatedStatisticsCombiner",
    "CorrelatedStatisticsCombiner",
    "UnCorrStatisticsCombiner",
]
