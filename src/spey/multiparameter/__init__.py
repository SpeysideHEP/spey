"""
Multi-parameter inference utilities for :mod:`spey`.

This sub-package provides tools for working with statistical models that have
more than one parameter of interest:

* :class:`MultiParamTemplate` / :class:`MultivariateNormal` — template classes
  that wrap a :class:`~spey.StatisticalModel` backed by
  ``"default.multivariate_normal"`` and accept a callable *signal_yields*
  function (see :mod:`spey.multiparameter.templates`).

* :func:`find_contour` — map the :math:`(1-\alpha)` chi-squared confidence
  contour in the full parameter space using a combination of radial search and
  constrained RATTLE HMC (see :mod:`spey.multiparameter.contour`).

* :class:`ContourResult` — dataclass returned by :func:`find_contour`.

* :func:`find_best_model` — scan the shared parameter space of a collection
  of :class:`MultiParamTemplate` objects and select the most constraining one
  (see :mod:`spey.multiparameter.multiparam`).
"""

from spey.multiparameter.contour import ContourResult, find_contour

# from spey.multiparameter.multiparam import find_best_model

__all__ = [
    "ContourResult",
    "find_contour",
    # "find_best_model",
]
