from typing import Optional, Text, Tuple, List, Callable, Any
from functools import wraps

import numpy as np

from spey.utils import ExpectationType
from spey.base.backend_base import BackendBase
from spey.backends import AvailableBackends
from spey.system.exceptions import UnknownCrossSection
from spey.base.hypotest_base import HypothesisTestingBase

__all__ = ["StatisticalModel", "statistical_model_wrapper"]


class StatisticalModel(HypothesisTestingBase):
    """
    Statistical model base

    :param backend: Statistical model backend
    :param xsection: Cross-section in pb
    :param analysis: name of the analysis
    """

    __slots__ = ["_backend", "xsection", "analysis"]

    def __init__(self, backend: BackendBase, analysis: Text, xsection: float = np.NaN):
        assert isinstance(backend, BackendBase), "Invalid backend"
        self._backend: BackendBase = backend
        self.xsection: float = xsection
        self.analysis: Text = analysis

    def __repr__(self):
        return (
            f"StatisticalModel(analysis='{self.analysis}', "
            f"xsection={self.xsection:.3e} [pb], "
            f"backend={str(self.backend_type)})"
        )

    @property
    def backend(self) -> BackendBase:
        """Get backend"""
        return self._backend

    @property
    def backend_type(self) -> AvailableBackends:
        return self.backend.type

    def excluded_cross_section(
        self, expected: Optional[ExpectationType] = ExpectationType.observed
    ) -> float:
        """
        Compute excluded cross section at 95% CLs

        :param expected: observed, apriori or aposteriori
        :return: excluded cross section value in pb
        :raises UnknownCrossSection: if cross section is nan.
        """
        if np.isnan(self.xsection):
            raise UnknownCrossSection("Cross-section value has not been initialised.")

        return self.poi_upper_limit(expected=expected, confidence_level=0.95) * self.xsection

    @property
    def s95exp(self) -> float:
        """Expected excluded cross-section (apriori)"""
        return self.excluded_cross_section(ExpectationType.apriori)

    @property
    def s95obs(self) -> float:
        """Observed excluded cross-section"""
        return self.excluded_cross_section(ExpectationType.observed)

    def likelihood(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        return_nll: Optional[bool] = True,
        **kwargs,
    ) -> float:
        """
        Compute the likelihood of the given statistical model

        :param poi_test: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param return_nll: if true returns negative log-likelihood value
        :param kwargs: backend specific inputs.
        :return: (float) likelihood
        """
        return self.backend.likelihood(
            poi_test=poi_test, expected=expected, return_nll=return_nll, **kwargs
        )

    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = True,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Find the POI that maximizes the likelihood and the value of the maximum likelihood

        :param return_nll: if true, likelihood will be returned
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: allow negative POI
        :param kwargs: backend specific inputs.
        :return: muhat, maximum of the likelihood
        """
        return self.backend.maximize_likelihood(
            return_nll=return_nll,
            expected=expected,
            allow_negative_signal=allow_negative_signal,
            **kwargs,
        )


def statistical_model_wrapper(
    func,
) -> Callable[[tuple[Any, ...], str, float, dict[str, Any]], StatisticalModel]:
    """
    Wrapper for statistical model backends

    :param func: Takes a specific statistical model backend and turns it into StatisticalModel class
    """

    @wraps(func)
    def wrapper(*args, analysis: Text = "__unknown_analysis__", xsection: float = np.nan, **kwargs):
        """
        :param args: Input arguments for statistical model backend
        :param analysis: analysis name
        :param xsection: cross section value
        :param kwargs: keyword arguments for statistical model backend.
        :return: Statistical model interface
        :raises AssertionError: if the input function is not BacendBase
        """
        return StatisticalModel(backend=func(*args, **kwargs), analysis=analysis, xsection=xsection)

    wrapper.__doc__ += (
        "\nFollowing additional keyword arguments can be passed to the object\n"
        "\n:param analysis: analysis name (default unknown)"
        "\n:param xsection: cross section (default nan)"
    )
    return wrapper
