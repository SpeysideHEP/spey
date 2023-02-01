from typing import Optional, Text, Tuple, List, Callable
import scipy

import numpy as np

from spey.utils import ExpectationType
from spey.base.backend_base import BackendBase
from spey.backends import AvailableBackends
from spey.system.exceptions import UnknownCrossSection
from spey.hypothesis_testing.test_statistics import compute_teststatistics
from spey.hypothesis_testing.utils import compute_confidence_level, find_poi_upper_limit

__all__ = ["StatisticalModel"]


class StatisticalModel:
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

    def _hypotest_tools(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Callable[[bool], Tuple[float, float]], Callable[[float, bool], float]]:
        """

        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, allow negative mu
        :param kwargs: backend specific arguments
        :return: muhat, minimum logpdf, muhat for asimov dataset, minimum logpdf for asimov dataset
                and logpdf function
        """
        maximize_likelihood = lambda isAsimov: self.maximize_likelihood(
            return_nll=True,
            expected=expected,
            allow_negative_signal=allow_negative_signal,
            isAsimov=isAsimov,
            **kwargs,
        )

        logpdf = lambda mu, isAsimov: -self.likelihood(
            poi_test=mu if isinstance(mu, float) else mu[0],
            expected=expected,
            return_nll=True,
            isAsimov=isAsimov,
            **kwargs,
        )

        return maximize_likelihood, logpdf

    def exclusion_confidence_level(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: bool = True,
        **kwargs,
    ) -> List[float]:
        """
        Compute exclusion confidence level of a given statistical model.

        :param expected: observed, apriori or aposteriori
        :param kwargs: backend specific inputs.
        :return: 1-CLs value (float)
        """
        if hasattr(self.backend, "exclusion_confidence_level"):
            return self.backend.exclusion_confidence_level(
                poi_test=1.0,
                expected=expected,
                allow_negative_signal=allow_negative_signal,
                **kwargs,
            )
        else:
            maximize_likelihood, logpdf = self._hypotest_tools(
                expected=expected, allow_negative_signal=allow_negative_signal, **kwargs
            )
            test_stat = "q" if allow_negative_signal else "qtilde"
            _, sqrt_qmuA, delta_teststat = compute_teststatistics(
                1.0, maximize_likelihood, logpdf, test_stat
            )
            pvalue = list(
                map(
                    lambda x: 1.0 - x, compute_confidence_level(sqrt_qmuA, delta_teststat, expected)
                )
            )
            return pvalue

    def poi_upper_limit(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: bool = True,
        confidence_level: float = 0.95,
        **kwargs,
    ) -> float:
        """
        Compute the upper limit on parameter of interest, described by the confidence level

        :param expected: observed, apriori or aposteriori
        :param confidence_level: exclusion confidence level (default 1 - CLs = 95%)
        :param kwargs: backend specific inputs.
        :return: excluded parameter of interest
        """
        assert 0.0 <= confidence_level <= 1.0, "Confidence level must be between zero and one."
        if hasattr(self.backend, "poi_upper_limit"):
            return self.backend.poi_upper_limit(
                expected=expected,
                allow_negative_signal=allow_negative_signal,
                confidence_level=confidence_level,
                **kwargs,
            )
        else:
            maximize_likelihood, logpdf = self._hypotest_tools(
                expected=expected, allow_negative_signal=allow_negative_signal, **kwargs
            )

            return find_poi_upper_limit(
                maximize_likelihood=maximize_likelihood,
                logpdf=logpdf,
                expected=expected,
                sigma_mu=1.0,
                confidence_level=confidence_level,
                allow_negative_signal=allow_negative_signal,
            )
