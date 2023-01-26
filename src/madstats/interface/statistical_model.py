from typing import Optional, Text, Tuple

import numpy as np

from madstats.utils import ExpectationType
from madstats.base.backend_base import BackendBase
from madstats.backends import AvailableBackends
from madstats.system.exceptions import UnknownCrossSection


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
        Compute excluded cross section

        :param expected: observed, apriori or aposteriori
        :return: excluded cross section value in pb
        :raises UnknownCrossSection: if cross section is nan.
        """
        if np.isnan(self.xsection):
            raise UnknownCrossSection("Cross-section value has not been initialised.")

        return self.computeUpperLimitOnMu(expected=expected) * self.xsection

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
        mu: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        return_nll: Optional[bool] = False,
        **kwargs,
    ) -> float:
        """
        Compute the likelihood of the given statistical model

        :param mu: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param return_nll: if true returns negative log-likelihood value
        :param kwargs: backend specific inputs.
        :return: (float) likelihood
        """
        return self.backend.likelihood(mu=mu, expected=expected, return_nll=return_nll, **kwargs)

    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = False,
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

    def exclusion_confidence_level(
        self, expected: Optional[ExpectationType] = ExpectationType.observed, **kwargs
    ) -> float:
        """
        Compute exclusion confidence level of a given statistical model.

        :param expected: observed, apriori or aposteriori
        :param kwargs: backend specific inputs.
        :return: 1-CLs value (float)
        """
        if self.backend_type is AvailableBackends.pyhf:
            kwargs.update(
                dict(
                    CLs_obs=expected in [ExpectationType.apriori, ExpectationType.observed],
                    CLs_exp=expected == ExpectationType.aposteriori,
                )
            )
        return self.backend.computeCLs(mu=1.0, expected=expected, **kwargs)

    def computeUpperLimitOnMu(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        confidence_level: float = 0.95,
        **kwargs,
    ) -> float:
        """
        Compute the POI where the signal is excluded with 95% CL

        :param expected: observed, apriori or aposteriori
        :param confidence_level: confidence level (default 95%)
        :param kwargs: backend specific inputs.
        :return: mu
        """
        return self.backend.computeUpperLimitOnMu(
            expected=expected, confidence_level=confidence_level, **kwargs
        )