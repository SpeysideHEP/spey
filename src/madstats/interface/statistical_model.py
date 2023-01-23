from typing import Optional, Text, Any

from madstats.utils import ExpectationType
from madstats.base.backend_base import BackendBase
from madstats.backends import AvailableBackends
from madstats.system.exceptions import FrozenInstanceError


class StatisticalModel:
    """
    Statistical model base

    :param backend: Statistical model backend
    :param xsection: Cross-section in pb
    :param analysis: name of the analysis
    """

    __slots__ = "_backend", "xsection", "analysis"

    def __init__(self, backend: BackendBase, xsection: float, analysis: Text):
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
        return self.computeUpperLimitOnMu(expected=expected) * self.xsection

    @property
    def s95exp(self) -> float:
        """Expected excluded cross-section (apriori)"""
        return self.excluded_cross_section(ExpectationType.apriori)

    @property
    def s95obs(self) -> float:
        """Observed excluded cross-section"""
        return self.excluded_cross_section(ExpectationType.observed)

    def exclusion_confidence_level(
        self, expected: Optional[ExpectationType] = ExpectationType.observed, **kwargs
    ) -> float:
        """
        Compute exclusion confidence level of a given statistical model.

        :param expected: observed, expected (true, apriori) or aposteriori
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
        :param expected: observed, expected (true, apriori) or aposteriori
        :param return_nll: if true returns negative log-likelihood value
        :param kwargs: backend specific inputs.
        :return: (float) likelihood
        """
        return self.backend.likelihood(mu=mu, expected=expected, return_nll=return_nll, **kwargs)

    def computeUpperLimitOnMu(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        **kwargs,
    ) -> float:
        """
        Compute the POI where the signal is excluded with 95% CL

        :param expected: observed, expected (true, apriori) or aposteriori
        :param kwargs: backend specific inputs.
        :return: mu
        """
        return self.backend.computeUpperLimitOnMu(expected=expected, **kwargs)
