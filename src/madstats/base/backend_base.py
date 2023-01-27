from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Tuple

from madstats.utils import ExpectationType
from madstats.backends import AvailableBackends


__all__ = ["BackendBase"]


class BackendBase(ABC):
    @property
    @abstractmethod
    def type(self) -> AvailableBackends:
        """Type of the backend"""
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
    def computeCLs(
        self,
        mu: float = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        **kwargs,
    ) -> Union[float, Dict]:
        """
        Compute exclusion confidence level.

        :param mu: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param kwargs: backend specific inputs
        :return: 1 - CLs values
        """
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
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
        :return: mu
        """
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
    def likelihood(
        self,
        mu: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        return_nll: Optional[bool] = False,
        isAsimov: Optional[bool] = False,
        **kwargs,
    ) -> float:
        """
        Compute the likelihood of the given statistical model

        :param mu: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, POI can get negative values
        :param return_nll: if true returns negative log-likelihood value
        :param isAsimov: if true, computes likelihood for Asimov data
        :param kwargs: backend specific inputs
        :return: (float) likelihood
        """
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = False,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Find the POI that maximizes the likelihood and the value of the maximum likelihood

        :param return_nll: if true, likelihood will be returned
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: allow negative POI
        :param isAsimov: if true, computes likelihood for Asimov data
        :param kwargs: backend specific inputs
        :return: muhat, maximum of the likelihood
        """
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
    def chi2(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        **kwargs,
    ) -> float:
        """
        Compute $$\chi^2$$

        .. math::

            \chi^2 = -2\log\left(\frac{\mathcal{L}_{\mu = 1}}{\mathcal{L}_{max}}\right)

        :param expected: observed, apriori or aposteriori
        :param marginalise: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :param allow_negative_signal: if true, allow negative mu
        :param isAsimov: if true, computes likelihood for Asimov data
        :return: \chi^2
        """
        raise NotImplementedError("This method has not been implemented")
