from abc import ABC, abstractmethod
from typing import Optional

from madstats.utils import ExpectationType


class StatisticalModelBase(ABC):
    @abstractmethod
    def computeCLs(self, expected: Optional[ExpectationType] = ExpectationType.observed) -> float:
        """
        Compute exclusion confidence level of a given statistical model.

        :param expected: observed, expected (true, apriori) or aposteriori
        :return: 1-CLs value (float)
        """
        raise NotImplemented("This method has not been implemented.")

    @abstractmethod
    def likelihood(
        self,
        mu: Optional[float] = 1.0,
        expected: Optional[bool] = False,
        allow_negative_signal: Optional[bool] = False,
        return_nll: Optional[bool] = False,
        isAsimov: Optional[bool] = False,
    ) -> float:
        """
        Compute the likelihood of the given statistical model

        :param mu: POI (signal strength)
        :param expected: observed, expected (true, apriori) or aposteriori
        :param allow_negative_signal: if true, POI can get negative values
        :param return_nll: if true returns negative log-likelihood value
        :param isAsimov: if true, computes likelihood for Asimov data
        :return: (float) likelihood
        """
        raise NotImplemented("This method has not been implemented.")

    @abstractmethod
    def computeUpperLimitOnMu(
        self, expected: Optional[ExpectationType] = ExpectationType.observed
    ) -> float:
        """
        Compute the POI where the signal is excluded with 95% CL

        :param expected: observed, expected (true, apriori) or aposteriori
        :return: mu
        """
        raise NotImplemented("This method has not been implemented.")
