from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Tuple, List
import numpy as np

from madstats.utils import ExpectationType


class BackendBase(ABC):
    def __init__(
        self, signal: Union[List, float, np.ndarray], background: Union[Dict, float, np.ndarray]
    ):
        self.signal = signal
        self.background = background

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
        :param expected: observed, expected (true, apriori) or aposteriori
        :param kwargs: backend specific inputs
        :return: CLs values {"CLs_obs": xx, "CLs_exp": [xx] * 5} or single CLs value
        """
        raise NotImplemented("This method has not been implemented")

    @abstractmethod
    def computeUpperLimitOnMu(
        self, expected: Optional[ExpectationType] = ExpectationType.observed
    ) -> float:
        """
        Compute the POI where the signal is excluded with 95% CL

        :param expected: observed, expected (true, apriori) or aposteriori
        :return: mu
        """
        raise NotImplemented("This method has not been implemented")

    @abstractmethod
    def likelihood(
        self,
        mu: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: bool = False,
        return_nll: Optional[bool] = False,
        isAsimov: Optional[bool] = False,
        **kwargs,
    ) -> float:
        """
        Compute the likelihood of the given statistical model

        :param mu: POI (signal strength)
        :param expected: observed, expected (true, apriori) or aposteriori
        :param allow_negative_signal: if true, POI can get negative values
        :param return_nll: if true returns negative log-likelihood value
        :param isAsimov: if true, computes likelihood for Asimov data
        :param kwargs: backend specific inputs
        :return: (float) likelihood
        """
        raise NotImplemented("This method has not been implemented")

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
        :param expected: observed, expected (true, apriori) or aposteriori
        :param allow_negative_signal: allow negative POI
        :param isAsimov: if true, computes likelihood for Asimov data
        :param kwargs: backend specific inputs
        :return: muhat, maximum of the likelihood
        """
        raise NotImplemented("This method has not been implemented")
