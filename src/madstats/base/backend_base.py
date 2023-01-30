from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

from madstats.utils import ExpectationType
from madstats.backends import AvailableBackends


__all__ = ["BackendBase", "DataBase"]


class DataBase(ABC):
    """
    Data base is a class that keeps track of the input data which then wrapped with a statistical
    model class which is based on BackendBase class.
    """
    @abstractmethod
    def minimum_poi_test(self) -> float:
        """Find minimum POI test that can be applied to this statistical model"""
        # This method must be casted as property
        raise NotImplementedError("This method has not been implemented")


class BackendBase(ABC):
    """
    Standard base for Statistical model which includes all the required functions to be able to work
    through out the software. This ensures the uniformity of the backends that will be included in
    the future.
    """

    @property
    @abstractmethod
    def model(self) -> DataBase:
        """Get statistical model"""
        # This method must be casted as property
        raise NotImplementedError("This method has not been implemented")

    @property
    @abstractmethod
    def type(self) -> AvailableBackends:
        """Type of the backend"""
        # This method must be casted as property
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
    def likelihood(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        return_nll: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        **kwargs,
    ) -> float:
        """
        Compute the likelihood of the given statistical model

        :param poi_test: POI (signal strength)
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
        return_nll: Optional[bool] = True,
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
        :param allow_negative_signal: if true, allow negative mu
        :param isAsimov: if true, computes likelihood for Asimov data
        :return: \chi^2
        """
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
    def exclusion_confidence_level(
        self,
        poi_test: float = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        **kwargs,
    ) -> List[float]:
        """
        Compute exclusion confidence level.

        :param poi_test: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, allow negative mu
        :param kwargs: backend specific inputs
        :return: 1 - CLs values
        """
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
    def poi_upper_limit(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        confidence_level: float = 0.95,
        allow_negative_signal: Optional[bool] = True,
        **kwargs,
    ) -> float:
        """
        Compute the POI where the signal is excluded with 95% CL

        :param expected: observed, apriori or aposteriori
        :param confidence_level: confidence level (default 95%)
        :param allow_negative_signal: if true, allow negative mu
        :return: POI
        """
        raise NotImplementedError("This method has not been implemented")

