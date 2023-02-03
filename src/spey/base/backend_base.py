from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

import numpy as np

from spey.utils import ExpectationType
from spey.backends import AvailableBackends


__all__ = ["BackendBase", "DataBase"]


class DataBase(ABC):
    """
    Data base is a class that keeps track of the input data which then wrapped with a statistical
    model class which is based on BackendBase class.
    """

    @property
    @abstractmethod
    def minimum_poi_test(self) -> float:
        """Find minimum POI test that can be applied to this statistical model"""
        # This method must be casted as property
        raise NotImplementedError("This method has not been implemented")

    @property
    @abstractmethod
    def poi_index(self) -> int:
        """Return the index of the parameter of interest withing nuisance parameters"""
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
        isAsimov: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute the likelihood of the given statistical model

        :param poi_test: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, POI can get negative values
        :param isAsimov: if true, computes likelihood for Asimov data
        :param kwargs: backend specific inputs
        :return: (float) likelihood
        """
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
    def maximize_likelihood(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[float, np.ndarray, float]:
        """
        Find the POI that maximizes the likelihood and the value of the maximum likelihood

        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: allow negative POI
        :param isAsimov: if true, computes likelihood for Asimov data
        :param kwargs: backend specific inputs
        :return: muhat, maximum of the likelihood
        """
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
    def sigma_mu(
        self, pars: np.ndarray, expected: Optional[ExpectationType] = ExpectationType.observed
    ) -> float:
        """Compute uncertainty on parameter of interest"""
        raise NotImplementedError("This method has not been implemented")
