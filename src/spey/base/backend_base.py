"""Abstract Methods for backend objects"""

from abc import ABC, abstractmethod
from typing import Text, Tuple, Callable, Union, List, Optional

import numpy as np

from spey.utils import ExpectationType
from .model_config import ModelConfig


__all__ = ["BackendBase", "DataBase"]


class DataBase(ABC):
    """
    Data base is a class that keeps track of the input data which then wrapped with a statistical
    model class which is based on BackendBase class.
    """

    @property
    @abstractmethod
    def minimum_poi(self) -> float:
        """Find minimum POI test that can be applied to this statistical model"""
        # This method must be casted as property

    @property
    @abstractmethod
    def isAlive(self) -> bool:
        """Does the statitical model has any non-zero signal events?"""
        # This method has to be a property

    @abstractmethod
    def config(
        self, allow_negative_signal: bool = True, poi_upper_bound: float = 40.0
    ) -> ModelConfig:
        """
        Configuration of the statistical model

        :param allow_negative_signal (`bool`, default `True`): if the negative POI is allowed during fits.
        :param poi_upper_bound (`float`, default `40.0`): sets the upper bound for POI
        :return `ModelConfig`: Configuration information of the model.
        """


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

    def get_twice_nll_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Callable[[np.ndarray], float]:
        """
        Generate function to compute twice negative log-likelihood for the statistical model
        Interface will first look for default likelihood computers that defined for the backend. If its not
        defined then it will call this function.

        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori, aposteriori.
        :param data (`Union[List[float], np.ndarray]`, default `None`): observed data to be used for nll computation.
        :raises `NotImplementedError`: If the method is not implemented
        :return `Callable[[np.ndarray], float]`: function to compute twice negative log-likelihood for given nuisance parameters.
        """
        raise NotImplementedError("This method has not been implemented")

    def get_gradient_twice_nll_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Callable[[np.ndarray], float]:
        """
        Generate function to compute gradient of twice negative log-likelihood for the statistical model.
        Interface will first look for default likelihood computers that defined for the backend. If its not
        defined then it will call this function.

        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori, aposteriori.
        :param data (`Union[List[float], np.ndarray]`, default `None`): observed data to be used for nll computation.
        :raises `NotImplementedError`: If the method is not implemented
        :return `Callable[[np.ndarray], float]`: function to compute twice negative log-likelihood for given nuisance parameters.
        """
        return None

    @abstractmethod
    def generate_asimov_data(
        self, expected: ExpectationType = ExpectationType.observed, test_statistics: Text = "qtilde", **kwargs
    ) -> Union[List[float], np.ndarray]:
        """
        Method to generate Asimov data for given statistical model

        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori, aposteriori.
        :param test_statistics (`Text`, default `"qtilde"`): definition of test statistics. `q`, `qtilde` or `q0`
        :raises `NotImplementedError`: if the method has not been implemented
        :return ` Union[List[float], np.ndarray]`: Asimov data
        """

    def negative_loglikelihood(
        self, poi_test: float = 1.0, expected: ExpectationType = ExpectationType.observed, **kwargs
    ) -> Tuple[float, np.ndarray]:
        """
        Negative log-likelihood computer. Interface will initially call this method.

        :param poi_test (`float`, default `1.0`): parameter of interest (signal strength).
        :param expected (`ExpectationType`, default `ExpectationType.observed`): expectation type, observed, apriori, aposteriori.
        :raises `NotImplementedError`: if the method has not been implemented
        :return `Tuple[float, np.ndarray]`: negative log-likelihood value and fit parameters
        """
        raise NotImplementedError("This method has not been implemented")

    def asimov_negative_loglikelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Negative log-likelihood computer for Asimov data. Interface will initially call this method.

        :param poi_test (`float`, default `1.0`): parameter of interest (signal strength).
        :param expected (`ExpectationType`, default `ExpectationType.observed`): expectation type, observed, apriori, aposteriori.
        :param test_statistics (`Text`, default `"qtilde"`): definition of test statistics. `q`, `qtilde` or `q0`
        :raises `NotImplementedError`: if the method has not been implemented
        :return `Tuple[float, np.ndarray]`: negative log-likelihood value and fit parameters
        """
        raise NotImplementedError("This method has not been implemented")

    def minimize_negative_loglikelihood(
        self,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = True,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Find nuisance parameters that maximizes the likelihood. Interface will initially call this method.

        :param expected (`ExpectationType`, default `ExpectationType.observed`): expectation type, observed, apriori, aposteriori.
        :param allow_negative_signal (`bool`, default `True`): If true negative signal values will be allowed.
        :raises `NotImplementedError`: if the method has not been implemented
        :return `Tuple[float, np.ndarray]`: negative log-likelihood value and fit parameters
        """
        raise NotImplementedError("This method has not been implemented")

    def minimize_asimov_negative_loglikelihood(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Find nuisance parameters that maximizes the likelihood for Asimov data.
        Interface will initially call this method.

        :param expected (`ExpectationType`, default `ExpectationType.observed`): expectation type, observed, apriori, aposteriori.
        :param test_statistics (`Text`, default `"qtilde"`): definition of test statistics. `q`, `qtilde` or `q0`
        :raises `NotImplementedError`: if the method has not been implemented
        :return `Tuple[float, np.ndarray]`: negative log-likelihood value and fit parameters
        """
        raise NotImplementedError("This method has not been implemented")
