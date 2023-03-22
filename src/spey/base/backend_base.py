"""Abstract Methods for backend objects"""

from abc import ABC, abstractmethod
from typing import Text, Tuple, Callable, Union

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

    @abstractmethod
    def negative_loglikelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute the likelihood of the given statistical model

        :param poi_test: POI (signal strength)
        :param ex∏pected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, POI can get negative values
        :param isAsimov: if true, computes likelihood for Asimov data
        :param kwargs: backend specific inputs
        :return: (float) likelihood
        """

    @abstractmethod
    def asimov_negative_loglikelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute likelihood for the asimov data

        :param float poi_test: parameter of interest, defaults to 1.0
        :param ExpectationType expected: observed, apriori or aposteriori.
                                                   defaults to ExpectationType.observed
        :param Text test_statistics: test statistics. `"qmu"` or `"qtilde"` for exclusion
                                     tests `"q0"` for discovery test, defaults to `"qtilde"`.
        :return Tuple[float, np.ndarray]: negative log-likelihood, fit parameters
        """

    @abstractmethod
    def minimize_negative_loglikelihood(
        self,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = True,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Find the POI that maximizes the likelihood and the value of the maximum likelihood

        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: allow negative POI
        :param kwargs: backend specific inputs
        :return: muhat, maximum of the likelihood
        """

    @abstractmethod
    def minimize_asimov_negative_loglikelihood(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute maximum likelihood for asimov data

        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori or aposteriori.
        :param test_statistics (`Text`, default `"qtilde"`): test statistics. `"qmu"` or `"qtilde"` for exclusion
                                     tests `"q0"` for discovery test, defaults to `"qtilde"`.
        :return `Tuple[float, np.ndarray]`: maximum negative log-likelihood, fit parameters
        """
