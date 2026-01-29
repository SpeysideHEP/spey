from abc import ABC, abstractmethod
from functools import partial
from typing import Union

import numpy as np

import spey
from spey.multiparameter.utils import inspect_function


class MultiParamTemplate(ABC):
    def __init__(
        self,
        signal_yields: callable,
        number_of_parameters: int,
        background_yields: list[float],
        data: list[int],
    ):
        assert callable(signal_yields), "Signal yields are not callable"
        self.signal_yields = signal_yields
        self.number_of_parameters = number_of_parameters

        # test the output
        test_output = signal_yields(np.random.random(number_of_parameters))
        assert isinstance(
            test_output, np.ndarray
        ), "`signal_yields` should return numpy array"

        assert len(background_yields) == len(
            test_output
        ), "Signal bins does not match with background."
        self.background_yields = background_yields
        self.data = data

    @abstractmethod
    def __call__(self, parameters: np.ndarray) -> spey.StatisticalModel:
        """
        Call method for the template

        Args:
            parameters (``np.ndarray``): parameters to be varied

        Returns:
            ``spey.StatisticalModel``:
            Statistical model to be returned
        """


class MultivariateNormal(MultiParamTemplate):
    def __init__(
        self,
        signal_yields: callable,
        background_yields: list[float],
        data: list[int],
        covariance_matrix: Union[list[list[float]], callable],
        number_of_parameters: int,
    ):
        super().__init__(
            signal_yields=signal_yields,
            number_of_parameters=number_of_parameters,
            background_yields=background_yields,
            data=data,
        )

        self.covariance_matrix = covariance_matrix
        self.cov_inspection = (
            inspect_function(covariance_matrix) if callable(covariance_matrix) else {}
        )
        if callable(covariance_matrix):
            cov_args = self.cov_inspection["args"]
            if len(cov_args) == 2:
                assert (
                    cov_args[-1] == "coefs"
                ), "Second argument of the covariance matrix should be `coefs`"

        pdf_wrapper = spey.get_backend("default.multivariate_normal")

        if self.cov_inspection.get("n_args", 1) == 1:

            def likelihood(parameters: np.ndarray) -> spey.StatisticalModel:
                return pdf_wrapper(
                    signal_yields=self.signal_yields(parameters),
                    background_yields=self.background_yields,
                    data=self.data,
                    covariance_matrix=self.covariance_matrix,
                )

        elif self.cov_inspection.get("n_args", 1) == 2:

            def likelihood(parameters: np.ndarray) -> spey.StatisticalModel:
                return pdf_wrapper(
                    signal_yields=self.signal_yields(parameters),
                    background_yields=self.background_yields,
                    data=self.data,
                    covariance_matrix=partial(self.covariance_matrix, coefs=parameters),
                )

        else:
            raise ValueError("Invalid functional construction")

        self.likelihood = likelihood

    def __call__(self, parameters: np.ndarray) -> spey.StatisticalModel:
        assert len(parameters) == self.number_of_parameters, "Wrong number of parameters"
        return self.likelihood(parameters)
