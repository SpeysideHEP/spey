"""Simplified Likelihood Interface"""

from typing import Optional, Text, Callable, List, Union, Tuple
import numpy as np

from spey.optimizer import fit
from spey.base import BackendBase, DataBase
from spey.utils import ExpectationType
from spey._version import __version__
from .sldata import SLData, expansion_output
from .operators import logpdf, hessian_logpdf_func, objective_wrapper
from .sampler import sample_generator

__all__ = ["SimplifiedLikelihoodInterface"]


class SimplifiedLikelihoodInterface(BackendBase):
    """
    Simplified Likelihood Interface.

    :param model (`SLData`): contains all the information regarding the regions,
                  yields and correlation matrices
    :param ntoys (`int`, default `10000`): number of toy examples to run for the
                    test statistics. Only used for marginalised likelihood.
    :raises AssertionError: if the input type is wrong.
    """

    name: Text = "simplified_likelihoods"
    version: Text = __version__
    author: Text = "SpeysideHEP"
    spey_requires: Text = __version__
    doi: List[Text] = ["10.1007/JHEP04(2019)064"]
    arXiv: List[Text] = ["1809.05548"]
    datastructure = SLData

    __slots__ = ["_model", "ntoys", "_third_moment_expansion", "_asimov_nuisance"]

    def __init__(self, model: SLData, ntoys: int = 10000):
        assert (
            isinstance(model, SLData) and isinstance(model, DataBase) and isinstance(ntoys, int)
        ), "Invalid statistical model."
        self._model = model
        self.ntoys = ntoys
        self._third_moment_expansion: Optional[expansion_output] = None
        self._asimov_nuisance = {
            str(ExpectationType.observed): None,
            str(ExpectationType.apriori): None,
        }

    @property
    def model(self) -> SLData:
        """Get statistical model"""
        return self._model

    @property
    def third_moment_expansion(self) -> expansion_output:
        """Get third moment expansion"""
        if self._third_moment_expansion is None:
            self._third_moment_expansion = self.model.compute_expansion()
        return self._third_moment_expansion

    def get_objective_function(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.ndarray] = None,
        do_grad: bool = True,
    ) -> Callable[[np.ndarray], Union[Tuple[float, np.ndarray], float]]:
        """
        Construct objective function for optimisation

        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori, aposteriori..
        :param data (`Optional[np.ndarray]`, default `None`): observed data to be used for nll computation.
        :param do_grad (`bool`, default `True`): if true include gradient.
        :return `Callable[[np.ndarray], Union[Tuple[float, np.ndarray], float]]`: function to compute objective
                function and its gradient
        """
        current_model: SLData = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )

        return objective_wrapper(
            signal=current_model.signal,
            background=current_model.background,
            data=data if data is not None else current_model.observed,
            third_moment_expansion=self.third_moment_expansion,
            do_grad=do_grad,
        )

    def get_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.array] = None,
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        """
        Generate function to compute twice negative log-likelihood for the statistical model

        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori, aposteriori.
        :param data (`Union[List[float], np.ndarray]`, default `None`): observed data to be used for nll computation.
        :return `Callable[[np.ndarray, np.ndarray], float]`: function to compute twice negative log-likelihood
            for given nuisance parameters.
        """
        current_model: SLData = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )
        return lambda pars: logpdf(
            pars=pars,
            signal=current_model.signal,
            background=current_model.background,
            observed=data or current_model.observed,
            third_moment_expansion=self.third_moment_expansion,
        )

    def get_hessian_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.ndarray] = None,
    ) -> Callable[[np.ndarray], float]:
        """
        Generate function to compute hessian of twice negative log-likelihood for the statistical model.

        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori, aposteriori.
        :param data (`Union[List[float], np.ndarray]`, default `None`): observed data to be used for nll computation.
        :return `Callable[[np.ndarray], float]`: function to compute hessian of twice negative log-likelihood
            for given nuisance parameters.
        """
        current_model: SLData = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )

        hess = hessian_logpdf_func(
            current_model.signal, current_model.background, self.third_moment_expansion
        )

        return lambda pars: hess(pars, data or current_model.observed)

    def get_sampler(self, pars: np.ndarray) -> Callable[[int], np.ndarray]:
        """
        Sampler function predefined with respect to the statistical model yields.

        :param pars (`np.ndarray`): nuisance parameters
        :return `Callable[[int], np.ndarray]`: eturns function to sample from
            a preconfigured statistical model
        """
        return sample_generator(
            pars=pars,
            signal=self.model.signal,
            background=self.model.background,
            third_moment_expansion=self.third_moment_expansion,
        )

    def generate_asimov_data(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> np.ndarray:
        """
        Generate Asimov data with respect to the given test statistics

        :param model (`SLData`): Container for the statistical model properties
        :param test_statistics (`Text`, default `"qtilde"`): test statistics, `q0`, `qtilde`, `q`.
        :return `np.ndarray`: Asimov data
        """
        model: SLData = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )

        # Do not allow asimov data to be negative!
        par_bounds = [(0.0, 1.0)] + [
            (-1 * (bkg + sig * (test_statistics == "q0")), 100.0)
            for sig, bkg in zip(model.signal, model.background)
        ]

        func = objective_wrapper(
            signal=model.signal,
            background=model.background,
            data=model.observed,
            third_moment_expansion=self.third_moment_expansion,
            do_grad=True,
        )

        _, fit_pars = fit(
            func=func,
            model_configuration=model.config(allow_negative_signal=test_statistics in ["q", "qmu"]),
            do_grad=True,
            fixed_poi_value=1.0 if test_statistics == "q0" else 0.0,
            bounds=par_bounds,
            **kwargs,
        )

        return model.background + fit_pars[1:]
