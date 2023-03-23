"""Simplified Likelihood Interface"""

from typing import Optional, Tuple, Text, Callable
import numpy as np

from spey.optimizer import fit
from spey.base import BackendBase, DataBase
from spey.utils import ExpectationType
from spey._version import __version__
from .sldata import SLData, expansion_output
from .utils import twice_nll_func, gradient_twice_nll_func
from .utils_marginalised import marginalised_negloglikelihood

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

    name = "simplified_likelihoods"
    version = __version__
    author = "SpeysideHEP"
    spey_requires = __version__
    doi = ["10.1007/JHEP04(2019)064"]
    arXiv = ["1809.05548"]
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

    def get_twice_nll_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.ndarray] = None,
    ) -> Callable[[np.ndarray], float]:
        """
        Generate function to compute twice negative log-likelihood for the statistical model

        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori, aposteriori.
        :param data (`Union[List[float], np.ndarray]`, default `None`): observed data to be used for nll computation.
        :return `Callable[[np.ndarray], float]`: function to compute twice negative log-likelihood for given nuisance parameters.
        """
        current_model: SLData = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )
        return twice_nll_func(
            current_model.signal,
            current_model.background,
            data if data is not None else current_model.observed,
            self.third_moment_expansion,
        )

    def get_gradient_twice_nll_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.ndarray] = None,
    ) -> Callable[[np.ndarray], float]:
        """
        Generate function to compute gradient of twice negative log-likelihood for the statistical model.

        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori, aposteriori.
        :param data (`Union[List[float], np.ndarray]`, default `None`): observed data to be used for nll computation.
        :return `Callable[[np.ndarray], float]`: function to compute twice negative log-likelihood for given nuisance parameters.
        """
        current_model: SLData = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )
        return gradient_twice_nll_func(
            current_model.signal,
            current_model.background,
            data if data is not None else current_model.observed,
            self.third_moment_expansion,
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
        # asimov_nuisance_key = (
        #     str(ExpectationType.apriori)
        #     if expected == ExpectationType.apriori
        #     else str(ExpectationType.observed)
        # )
        # fit_pars = self._asimov_nuisance.get(asimov_nuisance_key, None)
        # if fit_pars is None:
        # Generate the asimov data by fittin nuissance parameters to the observations
        model: SLData = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )

        # Do not allow asimov data to be negative!
        par_bounds = [(0.0, 1.0)] + [
            (-1 * (bkg + sig * (test_statistics == "q0")), 100.0)
            for sig, bkg in zip(model.signal, model.background)
        ]

        _, fit_pars = fit(
            func=twice_nll_func(
                model.signal, model.background, model.observed, self.third_moment_expansion
            ),
            model_configuration=model.config(allow_negative_signal=test_statistics in ["q", "qmu"]),
            gradient=gradient_twice_nll_func(
                model.signal, model.background, model.observed, self.third_moment_expansion
            ),
            fixed_poi_value=1.0 if test_statistics == "q0" else 0.0,
            bounds=par_bounds,
            **kwargs,
        )
        # self._asimov_nuisance[asimov_nuisance_key] = fit_pars

        return model.background + fit_pars[1:]

    def negative_loglikelihood(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        marginalize: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute the likelihood for the statistical model with a given POI

        :param poi_test (`Optional[float]`, default `1.0`): POI (signal strength).
        :param expected (`Optional[ExpectationType]`, default `ExpectationType.observed`): observed, apriori or aposteriori.
        :param marginalize (`Optional[bool]`, default `False`): if true, marginalize the likelihood.
                            if false compute profiled likelihood.
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :return `Tuple[float, np.ndarray]`: negative log-likelihood, fit parameters
        """

        if marginalize:
            current_model: SLData = (
                self.model if expected != ExpectationType.apriori else self.model.expected_dataset
            )

            nll = marginalised_negloglikelihood(
                poi_test, current_model, self.third_moment_expansion, self.ntoys
            )
            return nll, np.nan

        # If not marginalised then use default computation method
        raise NotImplementedError("This method has not been implemented")
