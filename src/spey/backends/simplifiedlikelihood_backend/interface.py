from typing import Optional, Tuple, Text, List
import numpy as np

from spey.optimizer import fit
from spey.base import BackendBase, DataBase
from spey.utils import ExpectationType
from spey.backends import AvailableBackends
from spey.interface.statistical_model import statistical_model_wrapper
from .sldata import SLData, expansion_output
from .utils import twice_nll_func, gradient_twice_nll_func
from .utils_marginalised import marginalised_negloglikelihood

__all__ = ["SimplifiedLikelihoodInterface"]


@statistical_model_wrapper
class SimplifiedLikelihoodInterface(BackendBase):
    """
    Simplified Likelihood Interface. This is object has been wrapped with
    `StatisticalModel` class to ensure universality across all platforms.

    :param model: contains all the information regarding the regions,
                  yields and correlation matrices
    :param ntoys: number of toy examples to run for the test statistics
    :param analysis: a unique name for the analysis (default `"__unknown_analysis__"`)
    :param xsection: cross section value for the signal, only used to compute excluded cross section
                     value. Default `NaN`
    :raises AssertionError: if the input type is wrong.
    """

    __slots__ = ["_model", "ntoys", "_third_moment_expansion", "_asimov_nuisance"]

    def __init__(self, model: SLData, ntoys: Optional[int] = 10000):
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
    def type(self) -> AvailableBackends:
        return AvailableBackends.simplified_likelihoods

    @property
    def third_moment_expansion(self) -> expansion_output:
        """Get third moment expansion"""
        if self._third_moment_expansion is None:
            self._third_moment_expansion = self.model.compute_expansion()
        return self._third_moment_expansion

    def generate_asimov_data(
        self,
        model: SLData,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        test_statistics: Text = "qtilde",
    ) -> np.ndarray:
        """
        Generate asimov data

        :param model (`SLData`): simplified likelihood model
        :param expected (`Optional[ExpectationType]`, default `ExpectationType.observed`): observed, apriori or aposteriori.
                                                   defaults to ExpectationType.observed.
        :param test_statistics (`Text`, default `"qtilde"`): `"qmu"` or `"qtilde"` for exclusion
                                     tests `"q0"` for discovery test, defaults to `"qtilde"`..
        :return `np.ndarray`: asimov data
        """
        asimov_nuisance_key = (
            str(ExpectationType.apriori)
            if expected == ExpectationType.apriori
            else str(ExpectationType.observed)
        )
        pars = self._asimov_nuisance.get(asimov_nuisance_key, None)
        # NOTE for test_stat = q0 asimov mu should be 1, default qtilde!!!
        if pars is None:
            # Generate the asimov data by fittin nuissance parameters to the observations
            init_pars = [0.0] * (len(model) + 1)
            par_bounds = [(model.minimum_poi, 1.0)] + [(-5.0, 5.0)] * len(model)
            _, pars = fit(
                model,
                init_pars,
                par_bounds,
                1.0 if test_statistics == "q0" else 0.0,
                self.third_moment_expansion,
            )
            self._asimov_nuisance[asimov_nuisance_key] = pars

        return model.background + pars[1:]

    # def logpdf(
    #     self,
    #     nuisance_parameters: np.ndarray,
    #     expected: Optional[ExpectationType] = ExpectationType.observed,
    #     isAsimov: Optional[bool] = False,
    # ) -> float:
    #     """
    #     Compute the log value of the full density.

    #     :param nuisance_parameters: nuisance parameters
    #     :param expected: observed, apriori or aposteriori
    #     :param isAsimov: if true, computes likelihood for Asimov data
    #     :return: negative log-likelihood
    #     """
    #     current_model: SLData = (
    #         self.model if expected != ExpectationType.apriori else self.model.expected_dataset
    #     )
    #     if isAsimov:
    #         current_model = self._get_asimov_data(current_model, expected)

    #     return -0.5 * twice_nll(
    #         nuisance_parameters,
    #         signal=current_model.signal,
    #         background=current_model.background,
    #         observed=current_model.observed,
    #         third_moment_expansion=self.third_moment_expansion,
    #     )

    def negative_loglikelihood(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        marginalize: Optional[bool] = False,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
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
        current_model: SLData = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )

        if marginalize:
            nll = marginalised_negloglikelihood(
                poi_test, current_model, self.third_moment_expansion, self.ntoys
            )
            return nll, np.nan

        twice_nll, fit_param = fit(
            func=twice_nll_func(
                current_model.signal,
                current_model.background,
                current_model.observed,
                self.third_moment_expansion,
            ),
            model_configuration=current_model.config(),
            gradient=gradient_twice_nll_func(
                current_model.signal,
                current_model.background,
                current_model.observed,
                self.third_moment_expansion,
            ),
            initial_parameters=init_pars,
            bounds=par_bounds,
            fixed_poi_value=poi_test,
            **kwargs,
        )

        return twice_nll / 2.0, fit_param

    def asimov_negative_loglikelihood(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        compute likelihood for the Asimov data

        :param poi_test (`Optional[float]`, default `1.0`): parameter of interest.
        :param expected (`Optional[ExpectationType]`, default `ExpectationType.observed`): observed, apriori or aposteriori.
        :param test_statistics (`Text`, default `"qtilde"`): test statistics: `q`, `qtilde`, `q0`.
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :return `Tuple[float, np.ndarray]`: negative log-likelihood, fit parameters
        """
        current_model: SLData = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )
        data = self.generate_asimov_data(
            current_model, expected=expected, test_statistics=test_statistics
        )

        twice_nll, fit_param = fit(
            func=twice_nll_func(
                current_model.signal,
                current_model.background,
                data,
                self.third_moment_expansion,
            ),
            model_configuration=current_model.config(),
            gradient=gradient_twice_nll_func(
                current_model.signal,
                current_model.background,
                data,
                self.third_moment_expansion,
            ),
            initial_parameters=init_pars,
            bounds=par_bounds,
            fixed_poi_value=poi_test,
            **kwargs,
        )

        return twice_nll / 2.0, fit_param

    def minimize_negative_loglikelihood(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Minimize negative log-likelihood of the statistical model with respect to POI

        :param expected (`Optional[ExpectationType]`, default `ExpectationType.observed`): observed, apriori or aposteriori.
        :param allow_negative_signal (`Optional[bool]`, default `True`): if true, allow negative mu.
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :return `Tuple[float, np.ndarray]`: minimum negative log-likelihood and fit parameters
        """
        current_model: SLData = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )

        twice_nll, fit_param = fit(
            func=twice_nll_func(
                current_model.signal,
                current_model.background,
                current_model.observed,
                self.third_moment_expansion,
            ),
            model_configuration=current_model.config(allow_negative_signal=allow_negative_signal),
            gradient=gradient_twice_nll_func(
                current_model.signal,
                current_model.background,
                current_model.observed,
                self.third_moment_expansion,
            ),
            initial_parameters=init_pars,
            bounds=par_bounds,
            **kwargs,
        )

        return twice_nll / 2.0, fit_param

    def minimize_asimov_negative_loglikelihood(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute maximum likelihood for asimov data

        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori or aposteriori.
        :param test_statistics (`Text`, default `"qtilde"`): test statistics. `"qmu"` or `"qtilde"` for exclusion
                                     tests `"q0"` for discovery test.
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :return `Tuple[float, np.ndarray]`: minimum negative log-likelihood and fit parameters
        """
        allow_negative_signal: bool = True if test_statistics in ["q", "qmu"] else False
        current_model: SLData = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )
        data = self.generate_asimov_data(
            current_model, expected=expected, test_statistics=test_statistics
        )

        twice_nll, fit_param = fit(
            func=twice_nll_func(
                current_model.signal,
                current_model.background,
                data,
                self.third_moment_expansion,
            ),
            model_configuration=current_model.config(allow_negative_signal=allow_negative_signal),
            gradient=gradient_twice_nll_func(
                current_model.signal,
                current_model.background,
                data,
                self.third_moment_expansion,
            ),
            initial_parameters=init_pars,
            bounds=par_bounds,
            **kwargs,
        )

        return twice_nll / 2.0, fit_param
