"""Statistical Model wrapper class"""

from typing import Optional, Text, Tuple, List, Callable, Union

import numpy as np

from spey.utils import ExpectationType
from spey.base.backend_base import BackendBase, DataBase
from spey.system.exceptions import UnknownCrossSection, MethodNotAvailable
from spey.base.hypotest_base import HypothesisTestingBase
from spey.optimizer.core import fit
from .functiontools import get_function

__all__ = ["StatisticalModel", "statistical_model_wrapper"]


class StatisticalModel(HypothesisTestingBase):
    """
    Statistical model base

    :param backend (`BackendBase`): Statistical model backend
    :param analysis (`Text`): Unique identifier of the statistical model
    :param xsection (`float`, default `np.nan`): cross section, unit is determined by the user
    """

    __slots__ = ["_backend", "xsection", "analysis"]

    def __init__(self, backend: BackendBase, analysis: Text, xsection: float = np.nan):
        assert isinstance(backend, BackendBase), "Invalid backend"
        self._backend: BackendBase = backend
        self.xsection: float = xsection
        self.analysis: Text = analysis

    def __repr__(self):
        return (
            f"StatisticalModel(analysis='{self.analysis}', "
            f"xsection={self.xsection:.3e} [pb], "
            f"backend={str(self.backend_type)})"
        )

    @property
    def backend(self) -> BackendBase:
        """Get backend"""
        return self._backend

    @property
    def backend_type(self) -> Text:
        """Return type of the backend"""
        return self.backend.name

    @property
    def isAlive(self) -> bool:
        """Is the statistical model has non-zero signal yields in any region"""
        return self.backend.model.isAlive

    def excluded_cross_section(
        self, expected: Optional[ExpectationType] = ExpectationType.observed
    ) -> float:
        """
        Compute excluded cross section at 95% CLs

        :param expected: observed, apriori or aposteriori
        :return: excluded cross section value in pb
        :raises UnknownCrossSection: if cross section is nan.
        """
        if np.isnan(self.xsection):
            raise UnknownCrossSection("Cross-section value has not been initialised.")

        return self.poi_upper_limit(expected=expected, confidence_level=0.95) * self.xsection

    @property
    def s95exp(self) -> float:
        """Expected excluded cross-section (apriori)"""
        return self.excluded_cross_section(ExpectationType.apriori)

    @property
    def s95obs(self) -> float:
        """Observed excluded cross-section"""
        return self.excluded_cross_section(ExpectationType.observed)

    def fixed_poi_fit(
        self,
        poi_test: float = 1.0,
        data: Optional[Union[List[float], np.ndarray]] = None,
        expected: ExpectationType = ExpectationType.observed,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        """
        Find the minimum of the negative log-likelihood for given parameter of interest.

        :param poi_test (`float`, default `1.0`): POI (signal strength).
        :param data (`Union[List[float], np.ndarray]`, default `None`): observed data to be used for nll computation.
            If not provided, observations that are defined in the statistical model definition will be used. Possible
            usecase of keyword data is to compute Asimov likelihood which has been achieved via `asimov_likelihood` function.
        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori or aposteriori.
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :param kwargs: keyword arguments for optimiser
        :return `float`: (float) negative log-likelihood value and fit parameters
        """
        twice_nll, fit_param = fit(
            func=self.backend.get_twice_nll_func(expected=expected, data=data),
            model_configuration=self.backend.model.config(),
            gradient=get_function(
                self.backend, "get_gradient_twice_nll_func", expected=expected, data=data
            ),
            # hessian=get_function(
            #     self.backend,
            #     "get_hessian_twice_nll_func",
            #     default=None,
            #     expected=expected,
            #     data=data,
            # ),
            initial_parameters=init_pars,
            bounds=par_bounds,
            fixed_poi_value=poi_test,
            **kwargs,
        )
        negloglikelihood = twice_nll / 2.0

        return negloglikelihood, fit_param

    def likelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> float:
        """
        Compute the likelihood of the given statistical model

        :param poi_test (`float`, default `1.0`): POI (signal strength).
        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori or aposteriori.
        :param return_nll (`bool`, default `True`): if true returns negative log-likelihood value.
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :param kwargs: keyword arguments for optimiser
        :return `float`: (float) likelihood or negative log-likelihood value for a given POI test
        """
        try:
            negloglikelihood, _ = self.backend.negative_loglikelihood(
                poi_test=poi_test,
                expected=expected,
                **kwargs,
            )
        except NotImplementedError:
            # add a debug message here
            _ = kwargs.pop("marginalize", False)
            negloglikelihood, _ = self.fixed_poi_fit(
                poi_test=poi_test,
                expected=expected,
                init_pars=init_pars,
                par_bounds=par_bounds,
                **kwargs,
            )

        return negloglikelihood if return_nll else np.exp(-negloglikelihood)

    def asimov_likelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        test_statistics: Text = "qtilde",
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> float:
        """
        Compute likelihood for the asimov data

        :param poi_test (`float`): parameter of interest. (default `1.0`)
        :param expected (`ExpectationType`): observed, apriori or aposteriori.
                                             (default `ExpectationType.observed`)
        :param return_nll (`bool`): if false returns likelihood value. (default `True`)
        :param test_statistics (`Text`): test statistics. `"qmu"` or `"qtilde"` for exclusion
                                     tests `"q0"` for discovery test. (default `"qtilde"`)
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :param kwargs: keyword arguments for optimiser
        :return float: likelihood computed for asimov data
        """
        try:
            negloglikelihood, _ = self.backend.asimov_negative_loglikelihood(
                poi_test=poi_test,
                expected=expected,
                test_statistics=test_statistics,
                **kwargs,
            )
        except NotImplementedError:
            # add a debug logger here saying backend has no implementation etc.
            data = self.backend.generate_asimov_data(
                expected=expected, test_statistics=test_statistics
            )
            negloglikelihood, _ = self.fixed_poi_fit(
                poi_test=poi_test,
                data=data,
                expected=expected,
                init_pars=init_pars,
                par_bounds=par_bounds,
                **kwargs,
            )

        return negloglikelihood if return_nll else np.exp(-negloglikelihood)

    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = True,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Find the POI that maximizes the likelihood and the value of the maximum likelihood

        :param return_nll: if true, likelihood will be returned
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: allow negative POI
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :param kwargs: keyword arguments for optimiser
        :return: muhat, maximum of the likelihood
        """
        try:
            negloglikelihood, fit_param = self.backend.minimize_negative_loglikelihood(
                expected=expected,
                allow_negative_signal=allow_negative_signal,
                init_pars=init_pars,
                par_bounds=par_bounds,
                **kwargs,
            )
        except NotImplementedError:
            twice_nll, fit_param = fit(
                func=self.backend.get_twice_nll_func(expected=expected),
                model_configuration=self.backend.model.config(
                    allow_negative_signal=allow_negative_signal
                ),
                gradient=get_function(
                    self.backend, "get_gradient_twice_nll_func", expected=expected
                ),
                # hessian=get_function(
                #     self.backend,
                #     "get_hessian_twice_nll_func",
                #     default=None,
                #     expected=expected,
                # ),
                initial_parameters=init_pars,
                bounds=par_bounds,
                **kwargs,
            )
            negloglikelihood = twice_nll / 2.0

        muhat = fit_param[self.backend.model.config().poi_index]
        return muhat, negloglikelihood if return_nll else np.exp(-negloglikelihood)

    def maximize_asimov_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Find maximum of the likelihood for the asimov data

        :param expected (`ExpectationType`): observed, apriori or aposteriori,.
            (default `ExpectationType.observed`)
        :param return_nll (`bool`): if false, likelihood value is returned.
            (default `True`)
        :param test_statistics (`Text`): test statistics. `"qmu"` or `"qtilde"` for exclusion
                                     tests `"q0"` for discovery test. (default `"qtilde"`)
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :param kwargs: keyword arguments for optimiser
        :return `Tuple[float, float]`: muhat, negative log-likelihood
        """
        try:
            negloglikelihood, fit_param = self.backend.minimize_asimov_negative_loglikelihood(
                expected=expected,
                test_statistics=test_statistics,
                init_pars=init_pars,
                par_bounds=par_bounds,
                **kwargs,
            )
        except NotImplementedError:
            allow_negative_signal: bool = True if test_statistics in ["q", "qmu"] else False

            data = self.backend.generate_asimov_data(
                expected=expected, test_statistics=test_statistics
            )

            twice_nll, fit_param = fit(
                func=self.backend.get_twice_nll_func(expected=expected, data=data),
                model_configuration=self.backend.model.config(
                    allow_negative_signal=allow_negative_signal
                ),
                gradient=get_function(
                    self.backend, "get_gradient_twice_nll_func", expected=expected, data=data
                ),
                # hessian=get_function(
                #     self.backend,
                #     "get_hessian_twice_nll_func",
                #     default=None,
                #     expected=expected,
                #     data=data,
                # ),
                initial_parameters=init_pars,
                bounds=par_bounds,
                **kwargs,
            )
            negloglikelihood = twice_nll / 2.0

        muhat: float = fit_param[self.backend.model.config().poi_index]
        return muhat, negloglikelihood if return_nll else np.exp(-negloglikelihood)

    def fixed_poi_sampler(
        self,
        poi_test: float,
        size: Optional[int] = None,
        expected: ExpectationType = ExpectationType.observed,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Union[np.ndarray, Callable[[int], np.ndarray]]:
        """
        Sample from statistical model using fixed POI nuissance parameters

        :param poi_test (`float`): signal strength
        :param size (`int`, default `None`): number of samples to be drawn from the model. If not initiated
            a callable function will be returned which initialized with respect to nuisance parameters that
            minimizes negative log-likelihood of the given statistical model. Then one can sample through
            this function by inputting the size of the sample.
        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori or aposteriori.
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :param kwargs: optimizer options
        :raises `MethodNotAvailable`: Will be raised if backend does not have sampling capabilities.
        :return `np.ndarray`: Sampled yields.
        """
        _, fit_param = self.fixed_poi_fit(
            poi_test=poi_test,
            expected=expected,
            init_pars=init_pars,
            par_bounds=par_bounds,
            **kwargs,
        )

        try:
            sampler = self.backend.get_sampler(fit_param)
        except NotImplementedError as exc:
            raise MethodNotAvailable(
                f"{self.backend_type} backend does not have sampling capabilities"
            ) from exc

        return sampler(size) if isinstance(size, int) else sampler

    def sigma_mu_from_hessian(
        self,
        poi_test: float,
        expected: ExpectationType = ExpectationType.observed,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> float:
        """
        Compute sigma mu from inverse Hessian. see eq. (28) in https://arxiv.org/abs/1007.1727

        :param poi_test (`float`): parameter of interest
        :param expected (`ExpectationType`, default `ExpectationType.observed`): observed, apriori or aposteriori.
        :param init_pars (`Optional[List[float]]`, default `None`): initial fit parameters.
        :param par_bounds (`Optional[List[Tuple[float, float]]]`, default `None`): bounds for fit parameters.
        :raises `MethodNotAvailable`: If the hessian is not defined for the backend.
        :return `float`: sigma mu
        """
        try:
            hessian_func = self.backend.get_hessian_twice_nll_func(expected=expected)
        except NotImplementedError as exc:
            raise MethodNotAvailable(
                f"{self.backend_type} backend does not have Hessian definition."
            ) from exc

        _, fit_param = self.fixed_poi_fit(
            poi_test=poi_test,
            expected=expected,
            init_pars=init_pars,
            par_bounds=par_bounds,
            **kwargs,
        )

        hessian = hessian_func(fit_param) / 2.0

        poi_index = self.backend.model.config().poi_index
        return 1.0 / hessian[poi_index, poi_index]


def statistical_model_wrapper(func: BackendBase) -> StatisticalModel:
    """
    Wrapper for statistical model backends. Converts a backend base type statistical
    model into `StatisticalModel` instance.

    :param func (`BackendBase`): Statistical model described in one of the backends
    :return `StatisticalModel`: initialised statistical model
    """

    def wrapper(
        model: DataBase, analysis: Text = "__unknown_analysis__", xsection: float = np.nan, **kwargs
    ) -> StatisticalModel:
        """
        Statistical Model Base wrapper

        :param model (`DataBase`): Container that holds yield counts for statistical model and model properties.
                                   See current statistical model properties below for details.
        :param analysis (`Text`, default `"__unknown_analysis__"`): analysis name.
        :param xsection (`float`, default `np.nan`): cross section value. This value is only used for excluded
                                                     cross section value computation and does not assume any units.
        :param kwargs: Backend specific inputs. See current statistical model properties below for details.
        :return `StatisticalModel`: Statistical model interface
        :raises AssertionError: if the input function or model does not satisfy basic properties
        """
        assert isinstance(model, DataBase), "Input model does not satisfy base data properties."
        return StatisticalModel(backend=func(model, **kwargs), analysis=analysis, xsection=xsection)

    wrapper.__doc__ += (
        "\n\n\t Current statistical model properties:\n"
        + getattr(func, "__doc__", "no docstring available").replace("\n", "\n\t")
        + "\n"
    )

    return wrapper
