"""Statistical Model wrapper class"""

from typing import Optional, Text, Tuple, List

import numpy as np

from spey.utils import ExpectationType
from spey.base.backend_base import BackendBase, DataBase
from spey.system.exceptions import UnknownCrossSection
from spey.base.hypotest_base import HypothesisTestingBase

__all__ = ["StatisticalModel", "statistical_model_wrapper"]


class StatisticalModel(HypothesisTestingBase):
    """
    Statistical model base

    :param backend: Statistical model backend
    :param xsection: Cross-section in pb
    :param analysis: name of the analysis
    """

    __slots__ = ["_backend", "xsection", "analysis"]

    def __init__(self, backend: BackendBase, analysis: Text, xsection: float = np.NaN):
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
        negloglikelihood, _ = self.backend.negative_loglikelihood(
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
        negloglikelihood, _ = self.backend.asimov_negative_loglikelihood(
            poi_test=poi_test,
            expected=expected,
            test_statistics=test_statistics,
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
        negloglikelihood, fit_param = self.backend.minimize_negative_loglikelihood(
            expected=expected,
            allow_negative_signal=allow_negative_signal,
            init_pars=init_pars,
            par_bounds=par_bounds,
            **kwargs,
        )
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
        negloglikelihood, fit_param = self.backend.minimize_asimov_negative_loglikelihood(
            expected=expected,
            test_statistics=test_statistics,
            init_pars=init_pars,
            par_bounds=par_bounds,
            **kwargs,
        )
        muhat: float = fit_param[self.backend.model.config().poi_index]
        return muhat, negloglikelihood if return_nll else np.exp(-negloglikelihood)


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
        Statistical Model wrapper

        :param model (`DataBase`): Container that holds yield counts for statistical model and model properties
        :param analysis (`Text`, default `"__unknown_analysis__"`): analysis name.
        :param xsection (`float`, default `np.nan`): cross section value.
        :param kwargs: Backend specific inputs
        :return `StatisticalModel`: Statistical model interface
        :raises AssertionError: if the input function or model does not satisfy basic properties
        """
        assert isinstance(model, DataBase), "Input model does not satisfy base data properties."
        return StatisticalModel(backend=func(model, **kwargs), analysis=analysis, xsection=xsection)

    return wrapper
