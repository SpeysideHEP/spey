"""Statistical Model wrapper class"""

from typing import Optional, Text, Tuple, List, Callable, Union

import numpy as np

from spey.utils import ExpectationType
from spey.base.backend_base import BackendBase, DataBase
from spey.system.exceptions import UnknownCrossSection, MethodNotAvailable
from spey.base.hypotest_base import HypothesisTestingBase
from spey.optimizer.core import fit

__all__ = ["StatisticalModel", "statistical_model_wrapper"]


class StatisticalModel(HypothesisTestingBase):
    r"""
    Statistical model base. This class wraps around the various statistical model backends available
    through `spey`'s plugin system. Each backend has to inherit :class:`~spey.BackendBase` which sets
    certain requirements on the available functionality to be used for hypothesis testing. These
    requirements are such as accessibility to log-likelihood, :math:`\log\mathcal{L}`, it's derivative
    with respect to :math:`\mu` and nuisance parameters, :math:`\partial_\theta\log\mathcal{L}`,
    its Hessian and Assimov data generation. Depending on availablility :class:`~spey.StatisticalModel`
    will take propriate action to perform requested computation. The goal of this class is to collect
    all different backends under same roof in order to perform combination of different likelihood recipies.

    Args:
        backend (~spey.BackendBase): Statistical model backend
        analysis (:obj:`Text`): Unique identifier of the statistical model. This attribue will be used
          for book keeping purposes.
        xsection (:obj:`float`, default :obj:`np.nan`): cross section, unit is determined by the user.
          Cross section value is only used for computing upper limit on excluded cross-section value.

    Raises:
        :obj:`AssertionError`: If the given backend does not inherit :class:`~spey.BackendBase`

    Returns:
        ~spey.StatisticalModel: General statistical model object that wraps around different likelihood prescriptions.
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

    def _get_objective_and_grad(
        self, expected: ExpectationType, data: np.ndarray
    ) -> Tuple[Callable, bool]:
        """
        Retreive objective and gradient function

        :param expected (`ExpectationType`): observed, apriori or aposteriori.
        :param data (`np.ndarray`): observations
        :return `Tuple[Callable, bool]`: objective and grad function and a boolean indicating
            that the backend is differentiable.
        """
        do_grad = True
        try:
            objective_and_grad = self.backend.get_objective_function(
                expected=expected, data=data, do_grad=do_grad
            )
        except NotImplementedError:
            do_grad = False
            objective_and_grad = self.backend.get_objective_function(
                expected=expected, data=data, do_grad=do_grad
            )
        return objective_and_grad, do_grad

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
        objective_and_grad, do_grad = self._get_objective_and_grad(expected, data)

        twice_nll, fit_param = fit(
            func=objective_and_grad,
            model_configuration=self.backend.model.config(),
            do_grad=do_grad,
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
        :param expected (~spey.ExpectationType, default ~spey.ExpectationType.observed): observed, apriori or aposteriori.
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
        r"""
        Compute likelihood of the statistical model generated with the Asimov data.

        Args:
            poi_test (:obj:`float`, default :obj:`1.0`): parameter of interest.
            expected (~spey.ExpectationType): :obj:`observed`, :obj:`apriori` or :obj:`aposteriori`.
              Default :attr:`~spey.ExpectationType.observed`.
            return_nll (:obj:`bool`, default :obj:`True`): if false returns likelihood value.
            test_statistics (:obj:`Text`, default :obj:`"qtilde"`): test statistics.

              * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
                :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`
                (:func:`~spey.hypothesis_testing.test_statistics.qmu_tilde`).

                .. warning::

                    Note that this assumes that :math:`\hat\mu\geq0`, hence :obj:`allow_negative_signal`
                    assumed to be :obj:`False`. If this function has been executed by user, spey assumes
                    that this is taken care of through out the external code consistently. Whilst executing
                    computing p-values or upper limit on :math:`\mu` this is taken care of automatically
                    in the backend.

              * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`, see
                eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
              * ``'q0'``: performs the calculation using the discovery test statistic, see eq. (47)
                of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

              The choice of :obj:`test_statistics` will effect the generation of the Asimov data where
              the fit is performed via :math:`\mu=1` if :obj:`test_statistics="q0"` and :math:`\mu=0`
              for others. Note that this :math:`\mu` does not correspond to the :obj:`poi_test` input
              of this function but it determines how Asimov data is generated.
            init_pars (:obj:`List[float]`, default :obj:`None`): initial fit parameters.
            par_bounds (:obj:`List[Tuple[float, float]]`, default :obj:`None`): bounds for fit
              parameters.

        Returns:
            :obj:`float`: likelihood computed for asimov data
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
            objective_and_grad, do_grad = self._get_objective_and_grad(expected, None)

            twice_nll, fit_param = fit(
                func=objective_and_grad,
                model_configuration=self.backend.model.config(
                    allow_negative_signal=allow_negative_signal
                ),
                do_grad=do_grad,
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

            objective_and_grad, do_grad = self._get_objective_and_grad(expected, data)

            twice_nll, fit_param = fit(
                func=objective_and_grad,
                model_configuration=self.backend.model.config(
                    allow_negative_signal=allow_negative_signal
                ),
                do_grad=do_grad,
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
            hessian_func = self.backend.get_hessian_logpdf_func(expected=expected)
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

        hessian = -1.0 * hessian_func(fit_param)

        poi_index = self.backend.model.config().poi_index
        return np.sqrt(np.linalg.inv(hessian)[poi_index, poi_index])


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
