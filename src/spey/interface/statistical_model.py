"""Statistical Model wrapper class"""

from typing import Optional, Text, Tuple, List, Callable, Union, Any

import numpy as np

from spey.utils import ExpectationType
from spey.base.backend_base import BackendBase
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
    all different backends under same roof in order to perform combination of different likelihood
    recipies.

    Args:
        backend (~spey.BackendBase): Statistical model backend
        analysis (``Text``): Unique identifier of the statistical model. This attribue will be used
          for book keeping purposes.
        xsection (``float``, default ``np.nan``): cross section, unit is determined by the user.
          Cross section value is only used for computing upper limit on excluded cross-section value.

    Raises:
        :obj:`AssertionError`: If the given backend does not inherit :class:`~spey.BackendBase`

    Returns:
        ~spey.StatisticalModel:
        General statistical model object that wraps around different likelihood prescriptions.
    """

    __slots__ = ["_backend", "xsection", "analysis"]

    def __init__(self, backend: BackendBase, analysis: Text, xsection: float = np.nan):
        assert isinstance(backend, BackendBase), "Invalid backend"
        self._backend: BackendBase = backend
        self.xsection: float = xsection
        """Value of the cross section, unit is defined by the user."""
        self.analysis: Text = analysis
        """Unique identifier as analysis name"""

    def __repr__(self):
        return (
            f"StatisticalModel(analysis='{self.analysis}', "
            f"xsection={self.xsection:.3e} [au], "
            f"backend={str(self.backend_type)})"
        )

    @property
    def backend(self) -> BackendBase:
        """Accessor to the backend"""
        return self._backend

    @property
    def backend_type(self) -> Text:
        """Return type of the backend e.g. ``simplified_likelihoods``"""
        return self.backend.name

    @property
    def isAlive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        return self.backend.model.isAlive

    def excluded_cross_section(
        self, expected: ExpectationType = ExpectationType.observed
    ) -> float:
        """
        Compute excluded cross section value at 95% CL

        Args:
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.

        Raises:
            ~spey.system.exceptions.UnknownCrossSection: If the cross-section is ``nan``.

        Returns:
            ``float``:
            Returns the upper limit at 95% CL on cross section value where the unit is defined
            by the user.

        Example:

        .. code-block:: python3

            >>> import spey
            >>> statistical_model = spey.get_uncorrelated_nbin_statistical_model(
            ...     1, 2.0, 1.1, 0.5, 0.123, "simple_sl", "simplified_likelihoods"
            ... )
            >>> for expectation in spey.ExpectationType:
            >>>     print(
            ...         f"Excluded cross section with {expectation}: ",
            ...         statistical_model.excluded_cross_section(expected=expectation)
            ...     )
            >>> # Excluded cross section with apriori: 1.126437181831991
            >>> # Excluded cross section with aposteriori: 0.9600107752838337
            >>> # Excluded cross section with observed: 0.828274776848163
        """
        if np.isnan(self.xsection):
            raise UnknownCrossSection("Cross-section value has not been initialised.")

        return (
            self.poi_upper_limit(expected=expected, confidence_level=0.95) * self.xsection
        )

    @property
    def s95exp(self) -> float:
        """
        Compute excluded cross section value at 95% CL with :obj:`~spey.ExpectationType.apriori`
        expectation. See :func:`~spey.StatisticalModel.excluded_cross_section`
        for reference.

        Raises:
            ~spey.system.exceptions.UnknownCrossSection: If the cross-section is ``nan``.
        """
        return self.excluded_cross_section(ExpectationType.apriori)

    @property
    def s95obs(self) -> float:
        """
        Compute excluded cross section value at 95% CL with :obj:`~spey.ExpectationType.observed`
        expectation. See :func:`~spey.StatisticalModel.excluded_cross_section`
        for reference.

        Raises:
            ~spey.system.exceptions.UnknownCrossSection: If the cross-section is ``nan``.
        """
        return self.excluded_cross_section(ExpectationType.observed)

    def _get_objective_and_grad(
        self, expected: ExpectationType, data: np.ndarray
    ) -> Tuple[Callable, bool]:
        """
        Retreive objective and gradient function

        :param expected (``ExpectationType``): observed, apriori or aposteriori.
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
        r"""
        Find the minimum of negative log-likelihood for a given parameter of interest.

        Args:
            poi_test (``float``, default ``1.0``): parameter of interest or signal strength,
              :math:`\mu`.
            data (``Union[List[float], np.ndarray]``, default ``None``): input data that to fit
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
              the optimiser.
            kwargs: keyword arguments for the optimiser.

        Returns:
            ``Tuple[float, np.ndarray]``:
            negative log-likelihood value and fit parameters.

        Example:

        .. code-block:: python3

            >>> import spey
            >>> statistical_model = spey.get_uncorrelated_nbin_statistical_model(
            ...     1, 2.0, 1.1, 0.5, 0.123, "simple_sl", "simplified_likelihoods"
            ... )
            >>> statistical_model.fixed_poi_fit(0.5) # (2.3078367000498305, array([0.5, -0.51327448]))
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
        r"""
        Compute the likelihood of the statistical model at a fixed parameter of interest

        Args:
            poi_test (``float``, default ``1.0``): parameter of interest or signal strength,
              :math:`\mu`.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            return_nll (``bool``, default ``True``): If ``True``, returns negative log-likelihood value.
              if ``False`` returns likelihood value.
            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
              the optimiser.
            kwargs: keyword arguments for the optimiser.

        Returns:
            ``float``:
            Likelihood of the statistical model at a fixed signal strength.

        Example:

        .. code-block:: python3

            >>> import spey
            >>> statistical_model = spey.get_uncorrelated_nbin_statistical_model(
            ...     1, 2.0, 1.1, 0.5, 0.123, "simple_sl", "simplified_likelihoods"
            ... )
            >>> statistical_model.likelihood(0.5) # 2.3078367000498305
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
            poi_test (``float``, default ``1.0``): parameter of interest, :math:`\mu`.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            return_nll (``bool``, default ``True``): If ``True``, returns negative log-likelihood value.
              if ``False`` returns likelihood value.
            test_statistics (``Text``, default ``"qtilde"``): test statistics.

              * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
                :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`
                (:func:`~spey.hypothesis_testing.test_statistics.qmu_tilde`).

                .. warning::

                    Note that this assumes that :math:`\hat\mu\geq0`, hence ``allow_negative_signal``
                    assumed to be ``False``. If this function has been executed by user, ``spey``
                    assumes that this is taken care of throughout the external code consistently.
                    Whilst computing p-values or upper limit on :math:`\mu` through ``spey`` this
                    is taken care of automatically in the backend.

              * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`, see
                eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
              * ``'q0'``: performs the calculation using the discovery test statistic, see eq. (47)
                of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

              The choice of ``test_statistics`` will effect the generation of the Asimov data where
              the fit is performed via :math:`\mu=1` if ``test_statistics="q0"`` and :math:`\mu=0`
              for others. Note that this :math:`\mu` does not correspond to the ``poi_test`` input
              of this function but it determines how Asimov data is generated.
            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
              the optimiser.
            kwargs: keyword arguments for the optimiser.

        Returns:
            ``float``:
            likelihood computed for asimov data

        Example:

        .. code-block:: python3

            >>> import spey
            >>> statistical_model = spey.get_uncorrelated_nbin_statistical_model(
            ...     1, 2.0, 1.1, 0.5, 0.123, "simple_sl", "simplified_likelihoods"
            ... )
            >>> for test_stat in ["q", "qtilde", "q0"]:
            >>>     print(
            ...         f"L_A with {test_stat}: ",
            ...         statistical_model.asimov_likelihood(0.5, test_statistics=test_stat)
            ...     )
            >>> # L_A with q:  2.2866167339701358
            >>> # L_A with qtilde:  2.2866167339701358
            >>> # L_A with q0:  2.2829074109879595
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
        r"""
        Find the maximum of the likelihood.

        Args:
            return_nll (``bool``, default ``True``): If  ``True``, returns negative log-likelihood value.
              if ``False`` returns likelihood value.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.

            allow_negative_signal (``bool``, default ``True``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.
            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default  ``None``): parameter bounds for
              the optimiser.
            kwargs: keyword arguments for the optimiser.

        Returns:
            ``Tuple[float, float]``:
            :math:`\hat\mu` value and maximum value of the likelihood.

        Examples:

        .. code-block:: python3

            >>> import spey
            >>> statistical_model = spey.get_uncorrelated_nbin_statistical_model(
            ...     1, 2.0, 1.1, 0.5, 0.123, "simple_sl", "simplified_likelihoods"
            ... )
            >>> print("muhat: %.3f, negative log-likelihood %.3f" % statistical_model.maximize_likelihood())
            >>> # muhat: -2.001, negative log-likelihood 2.014
            >>> print("muhat: %.3f, negative log-likelihood %.3f" % statistical_model.maximize_likelihood(allow_negative_signal=False))
            >>> # muhat: 0.000, negative log-likelihood 2.210
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
        r"""
        Find the maximum of the likelihood which computed with respect to Asimov data.

        Args:
            return_nll (``bool``, default ``True``): If ``True``, returns negative log-likelihood value.
              if ``False`` returns likelihood value.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.

            test_statistics (``Text``, default ``"qtilde"``): test statistic.

              * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
                :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`
                (:func:`~spey.hypothesis_testing.test_statistics.qmu_tilde`).

                .. warning::

                    Note that this assumes that :math:`\hat\mu\geq0`, hence ``allow_negative_signal``
                    assumed to be ``False``. If this function has been executed by user, ``spey``
                    assumes that this is taken care of throughout the external code consistently.
                    Whilst computing p-values or upper limit on :math:`\mu` through ``spey`` this
                    is taken care of automatically in the backend.

              * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`, see
                eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
              * ``'q0'``: performs the calculation using the discovery test statistic, see eq. (47)
                of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
              the optimiser.
            kwargs: keyword arguments for the optimiser.

        Returns:
            ``Tuple[float, float]``:
            :math:`\hat\mu` value and maximum value of the likelihood.

        Example:

        .. code-block:: python3

            >>> import spey
            >>> statistical_model = spey.get_uncorrelated_nbin_statistical_model(
            ...     1, 2.0, 1.1, 0.5, 0.123, "simple_sl", "simplified_likelihoods"
            ... )
            >>> print("muhat: %.3f, negative log-likelihood %.3f" % statistical_model.maximize_asimov_likelihood(test_statistics="q"))
            >>> # muhat: -0.871, negative log-likelihood 2.209
            >>> print("muhat: %.3f, negative log-likelihood %.3f" % statistical_model.maximize_asimov_likelihood(test_statistics="qtilde"))
            >>> # muhat: 0.000, negative log-likelihood 2.242
            >>> print("muhat: %.3f, negative log-likelihood %.3f" % statistical_model.maximize_asimov_likelihood(test_statistics="q0"))
            >>> # muhat: 0.000, negative log-likelihood 2.225
        """
        try:
            (
                negloglikelihood,
                fit_param,
            ) = self.backend.minimize_asimov_negative_loglikelihood(
                expected=expected,
                test_statistics=test_statistics,
                init_pars=init_pars,
                par_bounds=par_bounds,
                **kwargs,
            )
        except NotImplementedError:
            allow_negative_signal: bool = (
                True if test_statistics in ["q", "qmu"] else False
            )

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
        r"""
        Sample data from the statistical model with fixed parameter of interest.

        Args:
            poi_test (``float``, default ``1.0``): parameter of interest or signal strength,
              :math:`\mu`.
            size (``int``, default ``None``): sample size. If ``None`` a callable function
              will be returned which takes sample size as input.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
              the optimiser.
            kwargs: keyword arguments for the optimiser.

        Raises:
            ~spey.system.exceptions.MethodNotAvailable: If bacend does not have sampler implementation.

        Returns:
            ``Union[np.ndarray, Callable[[int], np.ndarray]]``:
            Sampled data with shape of ``(size, number of bins)`` or callable function to sample from
            directly.

        Example:

        .. code-block:: python3

            >>> import spey
            >>> statistical_model = spey.get_uncorrelated_nbin_statistical_model(
            ...     1, 2.0, 1.1, 0.5, 0.123, "simple_sl", "simplified_likelihoods"
            ... )
            >>> bkg_sample = statistical_model.fixed_poi_sampler(0., 10)
            >>> bkg_sample.shape
            >>> # (10, 1)
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
        r"""
        Compute variance of :math:`\mu` from inverse Hessian. See eq. (27-28) in :xref:`1007.1727`.

        Args:
            poi_test (``float``, default ``1.0``): parameter of interest or signal strength,
              :math:`\mu`.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
              the optimiser.
            kwargs: keyword arguments for the optimiser.
        Raises:
            ~spey.system.exceptions.MethodNotAvailable: If bacend does not have Hessian implementation.

        Returns:
            ``float``:
            variance on parameter of interest.

        Example:

        .. code-block:: python3

            >>> import spey
            >>> statistical_model = spey.get_uncorrelated_nbin_statistical_model(
            ...     1, 2.0, 1.1, 0.5, 0.123, "simple_sl", "simplified_likelihoods"
            ... )
            >>> statistical_model.sigma_mu_from_hessian(0.3)
            >>> # 3.993148035554591
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


def statistical_model_wrapper(
    func: BackendBase,
) -> Callable[[Any, ...], StatisticalModel]:
    """
    Backend wrapper for :class:`~spey.StatisticalModel`. This function allows a universal
    integration of each backend to the :obj:`spey` environment. :func:`~spey.get_backend` function
    automatically wraps the backend with :func:`~spey.statistical_model_wrapper` before returning
    the object.

    Args:
        func (~spey.BackendBase): Desired backend to be used for statistical analysis.

    Raises:
        :obj:`AssertionError`: If the input function does not inherit :obj:`~spey.BackendBase`

    Returns:
        ``Callable[[Any, ...], StatisticalModel]``:
        Wrapper that takes the following inputs

        * **args**: Backend specific arguments.
        * **analysis** (``Text``, default ``"__unknown_analysis__"``): Unique identifier of the
          statistical model. This attribue will be used for book keeping purposes.
        * **xsection** (``float``, default ``np.nan``): cross section, unit is determined by the
          user. Cross section value is only used for computing upper limit on excluded
          cross-section value.
        * **other keyword arguments**: Backend specific keyword inputs.
    """

    def wrapper(
        *args, analysis: Text = "__unknown_analysis__", xsection: float = np.nan, **kwargs
    ) -> StatisticalModel:
        """
        Statistical Model Backend wrapper.

        Args:
            args: Backend dependent arguments.
            analysis (``Text``, default ``"__unknown_analysis__"``): Unique identifier of the
              statistical model. This attribue will be used for book keeping purposes.
            xsection (``float``, default ``np.nan``): cross section, unit is determined by the
              user. Cross section value is only used for computing upper limit on excluded
              cross-section value.
            kwargs: Backend specific keyword inputs.

        Raises:
            :obj:`AssertionError`: If the model input does not inherit :class:`~spey.DataBase`.

        Returns:
            ~spey.StatisticalModel:
            Backend wraped with statistical model interface.
        """
        return StatisticalModel(
            backend=func(*args, **kwargs), analysis=analysis, xsection=xsection
        )

    wrapper.__doc__ += (
        "\n\n\t Current statistical model backend properties:\n"
        + getattr(func, "__doc__", "no docstring available").replace("\n", "\n\t")
        + "\n"
    )

    return wrapper
