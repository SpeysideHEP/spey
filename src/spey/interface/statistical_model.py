"""Statistical Model wrapper class"""
import logging
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union

import numpy as np

from spey.base.backend_base import BackendBase
from spey.base.hypotest_base import HypothesisTestingBase
from spey.optimizer.core import fit
from spey.system.exceptions import (
    CombinerNotAvailable,
    MethodNotAvailable,
    UnknownCrossSection,
)
from spey.utils import ExpectationType

__all__ = ["StatisticalModel", "statistical_model_wrapper"]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=W1203


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
        ntoys (``int``, default ``1000``): Number of toy samples for hypothesis testing. (Only used for
          toy-based hypothesis testing)

    Raises:
        :obj:`AssertionError`: If the given backend does not inherit :class:`~spey.BackendBase`

    Returns:
        ~spey.StatisticalModel:
        General statistical model object that wraps around different likelihood prescriptions.
    """

    __slots__ = ["_backend", "xsection", "analysis"]

    def __init__(
        self,
        backend: BackendBase,
        analysis: Text,
        xsection: float = np.nan,
        ntoys: int = 1000,
    ):
        assert isinstance(backend, BackendBase), "Invalid backend"
        self._backend: BackendBase = backend
        self.xsection: float = xsection
        """Value of the cross section, unit is defined by the user."""
        self.analysis: Text = analysis
        """Unique identifier as analysis name"""
        super().__init__(ntoys=ntoys)

    def __repr__(self):
        calc = f"calculators={self.available_calculators}"

        if np.isnan(self.xsection):
            return (
                f"StatisticalModel(analysis='{self.analysis}', backend={self.backend_type}, "
                + f"{calc})"
            )

        return (
            f"StatisticalModel(analysis='{self.analysis}', "
            f"xsection={self.xsection:.3e} [au], backend={self.backend_type}, {calc})"
        )

    @property
    def backend(self) -> BackendBase:
        """Accessor to the backend"""
        return self._backend

    @property
    def backend_type(self) -> Text:
        """Return type of the backend"""
        return getattr(self.backend, "name", self.backend.__class__.__name__)

    @property
    def is_asymptotic_calculator_available(self) -> bool:
        """Check if Asymptotic calculator is available for the backend"""
        return self.backend.expected_data != BackendBase.expected_data or (
            self.backend.asimov_negative_loglikelihood
            != BackendBase.asimov_negative_loglikelihood
            and self.backend.minimize_asimov_negative_loglikelihood
            != BackendBase.minimize_asimov_negative_loglikelihood
        )

    @property
    def is_toy_calculator_available(self) -> bool:
        """Check if Toy calculator is available for the backend"""
        return self.backend.get_sampler != BackendBase.get_sampler

    @property
    def is_chi_square_calculator_available(self) -> bool:
        """Check if chi-square calculator is available for the backend"""
        return True

    @property
    def available_calculators(self) -> List[Text]:
        """
        Retruns available calculator names i.e. ``'toy'``,
        ``'asymptotic'`` and ``'chi_square'``.
        """
        calc = ["toy"] * self.is_toy_calculator_available
        calc += ["asymptotic"] * self.is_asymptotic_calculator_available
        calc += ["chi_square"] * self.is_chi_square_calculator_available
        return calc

    @property
    def is_alive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        return self.backend.is_alive

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

    def prepare_for_fit(
        self,
        data: Optional[Union[List[float], np.ndarray]] = None,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        **kwargs,
    ) -> Dict:
        r"""
        Prepare backend for the optimiser.

        Args:
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

            allow_negative_signal (``bool``, default ``True``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.

        Returns:
            ``Dict``:
            Dictionary of necessary toolset for the fit. objective function, ``"func"``, use gradient
            boolean, ``"do_grad"`` and function to compute negative log-likelihood with given
            fit parameters, ``"nll"``.
        """
        do_grad = kwargs.pop("do_grad", True)
        try:
            objective_and_grad = self.backend.get_objective_function(
                expected=expected, data=data, do_grad=do_grad
            )
        except NotImplementedError:
            log.debug("Gradient is not available, will not be included in computation.")
            do_grad = False
            objective_and_grad = self.backend.get_objective_function(
                expected=expected, data=data, do_grad=do_grad
            )

        constraints = kwargs.pop("constraints", [])
        if hasattr(self.backend, "constraints"):
            for constraint in self.backend.constraints:
                constraints.append(constraint)

        return {
            "func": objective_and_grad,
            "do_grad": do_grad,
            "model_configuration": self.backend.config(
                allow_negative_signal=allow_negative_signal
            ),
            "logpdf": self.backend.get_logpdf_func(expected=expected, data=data),
            "constraints": constraints,
            **kwargs,
        }

    def likelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        data: Optional[Union[List[float], np.ndarray]] = None,
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
            data (``Union[List[float], np.ndarray]``, default ``None``): input data that to fit. If
              ``None`` data will be set according to ``expected`` input.
            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
              the optimiser.
            kwargs: keyword arguments for the optimiser.

        Returns:
            ``float``:
            Likelihood of the statistical model at a fixed signal strength.
        """
        fit_opts = self.prepare_for_fit(expected=expected, data=data, **kwargs)

        if (
            fit_opts["model_configuration"].npar == 1
            and fit_opts["model_configuration"].poi_index is not None
        ):
            logpdf = fit_opts["logpdf"]([poi_test])
        else:
            logpdf, _ = fit(
                **fit_opts,
                initial_parameters=init_pars,
                bounds=par_bounds,
                fixed_poi_value=poi_test,
            )

        return -logpdf if return_nll else np.exp(logpdf)

    def generate_asimov_data(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistic: Text = "qtilde",
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> List[float]:
        r"""
        Generate Asimov data for the statistical model. This function generates a set of parameters
        (nuisance and poi i.e. :math:`\theta` and :math:`\mu`) with respect to ``test_statistic`` input
        which determines the value of :math:`\mu` i.e. if ``test_statistic="q0"`` :math:`\mu=1` and 0 for
        anything else. The objective function is used to optimize the statistical model to find the fit
        parameters for fixed poi optimisation. Then fit parameters are used to retreive the expected
        data through :func:`~spey.BackendBase.expected_data` function.

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

            test_statistic (``Text``, default ``"qtilde"``): test statistics.

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
            ``List[float]``:
            Asimov data
        """
        fit_opts = self.prepare_for_fit(
            expected=expected,
            allow_negative_signal=test_statistic in ["q", "qmu"],
            **kwargs,
        )

        _, fit_pars = fit(
            **fit_opts,
            initial_parameters=init_pars,
            bounds=par_bounds,
            fixed_poi_value=1.0 if test_statistic == "q0" else 0.0,
        )
        log.debug(f"fit parameters: {fit_pars}")

        return self.backend.expected_data(fit_pars)

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
        """
        return self.likelihood(
            poi_test=poi_test,
            expected=expected,
            return_nll=return_nll,
            data=self.generate_asimov_data(
                expected=expected,
                test_statistic=test_statistics,
                init_pars=init_pars,
                par_bounds=par_bounds,
                **kwargs,
            ),
            init_pars=init_pars,
            par_bounds=par_bounds,
            **kwargs,
        )

    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = True,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        data: Optional[Union[List[float], np.ndarray]] = None,
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
            data (``Union[List[float], np.ndarray]``, default ``None``): input data that to fit. If
              ``None`` data will be set according to ``expected`` input.
            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default  ``None``): parameter bounds for
              the optimiser.
            kwargs: keyword arguments for the optimiser.

        Returns:
            ``Tuple[float, float]``:
            :math:`\hat\mu` value and maximum value of the likelihood.
        """
        fit_opts = self.prepare_for_fit(
            expected=expected,
            allow_negative_signal=allow_negative_signal,
            data=data,
            **kwargs,
        )

        logpdf, fit_param = fit(
            **fit_opts, initial_parameters=init_pars, bounds=par_bounds
        )
        log.debug(f"fit parameters: {fit_param}")

        muhat = fit_param[self.backend.config().poi_index]
        return muhat, -logpdf if return_nll else np.exp(logpdf)

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
        """
        allow_negative_signal: bool = True if test_statistics in ["q", "qmu"] else False

        return self.maximize_likelihood(
            return_nll=return_nll,
            expected=expected,
            allow_negative_signal=allow_negative_signal,
            data=self.generate_asimov_data(
                expected=expected,
                test_statistic=test_statistics,
                init_pars=init_pars,
                par_bounds=par_bounds,
                **kwargs,
            ),
            init_pars=init_pars,
            par_bounds=par_bounds,
            **kwargs,
        )

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
        """
        fit_opts = self.prepare_for_fit(expected=expected, **kwargs)

        if (
            fit_opts["model_configuration"].npar == 1
            and fit_opts["model_configuration"].poi_index is not None
        ):
            fit_param = np.array(list(poi_test))
        else:
            _, fit_param = fit(
                **fit_opts,
                initial_parameters=init_pars,
                bounds=par_bounds,
                fixed_poi_value=poi_test,
            )

        log.debug(f"fit parameters: {fit_param}")
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
        """
        try:
            hessian_func = self.backend.get_hessian_logpdf_func(expected=expected)
        except NotImplementedError as exc:
            raise MethodNotAvailable(
                f"{self.backend_type} backend does not have Hessian definition."
            ) from exc

        fit_opts = self.prepare_for_fit(expected=expected, **kwargs)
        _ = fit_opts.pop("logpdf")

        _, fit_param = fit(
            **fit_opts,
            initial_parameters=init_pars,
            bounds=par_bounds,
            fixed_poi_value=poi_test,
        )
        log.debug(f"fit parameters: {fit_param}")

        hessian = -1.0 * hessian_func(fit_param)
        log.debug(f"full hessian: {hessian}")

        poi_index = self.backend.config().poi_index
        return np.sqrt(np.linalg.inv(hessian)[poi_index, poi_index])

    def combine(self, other, **kwargs):
        """
        Combination routine between two statistical models.

        .. note::

            This function's availability is backend dependent.

        Args:
            other (:obj:`~spey.StatisticalModel`): Statistical model to be combined with
              this model
            kwargs: backend specific arguments.

        Raises:
            :obj:`~spey.system.exceptions.CombinerNotAvailable`: If this statistical model
              does not have a combination routine implementation.
            ``AssertionError``: If the combination routine in the backend does not return
              a :obj:`~spey.BackendBase` object.

        Returns:
            :obj:`~spey.StatisticalModel`:
            Returns a new combined statistical model.
        """
        try:
            combined = self.backend.combine(other.backend, **kwargs)
            assert isinstance(combined, BackendBase), "Invalid combination operation."

            return StatisticalModel(
                backend=combined, analysis=f"combine[{self.analysis}, {other.analysis}]"
            )

        except NotImplementedError as err:
            raise CombinerNotAvailable(
                f"{self.backend_type} backend does not have a combination routine."
            ) from err

    def __matmul__(self, other):
        """
        Combination routine between two statistical models.
        See :func:`~spey.StatisticalModel.combine` function for details.
        """
        return self.combine(other)


def statistical_model_wrapper(
    func: BackendBase,
) -> Callable[[Any], StatisticalModel]:
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
        ``Callable[[Any], StatisticalModel]``:
        Wrapper that takes the following inputs

        * **args**: Backend specific arguments.
        * **analysis** (``Text``, default ``"__unknown_analysis__"``): Unique identifier of the
          statistical model. This attribue will be used for book keeping purposes.
        * **xsection** (``float``, default ``nan``): cross section, unit is determined by the
          user. Cross section value is only used for computing upper limit on excluded
          cross-section value.
        * **ntoys** (``int``, default ``1000``): Number of toy samples for hypothesis testing.
            (Only used for toy-based hypothesis testing)
        * **other keyword arguments**: Backend specific keyword inputs.
    """

    def wrapper(
        *args,
        analysis: Text = "__unknown_analysis__",
        xsection: float = np.nan,
        ntoys: int = 1000,
        **kwargs,
    ) -> StatisticalModel:
        """
        Statistical Model Backend wrapper.

        Args:
            args: Backend specific arguments.
            analysis (``Text``, default ``"__unknown_analysis__"``): Unique identifier of the
              statistical model. This attribue will be used for book keeping purposes.
            xsection (``float``, default ``nan``): cross section, unit is determined by the
              user. Cross section value is only used for computing upper limit on excluded
              cross-section value.
            ntoys (``int``, default ``1000``): Number of toy samples for hypothesis testing.
              (Only used for toy-based hypothesis testing)
            kwargs: Backend specific keyword inputs.

        Raises:
            :obj:`AssertionError`: If the model input does not inherit :class:`~spey.DataBase`.

        Returns:
            ~spey.StatisticalModel:
            Backend wraped with statistical model interface.
        """
        return StatisticalModel(
            backend=func(*args, **kwargs),
            analysis=analysis,
            xsection=xsection,
            ntoys=ntoys,
        )

    docstring = getattr(func, "__doc__", "no docstring available")
    if docstring is None:
        docstring = "Documentation is not available..."

    wrapper.__doc__ += (
        "\n\t"
        + "<>" * 30
        + "\n\n\t Current statistical model backend properties:\n"
        + docstring.replace("\n", "\n\t")
        + "\n"
    )

    return wrapper
