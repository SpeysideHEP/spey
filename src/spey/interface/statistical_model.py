"""
This module defines :class:`~spey.StatisticalModel`, the central user-facing object in
``spey``.  It wraps any backend that inherits :class:`~spey.BackendBase` and provides a
unified API for likelihood evaluation, hypothesis testing, upper-limit extraction, and
model combination.  The module also exposes :func:`statistical_model_wrapper`, the
decorator that backends are registered with, and the :data:`PoiTest` type alias used
throughout.
"""
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spey.base.backend_base import BackendBase, ModelConfig
from spey.base.hypotest_base import HypothesisTestingBase
from spey.optimizer.core import fit
from spey.system.exceptions import (
    CombinerNotAvailable,
    MethodNotAvailable,
    UnknownCrossSection,
)
from spey.utils import ExpectationType

#: Type alias for ``poi_test``: either a single :obj:`float` or a :obj:`dict`
#: mapping POI indices / names to their fixed values.
PoiTest = Union[float, Dict[Union[int, str], float]]

__all__ = ["StatisticalModel", "statistical_model_wrapper", "PoiTest"]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=W1203


class StatisticalModel(HypothesisTestingBase):
    r"""
    Unified interface to any ``spey`` statistical model backend.

    :class:`~spey.StatisticalModel` is the central user-facing object in ``spey``.  It
    wraps any backend that inherits :class:`~spey.BackendBase`, giving every backend a
    consistent API for:

    * evaluating the (negative) log-likelihood :math:`-\log\mathcal{L}(\mu,\hat{\theta}_\mu)`;
    * maximising the likelihood to obtain :math:`\hat\mu` and
      :math:`\hat{\theta}`;
    * computing p-values, CLs values, and upper limits via asymptotic, toy-based,
      or :math:`\chi^2` calculators;
    * generating Asimov data;
    * combining two models with the ``@`` operator.

    Instances are normally obtained through :func:`spey.get_backend` rather than
    constructed directly:

    .. code-block:: python

        import spey

        # Obtain a backend constructor wrapped as StatisticalModel
        pdf_wrapper = spey.get_backend("default.poisson")

        # Build the statistical model
        model = pdf_wrapper(
            signal_yields=[12.0, 15.0],
            background_yields=[50.0, 60.0],
            data=[48, 63],
            analysis="my_analysis",
            xsection=0.05,  # pb
        )

    **Likelihood evaluation**

    .. code-block:: python

        # Negative log-likelihood at mu = 1
        nll = model.likelihood(poi_test=1.0)

        # Profile likelihood ratio test statistic
        nll_free, _ = model.maximize_likelihood()

        # Asimov (expected) likelihood
        nll_asimov = model.asimov_likelihood(poi_test=1.0)

    **Hypothesis testing**

    .. code-block:: python

        # Observed CLs value
        cls_obs = model.exclusion_confidence_level(poi_test=1.0)

        # Expected (apriori) CLs value
        cls_exp = model.exclusion_confidence_level(
            poi_test=1.0, expected=spey.ExpectationType.apriori
        )

        # One-sided 95 % CL upper limit on the signal strength
        mu_ul = model.poi_upper_limit(confidence_level=0.95)

        # Upper limit on the cross section (requires xsection to be set)
        xsec_ul = model.s95obs

    **Multi-parameter fits**

    When a backend exposes more than one parameter of interest, ``poi_test`` and
    ``poi_indices`` accept either an ``int`` index, a ``str`` parameter name, or a
    ``dict`` mapping indices / names to values:

    .. code-block:: python

        # Fix mu_0 = 1.0, mu_1 = 0.5
        nll = model.likelihood(poi_test={0: 1.0, 1: 0.5})

        # Retrieve fitted values of two named POIs
        muhat_dict, nll = model.maximize_likelihood(poi_indices=[0, 1])

    **Model combination**

    .. code-block:: python

        combined = model_a @ model_b          # uses the @ operator
        # or equivalently:
        combined = model_a.combine(model_b)

    Args:
        backend (:class:`~spey.BackendBase`): Statistical model backend.  Must be an instance of
          a class that inherits :class:`~spey.BackendBase`.
        analysis (``str``): Unique identifier of the statistical model used for
          book-keeping purposes.
        xsection (``float``, default ``np.nan``): Signal cross section in units chosen
          by the user.  Only required for
          :func:`~spey.StatisticalModel.excluded_cross_section`,
          :attr:`~spey.StatisticalModel.s95obs`, and
          :attr:`~spey.StatisticalModel.s95exp`.
        ntoys (``int``, default ``1000``): Number of pseudo-experiments (toys) used by
          the toy-based hypothesis-testing calculator.  Ignored when the asymptotic or
          :math:`\chi^2` calculator is used.

    Raises:
        :obj:`AssertionError`: If ``backend`` does not inherit :class:`~spey.BackendBase`.

    Returns:
        :class:`~spey.StatisticalModel`:
        A statistical model object wrapping the given backend with a unified hypothesis-
        testing interface.
    """

    __slots__ = ["xsection", "analysis", "_backend"]

    def __init__(
        self,
        backend: BackendBase,
        analysis: str,
        xsection: float = np.nan,
        ntoys: int = 1000,
    ):
        super().__init__(ntoys=ntoys)
        assert isinstance(backend, BackendBase), "Invalid backend"
        self._backend: BackendBase = backend
        self.xsection: float = xsection
        """Value of the cross section, unit is defined by the user."""
        self.analysis: str = analysis
        """Unique identifier as analysis name"""

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
        """
        The underlying backend instance.

        Returns:
            ~spey.BackendBase:
            The backend object that was supplied at construction time.  All likelihood
            and sampling calls are delegated to this object.
        """
        return self._backend

    @property
    def backend_type(self) -> str:
        """
        Human-readable name of the backend.

        Returns the value of the backend's ``name`` attribute when present, and falls
        back to the class name otherwise.

        Returns:
            ``str``:
            Backend identifier string (e.g. ``"default.poisson"``).
        """
        return getattr(self.backend, "name", self.backend.__class__.__name__)

    @property
    def config(self) -> ModelConfig:
        """Retreive model configuration"""
        return self.backend.config

    @property
    def is_asymptotic_calculator_available(self) -> bool:
        """
        Whether the asymptotic calculator can be used with this backend.

        The asymptotic calculator requires either:

        * a working :func:`~spey.BackendBase.expected_data` implementation, **or**
        * both :func:`~spey.BackendBase.asimov_negative_loglikelihood` and
          :func:`~spey.BackendBase.minimize_asimov_negative_loglikelihood` to be
          overridden by the backend.

        Returns:
            ``bool``:
            ``True`` if the asymptotic calculator is available.
        """
        return self.backend.expected_data != BackendBase.expected_data or (
            self.backend.asimov_negative_loglikelihood
            != BackendBase.asimov_negative_loglikelihood
            and self.backend.minimize_asimov_negative_loglikelihood
            != BackendBase.minimize_asimov_negative_loglikelihood
        )

    @property
    def is_toy_calculator_available(self) -> bool:
        """
        Whether the toy (pseudo-experiment) calculator can be used with this backend.

        Requires the backend to override :func:`~spey.BackendBase.get_sampler`.

        Returns:
            ``bool``:
            ``True`` if the toy calculator is available.
        """
        return self.backend.get_sampler != BackendBase.get_sampler

    @property
    def is_chi_square_calculator_available(self) -> bool:
        r"""
        Whether the :math:`\chi^2` calculator can be used with this backend.

        The :math:`\chi^2` calculator only requires the negative log-likelihood, which
        every backend must implement, so this property always returns ``True``.

        Returns:
            ``bool``:
            Always ``True``.
        """
        return True

    @property
    def available_calculators(self) -> List[str]:
        """
        Returns available calculator names.

        Possible entries are ``'toy'``, ``'asymptotic'``, and ``'chi_square'``,
        depending on what the underlying backend supports.

        Returns:
            ``List[str]``:
            Subset of ``['toy', 'asymptotic', 'chi_square']`` listing the calculators
            that are available for this model.
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
                prescription which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescription which means that the experimental data will be assumed to be
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
        Expected excluded cross section at 95% CL (pre-fit / *a-priori* expectation).

        Shorthand for ``excluded_cross_section(ExpectationType.apriori)``.  The result
        represents the cross-section value that would be excluded at the 95% confidence
        level if no signal were present (SM hypothesis), expressed in the same units as
        :attr:`~spey.StatisticalModel.xsection`.

        Raises:
            ~spey.system.exceptions.UnknownCrossSection: If
              :attr:`~spey.StatisticalModel.xsection` has not been set (i.e. is ``nan``).

        Returns:
            ``float``:
            Expected 95% CL excluded cross section value in user-defined units.
        """
        return self.excluded_cross_section(ExpectationType.apriori)

    @property
    def s95obs(self) -> float:
        """
        Observed excluded cross section at 95% CL (post-fit / *observed* expectation).

        Shorthand for ``excluded_cross_section(ExpectationType.observed)``.  The result
        represents the cross-section value excluded at the 95% confidence level using the
        actual observed data, expressed in the same units as
        :attr:`~spey.StatisticalModel.xsection`.

        Raises:
            ~spey.system.exceptions.UnknownCrossSection: If
              :attr:`~spey.StatisticalModel.xsection` has not been set (i.e. is ``nan``).

        Returns:
            ``float``:
            Observed 95% CL excluded cross section value in user-defined units.
        """
        return self.excluded_cross_section(ExpectationType.observed)

    def _resolve_poi_test(self, poi_test: PoiTest) -> Union[float, Dict[int, float]]:
        """Resolve a ``poi_test`` dict with string or int keys to ``{index: value}``.

        When ``poi_test`` is a plain ``float`` it is returned unchanged so the
        existing single-POI code paths stay untouched.
        """
        if not isinstance(poi_test, dict):
            return poi_test
        return self.backend.config().resolve_poi_indices(poi_test)

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
                prescription which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescription which means that the experimental data will be assumed to be
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
            **kwargs,
            "func": objective_and_grad,
            "do_grad": do_grad,
            "model_configuration": self.backend.config(
                allow_negative_signal=allow_negative_signal
            ),
            "logpdf": self.backend.get_logpdf_func(expected=expected, data=data),
            "constraints": constraints,
        }

    def likelihood(
        self,
        poi_test: PoiTest = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        data: Optional[Union[List[float], np.ndarray]] = None,
        return_parameters: bool = False,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        r"""
        Compute the likelihood of the statistical model at a fixed parameter of interest.

        Args:
            poi_test (:obj:`~spey.interface.statistical_model.PoiTest`, default ``1.0``):
              Parameter of interest, :math:`\mu`. Can be a single ``float`` (fixes the primary
              POI identified by :attr:`~spey.base.model_config.ModelConfig.poi_index`) or a
              ``dict`` mapping POI indices (``int``) or names (``str``) to their fixed values.
              String keys are resolved via
              :attr:`~spey.base.model_config.ModelConfig.parameter_names`.
              When a ``float`` is given, behaviour is identical to previous versions.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescription which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescription which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            return_nll (``bool``, default ``True``): If ``True``, returns negative log-likelihood value.
              if ``False`` returns likelihood value.
            data (``Union[List[float], np.ndarray]``, default ``None``): input data that to fit. If
              ``None`` data will be set according to ``expected`` input.
            return_parameters (``bool``, default ``False``): Return fit parameters.
            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
              the optimiser.
            kwargs: Keyword arguments forwarded through ``prepare_for_fit`` to the optimiser.
              The following keys are recognised:

              **Consumed by** ``prepare_for_fit``:

              * ``do_grad`` (``bool``, default ``True``): Whether to request the gradient of the
                objective function from the backend. Falls back to ``False`` automatically if the
                backend raises :obj:`NotImplementedError`.
              * ``constraints`` (``List[Dict]``, default ``[]``): Additional
                `scipy-style constraint dicts <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
                to pass to the optimiser. Any constraints defined on the backend itself are
                always appended.

              **Consumed by** ``fit`` (the core optimisation loop):

              * ``minimizer`` (``str``, default ``"scipy"`` or the value of the
                ``SPEY_OPTIMISER`` environment variable): Selects the numerical minimiser.
                Accepted values are ``"scipy"`` and ``"minuit"`` (requires ``iminuit``).
              * ``hessian`` (``Callable[[np.ndarray], np.ndarray]``, default ``None``):
                Hessian of the objective function with respect to the variational parameters.
                Passed to scipy as the ``hess`` argument; ignored by the minuit minimiser.

              **Scipy-minimiser options** (used when ``minimizer="scipy"``):

              * ``method`` (``str``, default ``"SLSQP"``): Scipy optimisation method
                (e.g. ``"SLSQP"``, ``"L-BFGS-B"``, ``"trust-constr"``).
              * ``maxiter`` (``int``, default ``10000``): Maximum number of iterations.
              * ``tol`` (``float``, default ``1e-6``): Convergence tolerance.
              * ``disp`` (``bool``, default ``False``): If ``True``, print convergence
                messages.
              * ``ntrials`` (``int``, default ``1``): Number of re-tries with progressively
                expanded parameter bounds when the minimiser does not converge.

              **Minuit-minimiser options** (used when ``minimizer="minuit"``):

              * ``method`` (``str``, default ``"migrad"``): Minuit algorithm.
                Accepted values are ``"migrad"`` and ``"simplex"``.
              * ``maxiter`` (``int``, default ``10000``): Maximum number of function calls.
              * ``tol`` (``float``, default ``1e-6``): Convergence tolerance.
              * ``disp`` (``int``, default ``0``): Minuit print level (``0`` = silent).
              * ``strategy`` (``int``, default ``0``): Minuit strategy
                (``0`` = fast, ``1`` = default, ``2`` = slow but more accurate).
              * ``errordef`` (``float``, default ``Minuit.LIKELIHOOD``): Value by which
                Minuit defines a one-sigma interval (``0.5`` for NLL, ``1.0`` for :math:`\chi^2`).

              Unknown keys are logged as a warning and silently discarded by the minimiser.

        Returns:
            ``float``:
            Likelihood of the statistical model at a fixed signal strength.
        """
        if "fixed_poi_value" in kwargs:
            log.warning(
                "Passing 'fixed_poi_value' as a keyword argument to likelihood() is not "
                "supported and has been ignored. Use the 'poi_test' argument instead."
            )
            kwargs.pop("fixed_poi_value")

        fit_opts = self.prepare_for_fit(expected=expected, data=data, **kwargs)

        if (
            fit_opts["model_configuration"].npar == 1
            and fit_opts["model_configuration"].poi_index is not None
            and isinstance(poi_test, float)
        ):
            fit_param = np.array([poi_test])
            logpdf = fit_opts["logpdf"](fit_param)
        else:
            logpdf, fit_param = fit(
                **fit_opts,
                initial_parameters=init_pars,
                bounds=par_bounds,
                fixed_poi_value=self._resolve_poi_test(poi_test),
            )
            log.debug(f"fit parameters : \n\t{fit_param}")

        out = -logpdf if return_nll else np.exp(logpdf)
        if return_parameters:
            return out, fit_param

        return out

    def generate_asimov_data(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistic: str = "qtilde",
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> List[float]:
        r"""
        Generate Asimov data for the statistical model. This function generates a set of parameters
        (nuisance and poi i.e. :math:`\theta` and :math:`\mu`) with respect to ``test_statistic`` input
        which determines the value of :math:`\mu` i.e. if ``test_statistic="q0"`` :math:`\mu=1` and 0 for
        anything else. The objective function is used to optimize the statistical model to find the fit
        parameters for fixed poi optimisation. Then fit parameters are used to retrieve the expected
        data through :func:`~spey.BackendBase.expected_data` function.

        Args:
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescription which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescription which means that the experimental data will be assumed to be
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
        if "fixed_poi_value" in kwargs:
            log.warning(
                "Passing 'fixed_poi_value' as a keyword argument to generate_asimov_data() is "
                "not supported and has been ignored. The POI value used for Asimov data "
                "generation is determined by 'test_statistic' (1.0 for 'q0', 0.0 otherwise)."
            )
            kwargs.pop("fixed_poi_value")

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
        log.debug(f"fit parameters:\n\t {fit_pars}")

        return self.backend.expected_data(fit_pars)

    def asimov_likelihood(
        self,
        poi_test: PoiTest = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        test_statistics: str = "qtilde",
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> float:
        r"""
        Compute likelihood of the statistical model generated with the Asimov data.

        Args:
            poi_test (:obj:`~spey.interface.statistical_model.PoiTest`, default ``1.0``):
              Parameter of interest, :math:`\mu`. Accepts the same formats as
              :func:`~spey.StatisticalModel.likelihood`: a plain ``float`` fixes the primary POI,
              while a ``dict`` of ``{index_or_name: value}`` fixes multiple parameters
              simultaneously.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescription which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescription which means that the experimental data will be assumed to be
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
            kwargs: Keyword arguments forwarded to both the Asimov data generation fit
              (via :func:`~spey.StatisticalModel.generate_asimov_data`) and the subsequent
              likelihood evaluation (via :func:`~spey.StatisticalModel.likelihood`).
              Both calls receive an independent copy of ``kwargs``.

              **Consumed by** ``prepare_for_fit``:

              * ``do_grad`` (``bool``, default ``True``): Whether to request the gradient of the
                objective function from the backend. Falls back to ``False`` automatically if the
                backend raises :obj:`NotImplementedError`.
              * ``constraints`` (``List[Dict]``, default ``[]``): Additional
                `scipy-style constraint dicts <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
                to pass to the optimiser. Any constraints defined on the backend itself are
                always appended.

              **Consumed by** ``fit`` (the core optimisation loop):

              * ``minimizer`` (``str``, default ``"scipy"`` or the value of the
                ``SPEY_OPTIMISER`` environment variable): Selects the numerical minimiser.
                Accepted values are ``"scipy"`` and ``"minuit"`` (requires ``iminuit``).
              * ``hessian`` (``Callable[[np.ndarray], np.ndarray]``, default ``None``):
                Hessian of the objective function. Passed to scipy as ``hess``; ignored by
                minuit.

              **Scipy-minimiser options** (used when ``minimizer="scipy"``):

              * ``method`` (``str``, default ``"SLSQP"``): Scipy optimisation method.
              * ``maxiter`` (``int``, default ``10000``): Maximum number of iterations.
              * ``tol`` (``float``, default ``1e-6``): Convergence tolerance.
              * ``disp`` (``bool``, default ``False``): If ``True``, print convergence messages.
              * ``ntrials`` (``int``, default ``1``): Number of re-tries with progressively
                expanded parameter bounds when the minimiser does not converge.

              **Minuit-minimiser options** (used when ``minimizer="minuit"``):

              * ``method`` (``str``, default ``"migrad"``): Minuit algorithm
                (``"migrad"`` or ``"simplex"``).
              * ``maxiter`` (``int``, default ``10000``): Maximum number of function calls.
              * ``tol`` (``float``, default ``1e-6``): Convergence tolerance.
              * ``disp`` (``int``, default ``0``): Minuit print level (``0`` = silent).
              * ``strategy`` (``int``, default ``0``): Minuit strategy
                (``0`` = fast, ``1`` = default, ``2`` = slow but more accurate).
              * ``errordef`` (``float``, default ``Minuit.LIKELIHOOD``): Value by which
                Minuit defines a one-sigma interval (``0.5`` for NLL, ``1.0`` for
                :math:`\chi^2`).

              .. note::

                  ``fixed_poi_value`` is **not** an accepted kwarg here.  The POI used for
                  Asimov data generation is determined by ``test_statistics`` (``1.0`` for
                  ``"q0"``, ``0.0`` otherwise), and ``poi_test`` fixes the POI for the
                  likelihood evaluation.  Passing ``fixed_poi_value`` would cause a
                  :obj:`TypeError` in both inner calls and is therefore intercepted and
                  discarded with a warning.

              Unknown keys are logged as a warning and silently discarded by the minimiser.

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
        poi_indices: Optional[List[Union[int, str]]] = None,
        **kwargs,
    ) -> Tuple[Union[float, Dict[Union[int, str], float]], float]:
        r"""
        Find the maximum of the likelihood.

        Args:
            return_nll (``bool``, default ``True``): If  ``True``, returns negative log-likelihood value.
              if ``False`` returns likelihood value.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescription which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescription which means that the experimental data will be assumed to be
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
            poi_indices (``List[Union[int, str]]``, default ``None``): If ``None``, returns a
              single ``float`` for the primary POI (identified by
              :attr:`~spey.base.model_config.ModelConfig.poi_index`). If a list of parameter
              indices (``int``) or names (``str``) is provided, returns a ``dict`` mapping each
              requested key to its fitted value. String keys are resolved via
              :attr:`~spey.base.model_config.ModelConfig.parameter_names`.
            kwargs: Keyword arguments forwarded through ``prepare_for_fit`` to the optimiser.
              Accepts the same keys as :func:`~spey.StatisticalModel.likelihood`:

              **Consumed by** ``prepare_for_fit``:

              * ``do_grad`` (``bool``, default ``True``): Whether to request the gradient of the
                objective function from the backend. Falls back to ``False`` automatically if the
                backend raises :obj:`NotImplementedError`.
              * ``constraints`` (``List[Dict]``, default ``[]``): Additional
                `scipy-style constraint dicts <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
                to pass to the optimiser. Any constraints defined on the backend itself are
                always appended.
              * ``fixed_poi_value`` (``Union[float, Dict[int, float]]``, default ``None``):
                Fix one or more parameters of interest during the optimisation while allowing
                the remaining parameters (nuisance and other POIs) to be minimised freely.
                A plain ``float`` fixes the primary POI (identified by
                :attr:`~spey.base.model_config.ModelConfig.poi_index`); a ``dict`` of
                ``{index: value}`` fixes multiple POIs simultaneously.  This is particularly
                useful in multi-POI fits where, for example, one signal strength is held fixed
                while others are profiled out.

              **Consumed by** ``fit`` (the core optimisation loop):

              * ``minimizer`` (``str``, default ``"scipy"`` or the value of the
                ``SPEY_OPTIMISER`` environment variable): Selects the numerical minimiser.
                Accepted values are ``"scipy"`` and ``"minuit"`` (requires ``iminuit``).
              * ``hessian`` (``Callable[[np.ndarray], np.ndarray]``, default ``None``):
                Hessian of the objective function with respect to the variational parameters.
                Passed to scipy as the ``hess`` argument; ignored by the minuit minimiser.

              **Scipy-minimiser options** (used when ``minimizer="scipy"``):

              * ``method`` (``str``, default ``"SLSQP"``): Scipy optimisation method.
              * ``maxiter`` (``int``, default ``10000``): Maximum number of iterations.
              * ``tol`` (``float``, default ``1e-6``): Convergence tolerance.
              * ``disp`` (``bool``, default ``False``): If ``True``, print convergence
                messages.
              * ``ntrials`` (``int``, default ``1``): Number of re-tries with progressively
                expanded parameter bounds when the minimiser does not converge.

              **Minuit-minimiser options** (used when ``minimizer="minuit"``):

              * ``method`` (``str``, default ``"migrad"``): Minuit algorithm
                (``"migrad"`` or ``"simplex"``).
              * ``maxiter`` (``int``, default ``10000``): Maximum number of function calls.
              * ``tol`` (``float``, default ``1e-6``): Convergence tolerance.
              * ``disp`` (``int``, default ``0``): Minuit print level.
              * ``strategy`` (``int``, default ``0``): Minuit strategy
                (``0`` = fast, ``1`` = default, ``2`` = slow but more accurate).
              * ``errordef`` (``float``, default ``Minuit.LIKELIHOOD``): Value by which
                Minuit defines a one-sigma interval.

              Unknown keys are logged as a warning and silently discarded by the minimiser.

        Returns:
            ``Tuple[Union[float, Dict[Union[int, str], float]], float]``:
            When ``poi_indices=None``: :math:`\hat\mu` (``float``) and the (negative)
            log-likelihood. When ``poi_indices`` is provided: a ``dict`` of
            ``{index_or_name: fitted_value}`` and the (negative) log-likelihood.
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
        log.debug(f"fit parameters:\n\t {fit_param}")

        nll = -logpdf if return_nll else np.exp(logpdf)
        if poi_indices is None:
            muhat = fit_param[self.backend.config().poi_index]
            return muhat, nll

        config = self.backend.config()
        result: Dict[Union[int, str], float] = {}
        for key in poi_indices:
            if isinstance(key, str):
                if config.parameter_names is None:
                    raise ValueError(
                        "Cannot resolve POI name: parameter_names not set in ModelConfig."
                    )
                idx = config.parameter_names.index(key)
            else:
                idx = int(key)
            result[key] = float(fit_param[idx])
        return result, nll

    def maximize_asimov_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: str = "qtilde",
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        poi_indices: Optional[List[Union[int, str]]] = None,
        **kwargs,
    ) -> Tuple[Union[float, Dict[Union[int, str], float]], float]:
        r"""
        Find the maximum of the likelihood which computed with respect to Asimov data.

        Args:
            return_nll (``bool``, default ``True``): If ``True``, returns negative log-likelihood value.
              if ``False`` returns likelihood value.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescription which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescription which means that the experimental data will be assumed to be
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
            poi_indices (``List[Union[int, str]]``, default ``None``): If ``None``, returns the
              primary POI value as a single ``float``. If a list of parameter indices (``int``) or
              names (``str``) is provided, returns a ``dict`` mapping each requested key to its
              fitted value. Passed directly to
              :func:`~spey.StatisticalModel.maximize_likelihood`.
            kwargs: Keyword arguments forwarded to both the Asimov data generation fit
              (via :func:`~spey.StatisticalModel.generate_asimov_data`) and the subsequent
              maximisation (via :func:`~spey.StatisticalModel.maximize_likelihood`).
              Both calls receive an independent copy of ``kwargs``.

              **Consumed by** ``prepare_for_fit``:

              * ``do_grad`` (``bool``, default ``True``): Whether to request the gradient of the
                objective function from the backend. Falls back to ``False`` automatically if the
                backend raises :obj:`NotImplementedError`.
              * ``constraints`` (``List[Dict]``, default ``[]``): Additional
                `scipy-style constraint dicts <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
                to pass to the optimiser. Any constraints defined on the backend itself are
                always appended.

              **Consumed by** ``fit`` (the core optimisation loop):

              * ``minimizer`` (``str``, default ``"scipy"`` or the value of the
                ``SPEY_OPTIMISER`` environment variable): Selects the numerical minimiser.
                Accepted values are ``"scipy"`` and ``"minuit"`` (requires ``iminuit``).
              * ``hessian`` (``Callable[[np.ndarray], np.ndarray]``, default ``None``):
                Hessian of the objective function. Passed to scipy as ``hess``; ignored by
                minuit.
              * ``fixed_poi_value`` (``Union[float, Dict[int, float]]``, default ``None``):
                Fix one or more POIs during the **maximisation** step while allowing the
                remaining parameters to be profiled freely.  A plain ``float`` fixes the
                primary POI; a ``dict`` of ``{index: value}`` fixes multiple POIs
                simultaneously.  This kwarg is intercepted and discarded (with a warning)
                in the Asimov data generation step, where the POI is already determined by
                ``test_statistics``.

              **Scipy-minimiser options** (used when ``minimizer="scipy"``):

              * ``method`` (``str``, default ``"SLSQP"``): Scipy optimisation method.
              * ``maxiter`` (``int``, default ``10000``): Maximum number of iterations.
              * ``tol`` (``float``, default ``1e-6``): Convergence tolerance.
              * ``disp`` (``bool``, default ``False``): If ``True``, print convergence messages.
              * ``ntrials`` (``int``, default ``1``): Number of re-tries with progressively
                expanded parameter bounds when the minimiser does not converge.

              **Minuit-minimiser options** (used when ``minimizer="minuit"``):

              * ``method`` (``str``, default ``"migrad"``): Minuit algorithm
                (``"migrad"`` or ``"simplex"``).
              * ``maxiter`` (``int``, default ``10000``): Maximum number of function calls.
              * ``tol`` (``float``, default ``1e-6``): Convergence tolerance.
              * ``disp`` (``int``, default ``0``): Minuit print level (``0`` = silent).
              * ``strategy`` (``int``, default ``0``): Minuit strategy
                (``0`` = fast, ``1`` = default, ``2`` = slow but more accurate).
              * ``errordef`` (``float``, default ``Minuit.LIKELIHOOD``): Value by which
                Minuit defines a one-sigma interval (``0.5`` for NLL, ``1.0`` for
                :math:`\chi^2`).

              Unknown keys are logged as a warning and silently discarded by the minimiser.

        Returns:
            ``Tuple[Union[float, Dict[Union[int, str], float]], float]``:
            When ``poi_indices=None``: :math:`\hat\mu` (``float``) and the (negative)
            log-likelihood. When ``poi_indices`` is provided: a ``dict`` of
            ``{index_or_name: fitted_value}`` and the (negative) log-likelihood.
        """
        allow_negative_signal: bool = test_statistics in ["q", "qmu"]

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
            poi_indices=poi_indices,
            **kwargs,
        )

    def fixed_poi_sampler(
        self,
        poi_test: PoiTest,
        size: Optional[int] = None,
        expected: ExpectationType = ExpectationType.observed,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> Union[np.ndarray, Callable[[int], np.ndarray]]:
        r"""
        Sample data from the statistical model with fixed parameter of interest.

        Args:
            poi_test (:obj:`~spey.interface.statistical_model.PoiTest`):
              Parameter of interest, :math:`\mu`. A plain ``float`` fixes the primary POI;
              a ``dict`` of ``{index_or_name: value}`` fixes multiple parameters at once.
            size (``int``, default ``None``): sample size. If ``None`` a callable function
              will be returned which takes sample size as input.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescription which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescription which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
              the optimiser.
            kwargs: keyword arguments for the optimiser.

        Raises:
            ~spey.system.exceptions.MethodNotAvailable: If backend does not have sampler implementation.

        Returns:
            ``Union[np.ndarray, Callable[[int], np.ndarray]]``:
            Sampled data with shape of ``(size, number of bins)`` or callable function to sample from
            directly.
        """
        fit_opts = self.prepare_for_fit(expected=expected, **kwargs)

        if (
            fit_opts["model_configuration"].npar == 1
            and fit_opts["model_configuration"].poi_index is not None
            and isinstance(poi_test, float)
        ):
            fit_param = np.array([poi_test])
        else:
            _, fit_param = fit(
                **fit_opts,
                initial_parameters=init_pars,
                bounds=par_bounds,
                fixed_poi_value=self._resolve_poi_test(poi_test),
            )

        log.debug(f"fit parameters:\n\t {fit_param}")
        try:
            sampler = self.backend.get_sampler(fit_param)
        except NotImplementedError as exc:
            raise MethodNotAvailable(
                f"{self.backend_type} backend does not have sampling capabilities"
            ) from exc

        return sampler(size) if isinstance(size, int) else sampler

    def sigma_mu_from_hessian(
        self,
        poi_test: PoiTest,
        expected: ExpectationType = ExpectationType.observed,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> float:
        r"""
        Compute variance of :math:`\mu` from inverse Hessian. See eq. (27-28) in :xref:`1007.1727`.

        Args:
            poi_test (:obj:`~spey.interface.statistical_model.PoiTest`):
              Parameter of interest, :math:`\mu`. A plain ``float`` fixes the primary POI;
              a ``dict`` of ``{index_or_name: value}`` fixes multiple parameters simultaneously.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescription which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescription which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
              the optimiser.
            kwargs: keyword arguments for the optimiser.
        Raises:
            ~spey.system.exceptions.MethodNotAvailable: If backend does not have Hessian implementation.

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
            fixed_poi_value=self._resolve_poi_test(poi_test),
        )
        log.debug(f"fit parameters:\n\t {fit_param}")

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
        Combine two statistical models using the ``@`` operator.

        Equivalent to calling :func:`~spey.StatisticalModel.combine`.  The combined
        model inherits its analysis label from both operands:

        .. code-block:: python

            combined = model_a @ model_b
            # same as: combined = model_a.combine(model_b)

        Args:
            other (:obj:`~spey.StatisticalModel`): Statistical model to combine with.

        Raises:
            ~spey.system.exceptions.CombinerNotAvailable: If this model's backend does
              not implement a combination routine.

        Returns:
            :obj:`~spey.StatisticalModel`:
            A new combined statistical model.
        """
        return self.combine(other)


def statistical_model_wrapper(
    func: BackendBase,
) -> Callable[[Any], StatisticalModel]:
    """
    Decorator that promotes a :class:`~spey.BackendBase` constructor into a
    :class:`~spey.StatisticalModel` factory.

    :func:`~spey.get_backend` applies this decorator automatically before returning a
    backend to the user, so direct use is only required when registering a custom
    backend outside of ``spey``'s plugin system.

    The returned callable accepts all backend-specific positional and keyword arguments
    plus the three universal keyword arguments documented below, and returns a fully
    initialised :class:`~spey.StatisticalModel`.

    Example usage for custom backend registration:

    .. code-block:: python

        from spey.interface.statistical_model import statistical_model_wrapper
        from my_package import MyBackend

        MyModel = statistical_model_wrapper(MyBackend)
        model = MyModel(
            *backend_args,
            analysis="my_analysis",
            xsection=0.05,
        )

    Args:
        func (~spey.BackendBase): Backend class (or callable) whose constructor will be
          wrapped.  Must produce an instance that inherits :class:`~spey.BackendBase`.

    Raises:
        :obj:`AssertionError`: If the object returned by ``func`` does not inherit
          :class:`~spey.BackendBase`.

    Returns:
        ``Callable[[Any], StatisticalModel]``:
        A wrapper callable that accepts the following inputs:

        * **\\*args**: Backend-specific positional arguments forwarded to ``func``.
        * **analysis** (``str``, default ``"__unknown_analysis__"``): Unique identifier
          of the statistical model used for book-keeping purposes.
        * **xsection** (``float``, default ``nan``): Signal cross section in user-defined
          units.  Only required for cross-section upper-limit computations.
        * **ntoys** (``int``, default ``1000``): Number of toy pseudo-experiments for
          toy-based hypothesis testing.
        * **\\*\\*kwargs**: Backend-specific keyword arguments forwarded to ``func``.
    """

    @wraps(func)
    def wrapper(
        *args,
        analysis: str = "__unknown_analysis__",
        xsection: float = np.nan,
        ntoys: int = 1000,
        **kwargs,
    ) -> StatisticalModel:
        return StatisticalModel(
            backend=func(*args, **kwargs),
            analysis=analysis,
            xsection=xsection,
            ntoys=ntoys,
        )

    docstring = (
        "\n\n"
        + "<>" * 30
        + "\n\n"
        + """
        Universal keyword arguments added by :func:`~spey.statistical_model_wrapper`:

        Args:
            analysis (``str``, default ``"__unknown_analysis__"``): Unique identifier of the
              statistical model used for book-keeping purposes.
            xsection (``float``, default ``nan``): Signal cross section in user-defined units.
              Only required for cross-section upper-limit computations (e.g.
              :attr:`~spey.StatisticalModel.s95obs`).
            ntoys (``int``, default ``1000``): Number of toy pseudo-experiments used by the
              toy-based hypothesis-testing calculator.  Ignored for asymptotic and
              :math:`\\chi^2` calculators.

        Raises:
            :obj:`AssertionError`: If the backend instance does not inherit
              :class:`~spey.BackendBase`.

        Returns:
            ~spey.StatisticalModel:
            Backend wrapped with the unified :class:`~spey.StatisticalModel` interface.
        """
    )

    if wrapper.__doc__ is not None:
        wrapper.__doc__ += docstring
    else:
        wrapper.__doc__ = docstring

    return wrapper
