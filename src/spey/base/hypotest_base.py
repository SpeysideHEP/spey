r"""
Abstract base class for hypothesis testing in ``spey``.

This module defines :class:`~spey.base.hypotest_base.HypothesisTestingBase`, which
provides the complete hypothesis-testing API built on top of any statistical model
backend.  :class:`~spey.StatisticalModel` inherits from this class, so every backend
automatically receives the following capabilities as soon as it implements the two
mandatory methods (:func:`~spey.BackendBase.config` and
:func:`~spey.BackendBase.get_logpdf_func`):

* :math:`\chi^2` test statistics and profile likelihood ratio computation;
* exclusion confidence levels (:math:`CL_s`) via asymptotic, toy, or :math:`\chi^2`
  calculators;
* one-sided and two-sided :math:`\chi^2` interval finding;
* POI upper limits at arbitrary confidence levels;
* discovery significance (:math:`\sqrt{q_0}`).

"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import tqdm
from scipy.stats import chi2

from spey.hypothesis_testing.asymptotic_calculator import (
    compute_asymptotic_confidence_level,
)
from spey.hypothesis_testing.test_statistics import (
    compute_teststatistics,
    get_test_statistic,
)
from spey.hypothesis_testing.toy_calculator import compute_toy_confidence_level
from spey.hypothesis_testing.upper_limits import bracket_and_solve, find_poi_upper_limit
from spey.system.exceptions import (
    AsimovTestStatZero,
    CalculatorNotAvailable,
    MethodNotAvailable,
    warning_tracker,
)
from spey.system.logger import capture_logs
from spey.utils import ExpectationType

from .utils import resolve_parameter_index

PoiTest = Union[float, Dict[Union[int, str], float]]

__all__ = ["HypothesisTestingBase"]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=W1203,C0103,possibly-used-before-assignment


class HypothesisTestingBase(ABC):
    r"""
    Abstract base class that provides the full hypothesis-testing API for ``spey``.

    Any class that inherits :class:`~spey.base.hypotest_base.HypothesisTestingBase`
    and implements the four abstract properties and four abstract methods listed below
    gains the complete set of hypothesis-testing utilities shown in the table.

    **Abstract interface** (must be implemented by subclasses)

    .. list-table::
        :header-rows: 1
        :widths: 45 55

        * - Member
          - Description
        * - :attr:`is_alive`
          - ``True`` if the signal hypothesis is non-trivially zero.
        * - :attr:`is_asymptotic_calculator_available`
          - ``True`` if the asymptotic calculator can be used.
        * - :attr:`is_toy_calculator_available`
          - ``True`` if the toy calculator can be used.
        * - :attr:`is_chi_square_calculator_available`
          - ``True`` if the :math:`\chi^2` calculator can be used.
        * - :func:`likelihood`
          - :math:`-\log\mathcal{L}` or :math:`\mathcal{L}` at fixed :math:`\mu`.
        * - :func:`maximize_likelihood`
          - Global minimisation of :math:`-\log\mathcal{L}` (free fit).
        * - :func:`asimov_likelihood`
          - :math:`-\log\mathcal{L}` on Asimov data at fixed :math:`\mu`.
        * - :func:`maximize_asimov_likelihood`
          - Global minimisation on Asimov data (free fit).

    **Concrete capabilities** (provided automatically)

    .. list-table::
        :header-rows: 1
        :widths: 45 55

        * - Method
          - Description
        * - :func:`chi2`
          - Profile likelihood ratio :math:`\chi^2`.
        * - :func:`exclusion_confidence_level`
          - :math:`CL_s` at a fixed :math:`\mu` (asymptotic, toy, or :math:`\chi^2`).
        * - :func:`significance`
          - Discovery significance :math:`\sqrt{q_0}`.
        * - :func:`poi_upper_limit`
          - One-sided 95% (or other) CL upper limit on :math:`\mu`.
        * - :func:`chi2_test`
          - One- or two-sided :math:`\chi^2` interval on :math:`\mu`.
        * - :func:`sigma_mu`
          - Standard deviation of :math:`\hat\mu` from Hessian or Asimov approximation.

    **Usage examples**

    All of the concrete methods below are available on :class:`~spey.StatisticalModel`,
    which inherits this class:

    .. code-block:: python

        import spey

        pdf = spey.get_backend("default.poisson")
        model = pdf(
            signal_yields=[5.0, 3.0],
            background_yields=[50.0, 30.0],
            data=[55, 31],
            analysis="example",
            xsection=0.05,
        )

        # Exclusion confidence level (CLs) at mu = 1
        cls_obs = model.exclusion_confidence_level(poi_test=1.0)

        # Expected CLs (5 values: -2s, -1s, central, +1s, +2s)
        cls_exp = model.exclusion_confidence_level(
            poi_test=1.0, expected=spey.ExpectationType.apriori
        )

        # 95% CL upper limit on mu
        mu_ul = model.poi_upper_limit(confidence_level=0.95)

        # Discovery significance
        sqrt_q0A, sqrt_q0, pvals, exp_pvals = model.significance()

        # Two-sided chi^2 interval at 68% CL
        mu_lo, mu_hi = model.chi2_test(confidence_level=0.68, limit_type="two-sided")

    Args:
        ntoys (``int``, default ``1000``): Number of pseudo-experiments (toys) used by
          the toy-based calculator.  Ignored when the asymptotic or :math:`\chi^2`
          calculator is selected.
    """

    __slots__ = ["ntoys"]

    def __init__(self, ntoys: int = 1000):
        self.ntoys = ntoys
        """Number of toy pseudo-experiments used by the toy-based calculator."""

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """
        Whether the signal hypothesis has at least one non-zero bin yield.

        Used as a fast pre-check before expensive likelihood evaluations.
        Must be implemented as a property by all subclasses.

        Returns:
            ``bool``:
            ``True`` if the model has at least one non-zero signal bin.
        """
        # This method has to be a property

    @property
    @abstractmethod
    def is_asymptotic_calculator_available(self) -> bool:
        """
        Whether the asymptotic calculator is available for this model.

        Must be implemented as a property by all subclasses.

        Returns:
            ``bool``:
            ``True`` if the asymptotic calculator can be used.
        """
        # This method has to be a property

    @property
    @abstractmethod
    def is_toy_calculator_available(self) -> bool:
        """
        Whether the toy (pseudo-experiment) calculator is available for this model.

        Must be implemented as a property by all subclasses.

        Returns:
            ``bool``:
            ``True`` if the toy calculator can be used.
        """
        # This method has to be a property

    @property
    @abstractmethod
    def is_chi_square_calculator_available(self) -> bool:
        r"""
        Whether the :math:`\chi^2` calculator is available for this model.

        Must be implemented as a property by all subclasses.

        Returns:
            ``bool``:
            ``True`` if the :math:`\chi^2` calculator can be used.
        """
        # This method has to be a property

    @abstractmethod
    def likelihood(
        self,
        poi_test: PoiTest = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        data: Optional[Union[List[float], np.ndarray]] = None,
        return_parameters: bool = False,
        **kwargs,
    ) -> float:
        r"""
        Compute the (negative) log-likelihood at a fixed parameter of interest.

        The nuisance parameters :math:`\theta` are profiled (i.e. minimised over)
        at the given :math:`\mu`, returning the profile likelihood
        :math:`-\log\mathcal{L}(\mu, \hat{\theta}_\mu)`.

        Args:
            poi_test (:obj:`PoiTest`, default ``1.0``): Parameter of interest
              :math:`\mu`.  A plain ``float`` fixes the primary POI (identified by
              :attr:`~spey.base.model_config.ModelConfig.poi_index`); a ``dict`` of
              ``{index_or_name: value}`` fixes multiple parameters simultaneously.
              String keys are resolved via
              :attr:`~spey.base.model_config.ModelConfig.parameter_names`.
            expected (~spey.ExpectationType): Selects which dataset to condition on.

              * :obj:`~spey.ExpectationType.observed`: Use observed data (post-fit,
                default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use observed data with
                post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Use background-only prediction
                (pre-fit / SM hypothesis).

            return_nll (``bool``, default ``True``): If ``True``, return the negative
              log-likelihood :math:`-\log\mathcal{L}`; if ``False``, return the
              likelihood :math:`\mathcal{L}`.
            data (``Union[List[float], np.ndarray]``, default ``None``): Explicit
              dataset to condition on.  When provided, overrides ``expected``.
            return_parameters (``bool``, default ``False``): Return fit parameters.
            kwargs: Additional keyword arguments forwarded to the optimiser.

        Returns:
            ``float``:
            The (negative) log-likelihood at the fixed signal strength :math:`\mu`.
        """

    @abstractmethod
    def maximize_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = True,
        data: Optional[Union[List[float], np.ndarray]] = None,
        poi_indices: Optional[List[Union[int, str]]] = None,
        **kwargs,
    ) -> Tuple[Union[float, Dict[Union[int, str], float]], float]:
        r"""
        Find the global maximum of the likelihood (free fit).

        Minimises :math:`-\log\mathcal{L}(\mu, \theta)` over all parameters,
        returning :math:`\hat\mu` and the minimum negative log-likelihood
        :math:`-\log\mathcal{L}(\hat\mu, \hat\theta)`.

        Args:
            return_nll (``bool``, default ``True``): If ``True``, return the negative
              log-likelihood; if ``False``, return the likelihood value.
            expected (~spey.ExpectationType): Selects which dataset to condition on.

              * :obj:`~spey.ExpectationType.observed`: Use observed data (post-fit,
                default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use observed data with
                post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Use background-only prediction
                (pre-fit / SM hypothesis).

            allow_negative_signal (``bool``, default ``True``): When ``True``,
              :math:`\hat\mu` is unconstrained; when ``False`` the fit enforces
              :math:`\hat\mu \geq 0`.
            data (``Union[List[float], np.ndarray]``, default ``None``): Explicit
              dataset to condition on.  When provided, overrides ``expected``.
            poi_indices (``List[Union[int, str]]``, default ``None``): When ``None``,
              returns the primary POI value as a single ``float``.  When a list of
              integer indices or string parameter names is given, returns a ``dict``
              mapping each requested key to its fitted value.
            kwargs: Additional keyword arguments forwarded to the optimiser.

        Returns:
            ``Tuple[Union[float, Dict[Union[int, str], float]], float]``:
            ``(muhat, nll)`` where ``muhat`` is either a ``float`` (single POI) or a
            ``dict`` of ``{index_or_name: fitted_value}`` (multiple POIs), and ``nll``
            is the (negative) log-likelihood at the optimum.
        """

    @abstractmethod
    def asimov_likelihood(
        self,
        poi_test: PoiTest = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        test_statistics: Literal["qtilde", "q", "q0"] = "qtilde",
        **kwargs,
    ) -> float:
        r"""
        Compute the (negative) log-likelihood on Asimov data at a fixed :math:`\mu`.

        The Asimov dataset is generated by fitting the nuisance parameters to the
        background-only or signal-plus-background hypothesis (controlled by
        ``test_statistics``) and then computing the expected bin counts.  The
        likelihood is then evaluated on this synthetic dataset instead of the observed
        data.

        Args:
            poi_test (:obj:`PoiTest`, default ``1.0``): Parameter of interest
              :math:`\mu`.  Accepts the same formats as :func:`likelihood`: a plain
              ``float`` or a ``dict`` of ``{index_or_name: value}`` to fix multiple
              parameters simultaneously.
            expected (~spey.ExpectationType): Selects which dataset is used to produce
              the Asimov data.

              * :obj:`~spey.ExpectationType.observed`: Post-fit (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Pre-fit / SM hypothesis.

            return_nll (``bool``, default ``True``): If ``True``, return the negative
              log-likelihood; if ``False``, return the likelihood value.
            test_statistics (``str``, default ``"qtilde"``): Test statistic that
              determines the :math:`\mu` value used for Asimov-data generation
              (``"q0"`` → :math:`\mu=1`; all others → :math:`\mu=0`).

              * ``'qtilde'``: Alternative test statistic :math:`\tilde{q}_\mu`,
                eq. (62) of :xref:`1007.1727`.

                .. warning::

                    This assumes :math:`\hat\mu \geq 0` (``allow_negative_signal=False``).
                    When called through ``spey``'s public API this constraint is enforced
                    automatically.

              * ``'q'``: Standard test statistic :math:`q_\mu`,
                eq. (54) of :xref:`1007.1727`.
              * ``'q0'``: Discovery test statistic :math:`q_0`,
                eq. (47) of :xref:`1007.1727`.

            kwargs: Additional keyword arguments forwarded to the optimiser.

        Returns:
            ``float``:
            The (negative) log-likelihood evaluated on the Asimov dataset at the
            fixed signal strength :math:`\mu`.
        """

    @abstractmethod
    def maximize_asimov_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Literal["qtilde", "q", "q0"] = "qtilde",
        poi_indices: Optional[List[Union[int, str]]] = None,
        **kwargs,
    ) -> Tuple[Union[float, Dict[Union[int, str], float]], float]:
        r"""
        Find the global maximum of the likelihood on Asimov data (free fit).

        Analogous to :func:`maximize_likelihood` but evaluated on the Asimov dataset
        generated according to ``test_statistics``.  The result is used internally
        by :func:`_prepare_for_hypotest` to build the asymptotic test statistic.

        Args:
            return_nll (``bool``, default ``True``): If ``True``, return the negative
              log-likelihood; if ``False``, return the likelihood value.
            expected (~spey.ExpectationType): Selects which dataset is used to produce
              the Asimov data.

              * :obj:`~spey.ExpectationType.observed`: Post-fit (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Pre-fit / SM hypothesis.

            test_statistics (``str``, default ``"qtilde"``): Test statistic that
              determines the :math:`\mu` value used for Asimov-data generation
              (``"q0"`` → :math:`\mu=1`; all others → :math:`\mu=0`).

              * ``'qtilde'``: Alternative test statistic :math:`\tilde{q}_\mu`,
                eq. (62) of :xref:`1007.1727`.

                .. warning::

                    This assumes :math:`\hat\mu \geq 0` (``allow_negative_signal=False``).
                    When called through ``spey``'s public API this constraint is enforced
                    automatically.

              * ``'q'``: Standard test statistic :math:`q_\mu`,
                eq. (54) of :xref:`1007.1727`.
              * ``'q0'``: Discovery test statistic :math:`q_0`,
                eq. (47) of :xref:`1007.1727`.

            poi_indices (``List[Union[int, str]]``, default ``None``): When ``None``,
              returns the primary POI value as a single ``float``.  When a list of
              integer indices or string parameter names is given, returns a ``dict``
              mapping each requested key to its fitted value.
            kwargs: Additional keyword arguments forwarded to the optimiser.

        Returns:
            ``Tuple[Union[float, Dict[Union[int, str], float]], float]``:
            ``(muhat_A, nll_A)`` where ``muhat_A`` is either a ``float`` (single POI)
            or a ``dict`` of ``{index_or_name: fitted_value}`` (multiple POIs), and
            ``nll_A`` is the (negative) log-likelihood on the Asimov dataset at the
            optimum.
        """

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
            poi_test (:obj:`PoiTest`): parameter of interest or signal strength, :math:`\mu`.
              Either a single ``float`` or a ``dict`` mapping POI indices/names to values.
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
            ~spey.system.exceptions.MethodNotAvailable: If the backend does not have a sampler implementation.

        Returns:
            ``Union[np.ndarray, Callable[[int], np.ndarray]]``:
            Sampled data with shape of ``(size, number of bins)`` or callable function to sample from
            directly.
        """
        raise NotImplementedError("This method has not been implemented")

    def chi2(
        self,
        poi_test: PoiTest = 1.0,
        poi_test_denominator: Optional[PoiTest] = None,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = False,
        **kwargs,
    ) -> float:
        r"""
        Compute the profile likelihood ratio :math:`\chi^2` test statistic.

        When ``poi_test_denominator=None``, evaluates the profile likelihood ratio
        against the unconditional maximum:

        .. math::

            \chi^2 = -2\log\left(\frac{\mathcal{L}(\mu,\hat\theta_\mu)}{\mathcal{L}(\hat\mu,\hat\theta)}\right)

        When ``poi_test_denominator`` is set, it replaces the denominator with a
        second fixed-:math:`\mu` likelihood:

        .. math::

            \chi^2 = -2\log\left(\frac{\mathcal{L}(\mu,\theta_\mu)}{\mathcal{L}(\mu_{\rm denom},\theta_{\mu_{\rm denom}})}\right)

        where :math:`\mu_{\rm denom}` is ``poi_test_denominator`` which is typically zero to compare signal
        model with the background only model.

        Args:
            poi_test (:obj:`PoiTest` or ``list[float]``, default ``1.0``): Parameter of interest,
              :math:`\mu`. A plain ``float`` (or iterable of floats) fixes the primary POI —
              when iterable, :math:`\chi^2` is computed for each element. Alternatively, a
              ``dict`` of ``{index_or_name: value}`` fixes multiple parameters simultaneously
              (iterating over dicts is not supported).
            poi_test_denominator (:obj:`PoiTest`, default ``None``): Parameter of interest for the
              denominator. Accepts the same formats as ``poi_test``.
              If ``None`` the maximum likelihood is computed instead.
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
              value will be allowed to be negative. Only valid when ``poi_test_denominator=None``.
            kwargs: keyword arguments for the optimiser.

        Returns:
            ``float``:
            value of the :math:`\chi^2`.
        """
        if poi_test_denominator is None:
            _, denominator = self.maximize_likelihood(
                expected=expected, allow_negative_signal=allow_negative_signal, **kwargs
            )
        else:
            denominator = self.likelihood(
                poi_test=poi_test_denominator, expected=expected, **kwargs
            )
        log.debug(f"denominator: {denominator}")

        if isinstance(poi_test, Iterable):
            with capture_logs(logging.INFO) as _:
                llhd = np.fromiter(
                    (
                        self.likelihood(poi_test=float(p), expected=expected, **kwargs)
                        for p in np.atleast_1d(poi_test)
                    ),
                    np.float32,
                )
        else:
            llhd = self.likelihood(poi_test=poi_test, expected=expected, **kwargs)

        return 2.0 * (llhd - denominator)

    def _prepare_for_hypotest(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Literal["qtilde", "q", "q0"] = "qtilde",
        **kwargs,
    ) -> Tuple[
        Tuple[float, float],
        Callable[[float], float],
        Tuple[float, float],
        Callable[[float], float],
    ]:
        r"""
        Compute the four ingredients needed for asymptotic hypothesis testing.

        Evaluates :func:`maximize_likelihood` and :func:`maximize_asimov_likelihood`
        and wraps :func:`likelihood` and :func:`asimov_likelihood` as callables
        suitable for :func:`~spey.hypothesis_testing.test_statistics.compute_teststatistics`.
        All public hypothesis-testing methods call this helper internally.

        Args:
            expected (~spey.ExpectationType): Selects which dataset to condition on.

              * :obj:`~spey.ExpectationType.observed`: Use observed data (post-fit,
                default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use observed data with
                post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Use background-only prediction
                (pre-fit / SM hypothesis).

            test_statistics (``str``, default ``"qtilde"``): Test statistic choice;
              controls ``allow_negative_signal`` and Asimov-data generation.

              * ``'qtilde'``: :math:`\tilde{q}_\mu`, eq. (62) of :xref:`1007.1727`.

                .. warning::

                    This assumes :math:`\hat\mu \geq 0`.  ``spey``'s public API
                    enforces this automatically.

              * ``'q'``: :math:`q_\mu`, eq. (54) of :xref:`1007.1727`.
              * ``'q0'``: Discovery statistic :math:`q_0`, eq. (47) of :xref:`1007.1727`.

            kwargs: Additional keyword arguments forwarded to the optimiser, including:

              * **init_pars** (``List[float]``, default ``None``): Initial parameter
                values for the optimiser.
              * **par_bounds** (``List[Tuple[float, float]]``, default ``None``):
                Parameter bounds for the optimiser.

        Returns:
            ``Tuple[Tuple[float, float], Callable, Tuple[float, float], Callable]``:
            A 4-tuple:

            * ``(hat_mu, nll)`` — best-fit POI and minimum NLL on observed data.
            * ``logpdf(mu)`` — callable returning :math:`-\log\mathcal{L}(\mu)` on
              observed data.
            * ``(hat_mu_A, nll_A)`` — best-fit POI and minimum NLL on Asimov data.
            * ``logpdf_asimov(mu)`` — callable returning :math:`-\log\mathcal{L}_A(\mu)`
              on Asimov data.
        """
        allow_negative_signal = test_statistics in ["q" or "qmu"]
        log.debug("Computing max-llhd")
        muhat, nll = self.maximize_likelihood(
            expected=expected,
            allow_negative_signal=allow_negative_signal,
            **kwargs,
        )
        log.debug(f"muhat: {muhat}, nll: {nll}")
        log.debug("Computing max-llhd for Asimov data")
        muhatA, nllA = self.maximize_asimov_likelihood(
            expected=expected,
            test_statistics=test_statistics,
            **kwargs,
        )
        log.debug(f"muhatA: {muhatA}, nllA: {nllA}")

        def logpdf(mu: Union[float, np.ndarray]) -> float:
            return -self.likelihood(
                poi_test=float(mu) if isinstance(mu, (float, int)) else mu[0],
                expected=expected,
                **kwargs,
            )

        def logpdf_asimov(mu: Union[float, np.ndarray]) -> float:
            return -self.asimov_likelihood(
                poi_test=float(mu) if isinstance(mu, (float, int)) else mu[0],
                expected=expected,
                test_statistics=test_statistics,
                **kwargs,
            )

        return (muhat, nll), logpdf, (muhatA, nllA), logpdf_asimov

    def sigma_mu(
        self,
        poi_test: PoiTest,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Literal["qtilde", "q", "q0"] = "qtilde",
        **kwargs,
    ) -> float:
        r"""
        Estimate the standard deviation of :math:`\hat\mu` at a fixed :math:`\mu`.

        Attempts the Hessian-based estimate first (via
        :func:`~spey.StatisticalModel.sigma_mu_from_hessian`) if that method exists on
        the subclass.  When the Hessian is not available, falls back to the Asimov
        approximation from eq. (31) of :xref:`1007.1727`:

        .. math::

            \sigma_A = \frac{|\mu - \mu^\prime|}{\sqrt{q_{\mu,A}}},
            \qquad q_{\mu,A} = -2\ln\lambda_A(\mu)

        where :math:`\mu^\prime` is the best-fit value on the Asimov dataset.

        Args:
            poi_test (:obj:`PoiTest`): Parameter of interest value :math:`\mu` at
              which to evaluate :math:`\sigma_\mu`.
            expected (~spey.ExpectationType): Selects which dataset to condition on.

              * :obj:`~spey.ExpectationType.observed`: Use observed data (post-fit,
                default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use observed data with
                post-fit nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Use background-only prediction
                (pre-fit / SM hypothesis).

            test_statistics (``str``, default ``"qtilde"``): Test statistic used for
              the Asimov approximation (ignored when the Hessian path is taken).

              * ``'qtilde'``: :math:`\tilde{q}_\mu`, eq. (62) of :xref:`1007.1727`.

                .. warning::

                    This assumes :math:`\hat\mu \geq 0`.  ``spey``'s public API
                    enforces this automatically.

              * ``'q'``: :math:`q_\mu`, eq. (54) of :xref:`1007.1727`.
              * ``'q0'``: Discovery statistic :math:`q_0`, eq. (47) of :xref:`1007.1727`.

            kwargs: Additional keyword arguments forwarded to the optimiser, including:

              * **init_pars** (``List[float]``, default ``None``): Initial parameter
                values for the optimiser.
              * **par_bounds** (``List[Tuple[float, float]]``, default ``None``):
                Parameter bounds for the optimiser.

        Returns:
            ``float``:
            Estimated standard deviation :math:`\sigma_\mu` of the parameter of
            interest at the given :math:`\mu`.
        """
        if hasattr(self, "sigma_mu_from_hessian"):
            try:
                return self.sigma_mu_from_hessian(
                    poi_test=poi_test, expected=expected, **kwargs
                )
            except MethodNotAvailable:
                log.warning(
                    "Hessian implementation is not available for this backend, "
                    "continuing with the approximate method."
                )
        teststat_func = get_test_statistic(test_statistics)

        muhatA, min_nllA = self.maximize_asimov_likelihood(
            expected=expected, test_statistics=test_statistics, **kwargs
        )
        log.debug(f"muhatA: {muhatA}, min_nllA: {min_nllA}")

        def logpdf_asimov(mu: Union[float, np.ndarray]) -> float:
            return -self.asimov_likelihood(
                poi_test=mu if isinstance(mu, float) else mu[0],
                expected=expected,
                test_statistics=test_statistics,
                **kwargs,
            )

        qmuA = teststat_func(poi_test, muhatA, -min_nllA, logpdf_asimov)

        return 1.0 if qmuA <= 0.0 else np.true_divide(poi_test, np.sqrt(qmuA))

    @warning_tracker
    def exclusion_confidence_level(
        self,
        poi_test: PoiTest = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = False,
        calculator: Literal["asymptotic", "toy", "chi_square"] = "asymptotic",
        **kwargs,
    ) -> List[float]:
        r"""
        Compute the exclusion confidence level :math:`CL_s` at a given :math:`\mu`.

        :math:`CL_s` is defined as

        .. math::

            CL_s = \frac{p_{s+b}}{1 - p_b}

        and is returned as :math:`1 - p\text{-value}`.  The number of returned values
        depends on the ``expected`` mode:

        * :obj:`~spey.ExpectationType.observed` → one value (fitted to observed data).
        * :obj:`~spey.ExpectationType.aposteriori` / :obj:`~spey.ExpectationType.apriori`
          → five values representing :math:`-2\sigma,\,-1\sigma,\,\text{central},\,+1\sigma,\,+2\sigma`
          fluctuations from the background.

        Args:
            poi_test (:obj:`PoiTest`, default ``1.0``): Parameter of interest
              :math:`\mu` at which to evaluate :math:`CL_s`.
            expected (~spey.ExpectationType): Selects the expectation mode.

              * :obj:`~spey.ExpectationType.observed`: Post-fit, returns one value
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Post-fit nuisance treatment,
                returns five expected values.
              * :obj:`~spey.ExpectationType.apriori`: Pre-fit / SM hypothesis, returns
                five expected values.

              Setting :code:`expected="all"` returns both the observed and the five expected
              values simultaneously.

            allow_negative_signal (``bool``, default ``False``): When ``True``,
              :math:`\hat\mu` is unconstrained, switching the test statistic from
              :math:`\tilde{q}_\mu` to :math:`q_\mu`.
            calculator (``'asymptotic'``, ``'toy'`` or ``'chi_square'``, default ``'asymptotic'``):

              * ``"asymptotic"``: Asymptotic formulae from :xref:`1007.1727`.
              * ``"toy"``: Pseudo-experiment-based p-values (requires
                :attr:`is_toy_calculator_available`).
              * ``"chi_square"``: :math:`\chi^2`-based p-values; uses
                :math:`\chi^2 = -2\log[\mathcal{L}(\mu,\hat\theta_\mu)/\mathcal{L}(0,\hat\theta_0)]`.

            kwargs: Additional keyword arguments forwarded to the optimiser, including:

              * **init_pars** (``List[float]``, default ``None``): Initial parameter
                values for the optimiser.
              * **par_bounds** (``List[Tuple[float, float]]``, default ``None``):
                Parameter bounds for the optimiser.

        Raises:
            :obj:`~spey.system.exceptions.CalculatorNotAvailable`:
                If the requested ``calculator`` is not available.

        Returns:
            ``List[float]``:
                :math:`CL_s` value(s).  One value for
                :obj:`~spey.ExpectationType.observed`; five values ordered
                :math:`(-2\sigma,\,-1\sigma,\,\text{central},\,+1\sigma,\,+2\sigma)` for
                expected modes.

        """
        if not getattr(self, f"is_{calculator}_calculator_available", False):
            raise CalculatorNotAvailable(f"{calculator} calculator is not available.")

        test_stat = "q" if allow_negative_signal else "qtilde"
        verbose = kwargs.pop("verbose", True)

        # NOTE Improve code efficiency, these are not necessary for asymptotic calculator
        if calculator in ["toy", "chi_square"]:
            test_stat_func = get_test_statistic(test_stat)

            def logpdf(
                mu: Union[float, np.ndarray], data: Union[float, np.ndarray]
            ) -> float:
                """Compute logpdf with respect to poi and given data"""
                return -self.likelihood(
                    poi_test=float(mu) if isinstance(mu, (float, int)) else mu[0],
                    expected=expected,
                    data=data,
                    **kwargs,
                )

            def maximize_likelihood(
                data: Union[float, np.ndarray]
            ) -> Tuple[float, float]:
                """Compute maximum likelihood with respect to given data"""
                return self.maximize_likelihood(
                    expected=expected,
                    allow_negative_signal=allow_negative_signal,
                    data=data,
                    **kwargs,
                )

            muhat, min_negloglike = maximize_likelihood(None)

        if calculator == "asymptotic":
            (
                maximum_likelihood,
                logpdf,
                maximum_asimov_likelihood,
                logpdf_asimov,
            ) = self._prepare_for_hypotest(
                expected=expected,
                test_statistics=test_stat,
                **kwargs,
            )

            try:
                log.debug("[asymptotic] - Computing test statistic")
                log.debug(
                    f"[asymptotic] - {maximum_likelihood=}, {maximum_asimov_likelihood=}"
                )
                _, sqrt_qmuA, delta_teststat = compute_teststatistics(
                    poi_test,
                    maximum_likelihood,
                    logpdf,
                    maximum_asimov_likelihood,
                    logpdf_asimov,
                    test_stat,
                )

                pvalues, expected_pvalues = compute_asymptotic_confidence_level(
                    sqrt_qmuA, delta_teststat, test_stat
                )
                log.debug(
                    f"[asymptotic] pval = {pvalues}, expected pval = {expected_pvalues}"
                )
            except AsimovTestStatZero as err:
                log.error(str(err))
                pvalues, expected_pvalues = [1.0], [1.0] * 5

        elif calculator == "toy":
            signal_samples = self.fixed_poi_sampler(
                poi_test=poi_test, size=self.ntoys, expected=expected, **kwargs
            )

            bkg_samples = self.fixed_poi_sampler(
                poi_test=0.0, size=self.ntoys, expected=expected, **kwargs
            )

            signal_like_test_stat, bkg_like_test_stat = [], []
            with tqdm.tqdm(
                total=self.ntoys,
                unit="toy sample",
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
                disable=not verbose,
            ) as pbar:
                for sig_smp, bkg_smp in zip(signal_samples, bkg_samples):
                    muhat_s_b, min_negloglike_s_b = maximize_likelihood(data=sig_smp)
                    signal_like_test_stat.append(
                        test_stat_func(
                            poi_test,
                            muhat_s_b,
                            -min_negloglike_s_b,
                            partial(logpdf, data=sig_smp),
                        )
                    )

                    muhat_b, min_negloglike_b = maximize_likelihood(data=bkg_smp)
                    bkg_like_test_stat.append(
                        test_stat_func(
                            poi_test,
                            muhat_b,
                            -min_negloglike_b,
                            partial(logpdf, data=bkg_smp),
                        )
                    )

                    pbar.update()

            pvalues, expected_pvalues = compute_toy_confidence_level(
                signal_like_test_stat,
                bkg_like_test_stat,
                test_statistic=test_stat_func(
                    poi_test, muhat, -min_negloglike, partial(logpdf, data=None)
                ),
                test_stat=test_stat,
            )
            pvalues, expected_pvalues = np.clip(pvalues, 0.0, 1.0), np.clip(
                expected_pvalues, 0.0, 1.0
            )

        elif calculator == "chi_square":
            ts_s_b = test_stat_func(
                poi_test, muhat, -min_negloglike, partial(logpdf, data=None)
            )
            null_logpdf = logpdf(0.0, None)
            max_logpdf = (
                -min_negloglike if muhat >= 0.0 or test_stat == "q" else null_logpdf
            )
            ts_b_only = np.clip(-2.0 * (null_logpdf - max_logpdf), 0.0, None)
            log.debug(
                f"<chi_square> test statistic: null hypothesis={ts_b_only}, s+b={ts_s_b}"
            )

            delta_ts = None
            sqrt_ts_s_b, sqrt_ts_b_only = np.sqrt(ts_s_b), np.sqrt(ts_b_only)
            if test_stat == "q" or sqrt_ts_s_b <= sqrt_ts_b_only:
                delta_ts = sqrt_ts_b_only - sqrt_ts_s_b
            else:
                try:
                    delta_ts = (ts_b_only - ts_s_b) / (2.0 * sqrt_ts_s_b)
                except ZeroDivisionError:
                    log.error(
                        "Lack of evidence for a signal or deviation from a null hypothesis."
                    )
            if delta_ts is not None:
                pvalues, expected_pvalues = compute_asymptotic_confidence_level(
                    sqrt_ts_s_b, delta_ts, test_stat=test_stat
                )
            else:
                pvalues, expected_pvalues = [1.0], [1.0] * 5

        if expected == "all":
            return list(map(lambda x: 1.0 - x, pvalues)), list(
                map(lambda x: 1.0 - x, expected_pvalues)
            )

        return list(
            map(
                lambda x: 1.0 - x if not np.isnan(x) else 0.0,
                pvalues if expected == ExpectationType.observed else expected_pvalues,
            )
        )

    def significance(
        self, expected: ExpectationType = ExpectationType.observed, **kwargs
    ) -> Tuple[float, float, List[float], List[float]]:
        r"""
        Compute the discovery significance of a positive signal.

        Uses the discovery test statistic :math:`q_0` (eq. 47 of :xref:`1007.1727`)
        to quantify the evidence for a signal above the background-only hypothesis.
        The Asimov significance :math:`\sqrt{q_{0,A}}` gives the median expected
        sensitivity, while :math:`\sqrt{q_0}` is computed from the observed data.
        See sec. 5.1 of :xref:`1007.1727` for details.

        .. note::

            :obj:`~spey.ExpectationType.aposteriori` and
            :obj:`~spey.ExpectationType.observed` both perform a post-fit computation
            and therefore return identical results.  The only meaningful distinction is
            between post-fit (:obj:`~spey.ExpectationType.observed`) and pre-fit
            (:obj:`~spey.ExpectationType.apriori`) computations.

        Args:
            expected (~spey.ExpectationType): Selects which dataset to condition on.

              * :obj:`~spey.ExpectationType.observed`: Post-fit (default).
              * :obj:`~spey.ExpectationType.apriori`: Pre-fit / SM hypothesis.

            kwargs: Additional keyword arguments forwarded to the optimiser, including:

              * **init_pars** (``List[float]``, default ``None``): Initial parameter
                values for the optimiser.
              * **par_bounds** (``List[Tuple[float, float]]``, default ``None``):
                Parameter bounds for the optimiser.

        Returns:
            ``Tuple[float, float, List[float], List[float]]``:
            A 4-tuple ``(sqrt_q0A, sqrt_q0, pvalues, expected_pvalues)`` where:

            * ``sqrt_q0A`` — Asimov discovery significance :math:`\sqrt{q_{0,A}}`.
            * ``sqrt_q0`` — Observed discovery significance :math:`\sqrt{q_0}`.
            * ``pvalues`` — Observed p-value(s) for the :math:`q_0` test.
            * ``expected_pvalues`` — Expected p-value(s) at
              :math:`-2\sigma,\,-1\sigma,\,\text{central},\,+1\sigma,\,+2\sigma`.
        """
        (
            maximum_likelihood,
            logpdf,
            maximum_asimov_likelihood,
            logpdf_asimov,
        ) = self._prepare_for_hypotest(expected=expected, test_statistics="q0", **kwargs)

        sqrt_q0, sqrt_q0A, delta_teststat = compute_teststatistics(
            0.0,
            maximum_likelihood,
            logpdf,
            maximum_asimov_likelihood,
            logpdf_asimov,
            "q0",
        )
        pvalues, expected_pvalues = compute_asymptotic_confidence_level(
            sqrt_q0A, delta_teststat, "q0"
        )

        return sqrt_q0A, sqrt_q0, pvalues, expected_pvalues

    @warning_tracker
    def poi_upper_limit(
        self,
        expected: ExpectationType = ExpectationType.observed,
        confidence_level: float = 0.95,
        low_init: float = 1.0,
        hig_init: float = 1.0,
        expected_pvalue: Literal["nominal", "1sigma", "2sigma"] = "nominal",
        maxiter: int = 10000,
        optimiser_arguments: Optional[Dict[str, Any]] = None,
    ) -> Union[float, List[float]]:
        r"""
        Compute the upper limit for the parameter of interest (POI), denoted as :math:`\mu`.

        Args:
            expected (:obj:`~spey.ExpectationType`, default :obj:`~spey.ExpectationType.observed`):
              Specifies the type of expectation for the fitting algorithm and p-value computation.

              * :obj:`~spey.ExpectationType.observed`: Computes p-values using post-fit prescription,
                assuming experimental data as the truth (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes expected p-values using post-fit
                prescription, assuming experimental data as the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes expected p-values using pre-fit
                prescription, assuming the Standard Model (SM) as the truth.

            confidence_level (``float``, default ``0.95``): Confidence level for the upper limit,
                representing :math:`1 - CL_s`. Must be between 0 and 1. Default is 0.95.
            low_init (``Optional[float]``, default ``1.0``): Initial lower limit for the search
              algorithm. If `None`, it is determined by :math:`\hat\mu + 1.5\sigma_{\hat\mu}`.
              Default is 1.0.

              .. note::

                :math:`\sigma_{\hat\mu}` is determined via
                :func:`~spey.base.hypotest_base.HypothesisTestingBase.sigma_mu` function.

            hig_init (``Optional[float]``, default ``1.0``): Initial upper limit for the search
              algorithm. If `None`, it is determined by :math:`\hat\mu + 2.5\sigma_{\hat\mu}`.
              Default is 1.0.

              .. note::

                :math:`\sigma_{\hat\mu}` is determined via
                :func:`~spey.base.hypotest_base.HypothesisTestingBase.sigma_mu` function.

            expected_pvalue (``Literal["nominal", "1sigma", "2sigma"]``, default ``"nominal"``):
              In case of :obj:`~spey.ExpectationType.aposteriori` and :obj:`~spey.ExpectationType.apriori`
              expectation, specifies the type of expected p-value for upper limit calculation.

              * ``"nominal"``: Computes the upper limit for the central p-value. Returns a single value.
              * ``"1sigma"``: Computes the upper limit for the central p-value and :math:`1\sigma`
                fluctuation from background. Returns 3 values.
              * ``"2sigma"``: Computes the upper limit for the central p-value and :math:`1\sigma`
                and :math:`2\sigma` fluctuation from background. Returns 5 values.

              .. note::

                For ``expected=spey.ExpectationType.observed``, ``expected_pvalue`` argument will
                be overwritten to ``"nominal"``.

            allow_negative_signal (``bool``, default ``True``): Allows for negative signal values,
                changing the computation of the test statistic. Default is False.
            maxiter (``int``, default ``10000``): Maximum number of iterations for the optimiser.
                Default is 10000.
            optimiser_arguments (``Dict``, default ``None``): Additional arguments for the optimiser
                used to compute the likelihood and its maximum. Default is None.

        Returns:
            ``Union[float, List[float]]``:

            - A single value representing the upper limit for the nominal case.
            - A list of values representing the upper limits for the central value and statistical
              deviations (for "1sigma" and "2sigma" cases). The order is: :math:`-2\sigma`,
              :math:`-1\sigma`, central value, :math:`1\sigma`, :math:`2\sigma`.

        Raises:
            AssertionError: If the confidence level is not between 0 and 1.
        """
        assert (
            0.0 <= confidence_level <= 1.0
        ), "Confidence level must be between zero and one."
        expected_pvalue = (
            "nominal" if expected == ExpectationType.observed else expected_pvalue
        )

        # If the signal yields in all regions are zero then return inf.
        # This means we are not able to set a bound with the given information.
        if not self.is_alive:
            return {"nominal": np.inf, "1sigma": [np.inf] * 3, "2sigma": [np.inf] * 5}[
                expected_pvalue
            ]

        optimiser_arguments = optimiser_arguments or {}

        with capture_logs(level=log.level):
            (
                maximum_likelihood,
                logpdf,
                maximum_asimov_likelihood,
                logpdf_asimov,
            ) = self._prepare_for_hypotest(
                expected=expected,
                test_statistics="qtilde",
                **optimiser_arguments,
            )

            if None in [low_init, hig_init]:
                muhat = maximum_likelihood[0] if maximum_likelihood[0] > 0.0 else 0.0
                sigma_mu = (
                    self.sigma_mu(muhat, expected=expected)
                    if not np.isclose(muhat, 0.0)
                    else 1.0
                )
                low_init = np.clip(low_init or muhat + 1.5 * sigma_mu, 1e-10, None)
                hig_init = np.clip(hig_init or muhat + 2.5 * sigma_mu, 1e-10, None)
                log.debug(f"new low_init = {low_init}, new hig_init = {hig_init}")

            return find_poi_upper_limit(
                maximum_likelihood=maximum_likelihood,
                logpdf=logpdf,
                maximum_asimov_likelihood=maximum_asimov_likelihood,
                asimov_logpdf=logpdf_asimov,
                expected=expected,
                confidence_level=confidence_level,
                allow_negative_signal=False,
                low_init=low_init,
                hig_init=hig_init,
                expected_pvalue=expected_pvalue,
                maxiter=maxiter,
            )

    def chi2_test(
        self,
        expected: ExpectationType = ExpectationType.observed,
        confidence_level: float = 0.95,
        limit_type: Literal["right", "left", "two-sided"] = "two-sided",
        allow_negative_signal: bool = None,
        parameter: Optional[Union[int, str]] = None,
        poi_value: float = 1.0,
    ) -> List[float]:
        r"""
        Determine parameter value(s) that constrain the :math:`\chi^2` distribution at a
        specified confidence level via 1D profiling.

        When ``parameter=None`` (default), the method profiles the primary POI and finds
        the POI values where the profile :math:`\chi^2` equals the threshold
        ``chi2.isf(alpha, df=1)``.

        When ``parameter`` is set to a nuisance parameter index or name, the POI is fixed
        to ``poi_value`` (default ``1.0``) and the method profiles the chosen nuisance
        parameter instead, locating the nuisance value(s) at the same :math:`\chi^2`
        threshold.  This is useful for setting 1D confidence intervals on any model
        parameter.

        .. versionadded:: 0.2.0

        .. attention::

            The degrees of freedom are set to one, referring to the single profiled
            parameter (either the POI or the selected nuisance parameter).

        Args:
            expected (~spey.ExpectationType): Specifies the type of expectation for the
              fitting algorithm and p-value computation.

              * :obj:`~spey.ExpectationType.observed`: Computes p-values using post-fit
                prescription, assuming experimental data as the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes expected p-values using
                pre-fit prescription, assuming the Standard Model (SM) as the truth.

            confidence_level (``float``, default ``0.95``): The confidence level for the
              interval.  Must be between 0 and 1.  This refers to the total inner area
              under the bell curve, noted as :math:`CL` below.

            limit_type (``'right'``, ``'left'`` or ``'two-sided'``, default
              ``"two-sided"``): Specifies which side of the :math:`\chi^2` distribution
              should be constrained.  For two-sided limits the inner area is set to
              ``confidence_level``, making the threshold :math:`\alpha=(1-CL)/2`.  For
              one-sided limits :math:`\alpha=1-CL`.  The :math:`\chi^2`-threshold is
              computed via the inverse survival function at :math:`\alpha`.

            allow_negative_signal (``bool``, default ``None``): Controls whether the POI
              can be negative during the global unconstrained maximisation.  If ``None``,
              it is set to ``True`` for two-sided and left limits, and ``False`` for right
              limits.  Ignored when ``parameter`` is not ``None`` (the global fit is
              always unconstrained in that case).

            parameter (``int`` or ``str``, default ``None``): Index or name of the
              nuisance parameter to profile.  When ``None`` (default) the primary POI is
              profiled (existing behaviour).  When set, the POI is fixed to ``poi_value``
              and the selected nuisance parameter is scanned instead.  String values are
              resolved via
              :attr:`~spey.base.model_config.ModelConfig.parameter_names`.

            poi_value (``float``, default ``1.0``): Fixed value of the primary POI when
              profiling a nuisance parameter (i.e. when ``parameter`` is not ``None``).
              Has no effect when ``parameter=None``.

        Returns:
            ``List[float]``:
            Parameter value(s) at which the profile :math:`\chi^2` equals the threshold.
            Returns one value for one-sided limits and two values for two-sided limits.

        Raises:
            :obj:`ValueError`: If ``parameter`` refers to the POI index, if the parameter
              name is not found in the model config, if the parameter index is out of
              range, or if the model has only one parameter (no nuisance parameters to
              profile).
        """
        assert (
            0.0 <= confidence_level <= 1.0
        ), "Confidence level must be between zero and one."
        assert limit_type in [
            "left",
            "right",
            "two-sided",
        ], f"Invalid limit type: {limit_type}"

        # Two-sided threshold halves alpha so the total inner area equals CL.
        alpha = (1.0 - confidence_level) * (0.5 if limit_type == "two-sided" else 1.0)
        chi2_threshold = chi2.isf(alpha, df=1)  # DoF = 1 (single profiled parameter)

        # ------------------------------------------------------------------ #
        # Build the profile computer and per-side bracket parameters          #
        # ------------------------------------------------------------------ #
        if parameter is None:
            # -- POI profiling (original behaviour) -------------------------
            allow_negative_signal = allow_negative_signal or limit_type in [
                "two-sided",
                "left",
            ]
            muhat, mllhd = self.maximize_likelihood(
                expected=expected, allow_negative_signal=allow_negative_signal
            )

            def computer(val: float) -> float:
                return (
                    2.0 * (self.likelihood(poi_test=val, expected=expected) - mllhd)
                    - chi2_threshold
                )

            try:
                sigma = self.sigma_mu(muhat, expected=expected)
            except MethodNotAvailable:
                sigma = 1.0

            is_gt0 = np.isclose(muhat, 0.0) or muhat > 0.0
            is_le0 = np.isclose(muhat, 0.0) or muhat < 0.0
            # Expand/contract via doubling (works for signed POI values).
            expand = lambda h: h * 2.0
            contract = lambda l: l * 0.5
            sides = {
                "left": dict(
                    inner=-1.0 if is_gt0 else muhat - 1.5 * sigma,
                    outer=-1.0 if is_gt0 else muhat - 2.5 * sigma,
                    expand=expand,
                    contract=contract,
                    outer_stop=lambda h: h <= -1e5,
                    inner_stop=lambda l: l >= -1e-5,
                    retry_inner=muhat if muhat > 0 else None,
                    retry_stop=(lambda l: l <= 1e-5) if muhat > 0 else None,
                    fallback=lambda o: -1e5 if o <= -1e5 else np.nan,
                    warn="Cannot find the left root. Check your chi^2 distribution.",
                ),
                "right": dict(
                    inner=1.0 if is_le0 else muhat + 1.5 * sigma,
                    outer=1.0 if is_le0 else muhat + 2.5 * sigma,
                    expand=expand,
                    contract=contract,
                    outer_stop=lambda h: h >= 1e5,
                    inner_stop=lambda l: l <= 1e-5,
                    retry_inner=muhat if muhat < 0 else None,
                    retry_stop=(lambda l: l >= -1e-5) if muhat < 0 else None,
                    fallback=lambda o: 1e5 if o >= 1e5 else np.nan,
                    warn="Cannot find the right root. Check your chi^2 distribution.",
                ),
            }
        else:
            # -- Nuisance-parameter profiling --------------------------------
            backend = getattr(self, "backend", None)
            if backend is None:
                raise NotImplementedError(
                    "`chi2_test` is not available for this backend."
                )
            cfg = backend.config()  # pylint: disable = no-member
            param_idx = resolve_parameter_index(parameter, cfg)

            _, mllhd = self.maximize_likelihood(
                expected=expected, allow_negative_signal=True
            )
            theta_hat: float = self.maximize_likelihood(
                poi_indices=[parameter], expected=expected, allow_negative_signal=True
            )[0][parameter]

            cfg_lo, cfg_hi = cfg.suggested_bounds[param_idx]
            abs_lo: float = cfg_lo if cfg_lo is not None else -1e5
            abs_hi: float = cfg_hi if cfg_hi is not None else 1e5

            def computer(val: float) -> float:
                nll = self.likelihood(
                    poi_test={cfg.poi_index: poi_value, param_idx: val}, expected=expected
                )
                return 2.0 * (nll - mllhd) - chi2_threshold

            step_l = max(abs(theta_hat - abs_lo) * 0.5, 0.5)
            step_r = max(abs(abs_hi - theta_hat) * 0.5, 0.5)
            # Reflect outer through theta_hat each step → doubles distance from center.
            expand_sym = lambda h: 2.0 * h - theta_hat
            contract_sym = lambda l: (l + theta_hat) / 2.0
            sides = {
                "left": dict(
                    inner=theta_hat - step_l * 0.5,
                    outer=theta_hat - step_l,
                    expand=expand_sym,
                    contract=contract_sym,
                    outer_stop=lambda h: h <= abs_lo,
                    inner_stop=lambda l: l >= theta_hat - 1e-10,
                    retry_inner=None,
                    retry_stop=None,
                    fallback=lambda o: abs_lo if o <= abs_lo else np.nan,
                    warn="Cannot find the nuisance left root. Profile may be too flat.",
                ),
                "right": dict(
                    inner=theta_hat + step_r * 0.5,
                    outer=theta_hat + step_r,
                    expand=expand_sym,
                    contract=contract_sym,
                    outer_stop=lambda h: h >= abs_hi,
                    inner_stop=lambda l: l <= theta_hat + 1e-10,
                    retry_inner=None,
                    retry_stop=None,
                    fallback=lambda o: abs_hi if o >= abs_hi else np.nan,
                    warn="Cannot find the nuisance right root. Profile may be too flat.",
                ),
            }

        # ------------------------------------------------------------------ #
        # Solve for each requested side                                        #
        # ------------------------------------------------------------------ #
        side_order = {
            "left": ["left"],
            "right": ["right"],
            "two-sided": ["left", "right"],
        }
        results = []
        for side_name in side_order[limit_type]:
            s = sides[side_name]
            with capture_logs(level=logging.ERROR):
                _, outer_final, x0 = bracket_and_solve(
                    computer,
                    s["inner"],
                    s["outer"],
                    s["expand"],
                    s["contract"],
                    s["outer_stop"],
                    s["inner_stop"],
                    retry_inner=s["retry_inner"],
                    retry_stop=s["retry_stop"],
                    debug_tag=side_name,
                )
            if np.isnan(x0):
                log.warning(s["warn"])
                x0 = s["fallback"](outer_final)
            results.append(x0)

        return results
