"""
Abstract class for Hypothesis base structure. This class contains necessary
tools to compute exclusion limits and POI upper limits
"""

import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import tqdm
from scipy.optimize import toms748
from scipy.stats import chi2

from spey.hypothesis_testing.asymptotic_calculator import (
    compute_asymptotic_confidence_level,
)
from spey.hypothesis_testing.test_statistics import (
    compute_teststatistics,
    get_test_statistic,
)
from spey.hypothesis_testing.toy_calculator import compute_toy_confidence_level
from spey.hypothesis_testing.upper_limits import ComputerWrapper, find_poi_upper_limit
from spey.system.exceptions import (
    AsimovTestStatZero,
    CalculatorNotAvailable,
    MethodNotAvailable,
    warning_tracker,
)
from spey.utils import ExpectationType

__all__ = ["HypothesisTestingBase"]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=W1203,C0103,possibly-used-before-assignment


class HypothesisTestingBase(ABC):
    """
    Abstract class that ensures classes that are performing hypothesis teststing includes certain
    set of function to perform necessary computations. This class gives the ability to compute
    exclusion limits and upper limits for the class inherits it.

    Args:
        ntoys (``int``, default ``1000``): Number of toy samples for hypothesis testing.
          (Only used for toy-based hypothesis testing)
    """

    __slots__ = ["ntoys"]

    def __init__(self, ntoys: int = 1000):
        self.ntoys = ntoys
        """Number of toy samples for sample generator during hypothesis testing"""

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        # This method has to be a property

    @property
    @abstractmethod
    def is_asymptotic_calculator_available(self) -> bool:
        """Check if Asymptotic calculator is available for the backend"""
        # This method has to be a property

    @property
    @abstractmethod
    def is_toy_calculator_available(self) -> bool:
        """Check if Toy calculator is available for the backend"""
        # This method has to be a property

    @property
    @abstractmethod
    def is_chi_square_calculator_available(self) -> bool:
        """Check if chi-square calculator is available for the backend"""
        # This method has to be a property

    @abstractmethod
    def likelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        data: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs,
    ) -> float:
        r"""
        Compute likelihood of the statistical model

        Args:
            poi_test (:obj:`float`, default :obj:`1.0`): parameter of interest, :math:`\mu`.
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

            return_nll (:obj:`bool`, default :obj:`True`): If ``True`` returns negative log-likelihood,
              else likelihood value.
            data (``Union[List[float], np.ndarray]``, default ``None``): input data that to fit. If
              ``None`` data will be set according to ``expected`` input.
            kwargs: keyword arguments for the optimiser.

        Returns:
            :obj:`float`:
            value of likelihood at fixed :math:`\mu`.
        """

    @abstractmethod
    def maximize_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = True,
        data: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs,
    ) -> Tuple[float, float]:
        r"""
        Compute maximum of the likelihood.

        Args:
            return_nll (:obj:`bool`, default :obj:`True`): If ``True`` returns negative log-likelihood,
              else likelihood value.
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

            allow_negative_signal (:obj:`bool`, default :obj:`True`): If :obj:`True` :math:`\hat\mu`
              value will be allowed to be negative.
            data (``Union[List[float], np.ndarray]``, default ``None``): input data that to fit. If
              ``None`` data will be set according to ``expected`` input.
            kwargs: keyword arguments for the optimiser.

        Returns:
            :obj:`Tuple[float, float]`:
            value of :math:`\hat\mu` and maximum likelihood.
        """

    @abstractmethod
    def asimov_likelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        test_statistics: Literal["qtilde", "q", "q0"] = "qtilde",
        **kwargs,
    ) -> float:
        r"""
        Compute likelihood at fixed :math:`\mu` for Asimov data

        Args:
            poi_test (:obj:`float`, default :obj:`1.0`): parameter of interest, :math:`\mu`.
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

            return_nll (`bool`, default `True`): If ``True`` returns negative log-likelihood,
              else likelihood value.
            test_statistics (`Text`, default `"qtilde"`): test statistics.

              * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
                :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`
                (:func:`~spey.hypothesis_testing.test_statistics.qmu_tilde`).

                .. warning::

                    Note that this assumes that :math:`\hat\mu\geq0`, hence :obj:`allow_negative_signal`
                    assumed to be :obj:`False`. If this function has been executed by user, :obj:`spey`
                    assumes that this is taken care of throughout the external code consistently.
                    Whilst computing p-values or upper limit on :math:`\mu` through :obj:`spey` this
                    is taken care of automatically in the backend.

              * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`, see
                eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
              * ``'q0'``: performs the calculation using the discovery test statistic, see eq. (47)
                of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

            kwargs: keyword arguments for the optimiser.

        Returns:
            :obj:`float`:
            value of the likelihood.
        """

    @abstractmethod
    def maximize_asimov_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Literal["qtilde", "q", "q0"] = "qtilde",
        **kwargs,
    ) -> Tuple[float, float]:
        r"""
        Compute maximum of the likelihood for Asimov data.

        Args:
            return_nll (:obj:`bool`, default :obj:`True`): If ``True`` returns negative log-likelihood,
              else likelihood value.
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

            test_statistics (`Text`, default `"qtilde"`): test statistics.

              * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
                :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`
                (:func:`~spey.hypothesis_testing.test_statistics.qmu_tilde`).

                .. warning::

                    Note that this assumes that :math:`\hat\mu\geq0`, hence `allow_negative_signal`
                    assumed to be `False`. If this function has been executed by user, `spey`
                    assumes that this is taken care of throughout the external code consistently.
                    Whilst computing p-values or upper limit on :math:`\mu` through `spey` this
                    is taken care of automatically in the backend.

              * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`, see
                eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
              * ``'q0'``: performs the calculation using the discovery test statistic, see eq. (47)
                of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

            kwargs: keyword arguments for the optimiser.

        Returns:
            :obj:`Tuple[float, float]`:
            value of :math:`\hat\mu` and maximum likelihood.
        """

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
        raise NotImplementedError("This method has not been implemented")

    def chi2(
        self,
        poi_test: float = 1.0,
        poi_test_denominator: Optional[float] = None,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = False,
        **kwargs,
    ) -> float:
        r"""
        If ``poi_test_denominator=None`` computes

        .. math::

            \chi^2 = -2\log\left(\frac{\mathcal{L}(\mu,\theta_\mu)}{\mathcal{L}(\hat\mu,\hat\theta)}\right)

        else

        .. math::

            \chi^2 = -2\log\left(\frac{\mathcal{L}(\mu,\theta_\mu)}{\mathcal{L}(\mu_{\rm denom},\theta_{\mu_{\rm denom}})}\right)

        where :math:`\mu_{\rm denom}` is ``poi_test_denominator`` which is typically zero to compare signal
        model with the background only model.

        Args:
            poi_test (``float``, default ``1.0``): parameter of interest, :math:`\mu`.
            poi_test_denominator (``float``, default ``None``): parameter of interest for the denominator, :math:`\mu`.
                If ``None`` maximum likelihood will be computed.
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

        return 2.0 * (
            self.likelihood(poi_test=poi_test, expected=expected, **kwargs) - denominator
        )

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
        Prepare necessary computations for hypothesis testing

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

            test_statistics (`Text`, default `"qtilde"`): test statistics.

                * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
                :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`
                (:func:`~spey.hypothesis_testing.test_statistics.qmu_tilde`).

                .. warning::

                    Note that this assumes that :math:`\hat\mu\geq0`, hence `allow_negative_signal`
                    assumed to be `False`. If this function has been executed by user, :obj:`spey`
                    assumes that this is taken care of throughout the external code consistently.
                    Whilst computing p-values or upper limit on :math:`\mu` through `spey` this
                    is taken care of automatically in the backend.

                * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`, see
                eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
                * ``'q0'``: performs the calculation using the discovery test statistic, see eq. (47)
                of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

            kwargs: keyword arguments for the optimiser.

              * **init_pars** (``List[float]``, default ``None``): initial parameters for the optimiser
              * **par_bounds** (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
                the optimiser.

        Returns:
            :obj:`Tuple[ Tuple[float, float], Callable[[float], float], Tuple[float, float], Callable[[float], float]]`:
            (:math:`\hat\mu`, :math:`\arg\min(-\log\mathcal{L})`), :math:`\log\mathcal{L(\mu, \theta_\mu)}`,
            (:math:`\hat\mu_A`, :math:`\arg\min(-\log\mathcal{L}_A)`), :math:`\log\mathcal{L_A(\mu, \theta_\mu)}`
        """
        allow_negative_signal = test_statistics in ["q" or "qmu"]

        muhat, nll = self.maximize_likelihood(
            expected=expected,
            allow_negative_signal=allow_negative_signal,
            **kwargs,
        )
        log.debug(f"muhat: {muhat}, nll: {nll}")
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
        poi_test: float,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Literal["qtilde", "q", "q0"] = "qtilde",
        **kwargs,
    ) -> float:
        r"""
        If available, :math:`\sigma_\mu` will be computed through Hessian of negative log-likelihood
        see :func:`spey.StatisticalModel.sigma_mu_from_hessian` for details.
        However, if not available it will be estimated via :math:`q_{\mu,A}`

        .. math::

            \sigma^2_A = \frac{(\mu - \mu^\prime)^2}{q_{\mu,A}}\quad , \quad q_{\mu,A} = -2\ln\lambda_A(\mu)

        see eq. (31) in :xref:`1007.1727`

        Args:
            poi_test (:obj:`float`, default :obj:`1.0`): parameter of interest, :math:`\mu`.
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

            test_statistics (`Text`, default `"qtilde"`): test statistics.

              * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
                :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`
                (:func:`~spey.hypothesis_testing.test_statistics.qmu_tilde`).

                .. warning::

                    Note that this assumes that :math:`\hat\mu\geq0`, hence `allow_negative_signal`
                    assumed to be `False`. If this function has been executed by user, `spey`
                    assumes that this is taken care of throughout the external code consistently.
                    Whilst computing p-values or upper limit on :math:`\mu` through `spey` this
                    is taken care of automatically in the backend.

              * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`, see
                eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
              * ``'q0'``: performs the calculation using the discovery test statistic, see eq. (47)
                of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

            kwargs: keyword arguments for the optimiser.

              * **init_pars** (``List[float]``, default ``None``): initial parameters for the optimiser
              * **par_bounds** (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
                the optimiser.

        Returns:
            :obj:`float`:
            value of the variance on :math:`\mu`.
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
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = False,
        calculator: Literal["asymptotic", "toy", "chi_square"] = "asymptotic",
        **kwargs,
    ) -> List[float]:
        r"""
        Compute exclusion confidence level (:math:`CL_s`) at a given POI, :math:`\mu`.

        Args:
            poi_test (``float``, default ``1.0``): parameter of interest, :math:`\mu`.
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed. If ``expected`` is set to ``"all"`` it will return both p-values
              and expected p-values where the likelihood will be fitted to observations.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).

                .. note::

                    In case of :obj:`~spey.ExpectationType.observed`, function will return one value
                    which has been fit to the observed data.


              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.

                .. note::

                    In case of :obj:`~spey.ExpectationType.aposteriori`, function will return five value
                    for expected p-values which has been fit to the observed data. Values represent
                    :math:`1\sigma` and :math:`2\sigma` fluctuations from the background. The order of the
                    output order is :math:`-2\sigma` value, :math:`-1\sigma` value, central value,
                    :math:`1\sigma` and :math:`2\sigma` value.

              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.

                .. note::

                    In case of :obj:`~spey.ExpectationType.apriori`, function will return five value
                    for expected p-values which has been fit to the SM background. Values represent
                    :math:`1\sigma` and :math:`2\sigma` fluctuations from the background. The
                    output order is :math:`-2\sigma` value, :math:`-1\sigma` value, central value,
                    :math:`1\sigma` and :math:`2\sigma` value.

            allow_negative_signal (``bool``, default ``False``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.
            calculator (``Literal["asymptotic", "toy", "chi_square"]``, default ``"asymptotic"``):
              Chooses the computation basis for hypothesis testing

              * ``"asymptotic"``: Uses asymptotic hypothesis testing to compute p-values.
              * ``"toy"``: Uses generated toy samples to compute p-values.
              * ``"chi_square"``: Computes p-values via chi-square;
                :math:`\chi^2=-2\log\frac{\mathcal{L}(1,\theta_1)}{\mathcal{L}(0,\theta_0)}`.

            kwargs: keyword arguments for the optimiser.

              * **init_pars** (``List[float]``, default ``None``): initial parameters for the optimiser
              * **par_bounds** (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
                the optimiser.

        Raises:
          :obj:`~spey.system.exceptions.CalculatorNotAvailable`: If calculator is not available.

        Returns:
            ``List[float]``:
            Exclusion confidence level i.e. :math:`CL_s`.
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
                _, sqrt_qmuA, delta_teststat = compute_teststatistics(
                    poi_test,
                    maximum_likelihood,
                    logpdf,
                    maximum_asimov_likelihood,
                    logpdf_asimov,
                    test_stat,
                )
                log.debug(
                    f"<asymptotic> sqrt_qmuA = {sqrt_qmuA}, test statistic = {delta_teststat}"
                )

                pvalues, expected_pvalues = compute_asymptotic_confidence_level(
                    sqrt_qmuA, delta_teststat, test_stat
                )
                log.debug(f"pval = {pvalues}, expected pval = {expected_pvalues}")
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
        Compute the discovery of a positive signal. See :xref:`1007.1727` eq. (53).
        and sec. 5.1.

        Args:
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.

                .. note::

                    Since :obj:`~spey.ExpectationType.aposteriori` and :obj:`~spey.ExpectationType.observed`
                    are both represent post-fit computation the result will be the same. The only difference
                    can be seen via prefit, :obj:`~spey.ExpectationType.apriori`, computation.

            kwargs: keyword arguments for the optimiser.

              * **init_pars** (``List[float]``, default ``None``): initial parameters for the optimiser
              * **par_bounds** (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
                the optimiser.

        Returns:
            ``Tuple[float, float, List[float], List[float]]``:
            (:math:`\sqrt{q_{0,A}}`, :math:`\sqrt{q_0}`, p-values and expected p-values)
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
    ) -> List[float]:
        r"""
        Determine the parameter of interest (POI) value(s) that constrain the
        :math:`\chi^2` distribution at a specified confidence level.

        .. versionadded:: 0.2.0

        .. attention::

            The degrees of freedom are set to one, referring to the POI. Currently, spey does not
            support multiple POIs, but this feature is planned for future releases.

        Args:
            expected (~spey.ExpectationType): Specifies the type of expectation for the fitting
              algorithm and p-value computation.

              * :obj:`~spey.ExpectationType.observed`: Computes p-values using post-fit prescription,
                assuming experimental data as the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes expected p-values using pre-fit
                prescription, assuming the Standard Model (SM) as the truth.

            confidence_level (``float``, default ``0.95``): The confidence level for the upper limit.
              Must be between 0 and 1. This refers to the total inner area under the bell curve. Noted
              as :math:`CL` below.

            limit_type (``'right'``, ``'left'`` or ``'two-sided'``, default ``"two-sided"``): Specifies
              which side of the :math:`\chi^2` distribution should be constrained. For two-sided limits,
              the inner area of the :math:`\chi^2` distribution is set to ``confidence_level``, making the
              threshold :math:`\alpha=(1-CL)/2`, where CL is the `confidence_level`. For left or right
              limits alone, :math:`\alpha=1-CL`. The :math:`\chi^2`-threshold is calculated using
              inverse survival function at :math:`\alpha`.

            allow_negative_signal (``bool``, default ``None``): Controls whether the signal can be
              negative. If ``None``, it will be set to ``True`` for two-sided and left limits, and
              ``False`` for right limits. Otherwise, user can control this behaviour.

        Returns:
            ``List[float]``:
            POI value(s) that constrain the :math:`\chi^2` distribution at the given threshold.
        """
        assert (
            0.0 <= confidence_level <= 1.0
        ), "Confidence level must be between zero and one."
        assert limit_type in [
            "left",
            "right",
            "two-sided",
        ], f"Invalid limit type: {limit_type}"

        # Two sided test statistic need to be halfed, total area within
        # two sides should be equal to confidence_level
        alpha = (1.0 - confidence_level) * (0.5 if limit_type == "two-sided" else 1.0)

        # DoF = # POI
        chi2_threshold = chi2.isf(alpha, df=1)
        allow_negative_signal = allow_negative_signal or limit_type in [
            "two-sided",
            "left",
        ]

        muhat, mllhd = self.maximize_likelihood(
            expected=expected, allow_negative_signal=allow_negative_signal
        )

        def computer(poi_test: float) -> float:
            """Compute chi^2 - chi^2 threshold"""
            llhd = self.likelihood(poi_test=poi_test, expected=expected)
            return 2.0 * (llhd - mllhd) - chi2_threshold

        try:
            sigma_muhat = self.sigma_mu(muhat, expected=expected)
        except MethodNotAvailable:
            sigma_muhat = 1.0

        results = []
        if limit_type in ["left", "two-sided"]:
            is_muhat_gt_0 = np.isclose(muhat, 0.0) or muhat > 0.0
            low = -1.0 if is_muhat_gt_0 else muhat - 1.5 * sigma_muhat
            hig = -1.0 if is_muhat_gt_0 else muhat - 2.5 * sigma_muhat
            hig_bound, low_bound = -1e5, -1e-5

            hig_computer = ComputerWrapper(computer)
            while hig_computer(hig) < 0.0 and hig > hig_bound:
                hig *= 2.0

            low_computer = ComputerWrapper(computer)
            while low_computer(low) > 0.0 and low < low_bound:
                low *= 0.5

            log.debug(
                f"Left first attempt:: low: f({low:.5e})={low_computer[-1]:.5e},"
                f" high: f({hig:.5e})={hig_computer[-1]:.5e}"
            )
            if np.sign(low_computer[-1]) == np.sign(hig_computer[-1]) and muhat > 0:
                low_computer = ComputerWrapper(computer)
                low = muhat
                while low_computer(low) > 0.0 and low > 1e-5:
                    low *= 0.5
                log.debug(
                    f"Left second attempt:: low: f({low:.5e})={low_computer[-1]:.5e},"
                    f" high: f({hig:.5e})={hig_computer[-1]:.5e}"
                )

            if np.sign(low_computer[-1]) != np.sign(hig_computer[-1]):
                x0, _ = toms748(
                    computer,
                    hig,
                    low,
                    k=2,
                    xtol=2e-12,
                    rtol=1e-4,
                    full_output=True,
                    maxiter=10000,
                )
                results.append(x0)
            else:
                log.error(
                    "Can not find the roots on the left side."
                    " Please check your chi^2 distribution, it might be too wide."
                )
                results.append(-1e5 if hig >= -1e5 else np.nan)

        if limit_type in ["right", "two-sided"]:
            is_muhat_le_0 = np.isclose(muhat, 0.0) or muhat < 0.0
            low = 1.0 if is_muhat_le_0 else muhat + 1.5 * sigma_muhat
            hig = 1.0 if is_muhat_le_0 else muhat + 2.5 * sigma_muhat
            hig_bound, low_bound = 1e5, 1e-5

            hig_computer = ComputerWrapper(computer)
            while hig_computer(hig) < 0.0 and hig < hig_bound:
                hig *= 2.0

            low_computer = ComputerWrapper(computer)
            while low_computer(low) > 0.0 and low > low_bound:
                low *= 0.5

            log.debug(
                f"Right first attempt:: low: f({low:.5e})={low_computer[-1]:.5e},"
                f" high: f({hig:.5e})={hig_computer[-1]:.5e}"
            )

            if np.sign(low_computer[-1]) == np.sign(hig_computer[-1]) and muhat < 0:
                low_computer = ComputerWrapper(computer)
                low = muhat
                while low_computer(low) > 0.0 and low < -1e-5:
                    low *= 0.5
                log.debug(
                    f"Left second attempt:: low: f({low:.5e})={low_computer[-1]:.5e},"
                    f" high: f({hig:.5e})={hig_computer[-1]:.5e}"
                )

            if np.sign(low_computer[-1]) != np.sign(hig_computer[-1]):
                x0, _ = toms748(
                    computer,
                    low,
                    hig,
                    k=2,
                    xtol=2e-12,
                    rtol=1e-4,
                    full_output=True,
                    maxiter=10000,
                )
                results.append(x0)
            else:
                log.error(
                    "Can not find the roots on the right side."
                    " Please check your chi^2 distribution, it might be too wide."
                )
                results.append(1e5 if hig >= 1e5 else np.nan)

        return results
