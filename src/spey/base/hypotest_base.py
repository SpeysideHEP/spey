"""
Abstract class for Hypothesis base structure. This class contains necessary
tools to compute exclusion limits and POI upper limits
"""

import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Text, Tuple, Union

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
from spey.hypothesis_testing.upper_limits import find_poi_upper_limit
from spey.system.exceptions import CalculatorNotAvailable, MethodNotAvailable
from spey.utils import ExpectationType

__all__ = ["HypothesisTestingBase"]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=W1203,C0103


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
        allow_negative_signal = True if test_statistics in ["q" or "qmu"] else False

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

    def exclusion_confidence_level(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = False,
        calculator: Literal["asymptotic", "toy", "chi_square"] = "asymptotic",
        poi_test_denominator: Optional[float] = None,
        **kwargs,
    ) -> List[float]:
        r"""
        Compute exclusion confidence level (:math:`1-CL_s`) at a given POI, :math:`\mu`.

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

            poi_test_denominator (``float``, default ``None``): Set the POI value for the null hypothesis.
                if ``None``, signal hypothesis will be compared against maximum likelihood otherwise
                with respect to the hypothesis determined with the POI value provided with this input.
                Only used when ``calculator="chi-square"``.
            kwargs: keyword arguments for the optimiser.

              * **init_pars** (``List[float]``, default ``None``): initial parameters for the optimiser
              * **par_bounds** (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
                the optimiser.

        Raises:
          :obj:`~spey.system.exceptions.CalculatorNotAvailable`: If calculator is not available.

        Returns:
            ``List[float]``:
            Exclusion confidence level i.e. :math:`1-CL_s`.
        """
        if not getattr(self, f"is_{calculator}_calculator_available", False):
            raise CalculatorNotAvailable(f"{calculator} calculator is not available.")

        test_stat = "q" if allow_negative_signal else "qtilde"

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

        elif calculator == "toy":
            signal_samples = self.fixed_poi_sampler(
                poi_test=poi_test, size=self.ntoys, expected=expected, **kwargs
            )

            bkg_samples = self.fixed_poi_sampler(
                poi_test=0.0, size=self.ntoys, expected=expected, **kwargs
            )

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

            signal_like_test_stat, bkg_like_test_stat = [], []
            with tqdm.tqdm(
                total=self.ntoys,
                unit="toy sample",
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            ) as pbar:
                for sig_smp, bkg_smp in zip(signal_samples, bkg_samples):
                    signal_like_test_stat.append(
                        test_stat_func(
                            poi_test,
                            *maximize_likelihood(data=sig_smp),
                            partial(logpdf, data=sig_smp),
                        )
                    )

                    bkg_like_test_stat.append(
                        test_stat_func(
                            poi_test,
                            *maximize_likelihood(data=bkg_smp),
                            partial(logpdf, data=bkg_smp),
                        )
                    )

                    pbar.update()

            pvalues, expected_pvalues = compute_toy_confidence_level(
                signal_like_test_stat,
                bkg_like_test_stat,
                test_statistic=poi_test,
                test_stat=test_stat,
            )

        elif calculator == "chi_square":
            chi_square = self.chi2(
                poi_test=1.0,
                poi_test_denominator=poi_test_denominator,
                expected=expected,
                allow_negative_signal=allow_negative_signal,
                **kwargs,
            )

            pvalues = [
                1.0
                - chi2.cdf(
                    chi_square,
                    1 if isinstance(poi_test, (float, int)) else len(poi_test),
                )
            ]
            expected_pvalues = pvalues

            if expected in [ExpectationType.aposteriori, ExpectationType.apriori]:
                fit = "post" if expected == ExpectationType.aposteriori else "pre"
                log.warning(
                    "chi-square calculator does not support expected p-values."
                    f" Only one p-value for {fit}fit will be returned."
                )

        if expected == "all":
            return list(map(lambda x: 1.0 - x, pvalues)), list(
                map(lambda x: 1.0 - x, expected_pvalues)
            )

        return list(
            map(
                lambda x: 1.0 - x,
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

    def poi_upper_limit(
        self,
        expected: ExpectationType = ExpectationType.observed,
        confidence_level: float = 0.95,
        low_init: Optional[float] = 1.0,
        hig_init: Optional[float] = 1.0,
        expected_pvalue: Literal["nominal", "1sigma", "2sigma"] = "nominal",
        maxiter: int = 10000,
        optimiser_arguments: Optional[Dict[Text, Any]] = None,
    ) -> Union[float, List[float]]:
        r"""
        Compute the upper limit for the parameter of interest i.e. :math:`\mu`.

        .. note::

            This function uses ``"qtilde"`` test statistic which means signal values are always
            assumed to be positive i.e. :math:`\hat\mu>0`.

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

            confidence_level (``float``, default ``0.95``): Determines the confidence level of the upper
              limit i.e. the value of :math:`1-CL_s`. It needs to be between ``[0,1]``.
            low_init (``Optional[float]``, default ``1.0``): Lower limit for the search algorithm to start
              If ``None`` it the lower limit will be determined by :math:`\hat\mu + 1.5\sigma_{\hat\mu}`.

              .. note::

                :math:`\sigma_{\hat\mu}` is determined via
                :func:`~spey.base.hypotest_base.HypothesisTestingBase.sigma_mu` function.

            hig_init (``Optional[float]``, default ``1.0``): Upper limit for the search algorithm to start
              If ``None`` it the upper limit will be determined by :math:`\hat\mu + 2.5\sigma_{\hat\mu}`.

              .. note::

                :math:`\sigma_{\hat\mu}` is determined via
                :func:`~spey.base.hypotest_base.HypothesisTestingBase.sigma_mu` function.

            expected_pvalue (``Literal["nominal", "1sigma", "2sigma"]``, default ``"nominal"``):
              In case of :obj:`~spey.ExpectationType.aposteriori` and :obj:`~spey.ExpectationType.apriori`
              expectation, gives the choice to find excluded upper limit for statistical deviations as well.

              * ``"nominal"``: only find the upper limit for the central p-value. Returns a single value.
              * ``"1sigma"``: find the upper limit for central p-value and :math:`1\sigma` fluctuation from
                background. Returns 3 values.
              * ``"2sigma"``: find the upper limit for central p-value and :math:`1\sigma` and
                :math:`2\sigma` fluctuation from background. Returns 5 values.

              .. note::

                For ``expected=spey.ExpectationType.observed``, ``expected_pvalue`` argument will
                be overwritten to ``"nominal"``.

            maxiter (``int``, default ``10000``): Maximum iteration limit for the optimiser.
            optimiser_arguments (``Dict``, default ``None``): Arguments for optimiser that is used
              to compute likelihood and its maximum.

        Returns:
            ``Union[float, List[float]]``:
            In case of nominal values it returns a single value for the upper limit. In case of
            ``expected_pvalue="1sigma"`` or ``expected_pvalue="2sigma"`` it will return a list of
            multiple upper limit values for fluctuations as well as the central value. The
            output order is :math:`-2\sigma` value, :math:`-1\sigma` value, central value,
            :math:`1\sigma` and :math:`2\sigma` value.
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
            low_init = low_init or muhat + 1.5 * sigma_mu
            hig_init = hig_init or muhat + 2.5 * sigma_mu
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
