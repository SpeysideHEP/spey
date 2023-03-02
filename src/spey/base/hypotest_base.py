from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable, List, Text, Union
import numpy as np

from spey.hypothesis_testing.utils import (
    compute_teststatistics,
    find_poi_upper_limit,
    compute_confidence_level,
)
from spey.hypothesis_testing.test_statistics import get_test_statistic
from spey.utils import ExpectationType


class HypothesisTestingBase(ABC):
    """Abstract class for accomodating hypothesis testing interface"""

    @abstractmethod
    def likelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        **kwargs,
    ) -> float:
        """Compute likelihood"""

    @abstractmethod
    def maximize_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = True,
        **kwargs,
    ) -> Tuple[float, float]:
        """Compute maximum  likelihood"""

    @abstractmethod
    def asimov_likelihood(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        return_nll: bool = True,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> float:
        """Compute likelihood for the asimov data"""

    @abstractmethod
    def maximize_asimov_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> Tuple[float, float]:
        """Compute maximum likelihood for asimov data"""

    def chi2(
        self,
        poi_test: float = 1.0,
        expected: ExpectationType = ExpectationType.observed,
        allow_negative_signal: bool = True,
        **kwargs,
    ) -> float:
        r"""
        Compute $$\chi^2$$

        .. math::

            \chi^2 = -2\log\left(\frac{\mathcal{L}_{\mu = 1}}{\mathcal{L}_{max}}\right)

        :param poi_test: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, allow negative mu
        :param isAsimov: if true, computes likelihood for Asimov data
        :return: chi^2
        """
        return 2.0 * (
            self.likelihood(poi_test=poi_test, expected=expected, **kwargs)
            - self.maximize_likelihood(
                expected=expected, allow_negative_signal=allow_negative_signal, **kwargs
            )[-1]
        )

    def _prepare_for_hypotest(
        self,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> Tuple[
        Callable[[], Tuple[float, float]],
        Callable[[float], float],
        Callable[[Text], Tuple[float, float]],
        Callable[[float, Text], float],
    ]:
        """
        Prepare necessary functions for hypothesis testing

        :param expected (`ExpectationType`, default `ExpectationType.observed`): _description_.
        :return `Tuple[ Callable[[], Tuple[float, float]],
        Callable[[float], float],
        Callable[[Text], Tuple[float, float]],
        Callable[[float, Text], float], ]`: _description_
        """
        allow_negative_signal = True if test_statistics in ["q" or "qmu"] else False

        muhat, nll = self.maximize_likelihood(
            expected=expected, allow_negative_signal=allow_negative_signal, **kwargs
        )
        muhatA, nllA = self.maximize_asimov_likelihood(
            expected=expected, test_statistics=test_statistics, **kwargs
        )

        def logpdf(mu: Union[float, np.ndarray]) -> float:
            return -self.likelihood(
                poi_test=mu if isinstance(mu, float) else mu[0],
                expected=expected,
                **kwargs,
            )

        def logpdf_asimov(mu: Union[float, np.ndarray]) -> float:
            return -self.asimov_likelihood(
                poi_test=mu if isinstance(mu, float) else mu[0],
                expected=expected,
                test_statistics=test_statistics,
                **kwargs,
            )

        return (muhat, nll), logpdf, (muhatA, nllA), logpdf_asimov

    def sigma_mu(
        self,
        poi_test: float,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qmu",
        **kwargs,
    ) -> float:
        r"""
        Estimation of `\sigma_{\mu}` denoted as `\sigma_A` where

        .. math::

            \sigma^2_A = \frac{\mu^2}{q_{\mu,A}}

        see eq. (31) in https://arxiv.org/abs/1007.1727

        :param poi_test (`float`): Parameter of interest
        :param expected (`ExpectationType`, default `ExpectationType.observed`):
                                                                observed, apriori or aposteriori.
        :param test_statistics (`Text`, default `"qmu"`): sets which test statistics to be used
                                                          i.e. `"qmu"`, `"qtilde"` or `"q0"`.
        :return `float`: deviation in POI
        """
        teststat_func = get_test_statistic(test_statistics)

        muhatA, min_nllA = self.maximize_asimov_likelihood(
            expected=expected, test_statistics=test_statistics, **kwargs
        )

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
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: bool = True,
        **kwargs,
    ) -> List[float]:
        """
        Compute exclusion confidence level of a given statistical model.

        :param poi_test: parameter of interest
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true muhat is allowed to be negative
        :param kwargs: backend specific inputs.
        :return: 1-CLs value (float)
        """
        test_stat = "q" if allow_negative_signal else "qmutilde"

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

        pvalues, expected_pvalues = compute_confidence_level(sqrt_qmuA, delta_teststat, test_stat)

        return list(
            map(
                lambda x: 1.0 - x,
                pvalues if expected == ExpectationType.observed else expected_pvalues,
            )
        )

    def significance(
        self, expected: Optional[ExpectationType] = ExpectationType.observed, **kwargs
    ) -> Tuple[float, float, List[float], List[float]]:
        """
        Compute the significance of the statistical model

        :param expected: observed, apriori or aposteriori
        :param kwargs: backend dependent arguments
        :return: sqrt(q0_A), sqrt(q0), pvalues, expected pvalues
        """
        (
            maximum_likelihood,
            logpdf,
            maximum_asimov_likelihood,
            logpdf_asimov,
        ) = self._prepare_for_hypotest(expected=expected, test_statistics="q0", **kwargs)

        sqrt_q0, sqrt_q0A, delta_teststat = compute_teststatistics(
            0.0, maximum_likelihood, logpdf, maximum_asimov_likelihood, logpdf_asimov, "q0"
        )
        pvalues, expected_pvalues = compute_confidence_level(sqrt_q0A, delta_teststat, "q0")

        return sqrt_q0A, sqrt_q0, pvalues, expected_pvalues

    def poi_upper_limit(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: bool = True,
        confidence_level: float = 0.95,
        low_init: Optional[float] = None,
        hig_init: Optional[float] = None,
        expected_pvalue: Text = "nominal",
        maxiter: int = 200,
        **kwargs,
    ) -> Union[float, List[float]]:
        r"""
        Compute the upper limit on parameter of interest, described by the confidence level

        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true muhat is allowed to be negative
        :param confidence_level: exclusion confidence level (default 1 - CLs = 95%)
        :param low_init (`Optional[float]`, default `None`): initialized lower bound for bracketing.
                                    if None its set to `$\hat{\mu} + 1.5\sigma_\mu$`
        :param hig_init (`Optional[float]`, default `None`): initialised upper bound for bracketing.
                                    if None its set to `$\hat{\mu} + 2.5\sigma_\mu$`
        :param expected_pvalue (`Text`, default `"nominal"`): find the upper limit for pvalue range,
                                                        only for expected. `nominal`, `1sigma`, `2sigma`
        :param maxiter (`int`, default `200`): If convergence is not achieved in maxiter iterations,
                                           an error is raised. Must be >= 0.
        :param kwargs: backend specific inputs.
        :return: excluded parameter of interest
        """
        assert 0.0 <= confidence_level <= 1.0, "Confidence level must be between zero and one."
        test_stat = "q" if allow_negative_signal else "qmutilde"

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

        if None in [low_init, hig_init]:
            muhat = maximum_likelihood[0] if maximum_likelihood[0] > 0.0 else 0.0
            sigma_mu = self.sigma_mu(muhat, expected=expected) if muhat != 0.0 else 1.0
            low_init = muhat + 1.5 * sigma_mu if not low_init else low_init
            hig_init = muhat + 2.5 * sigma_mu if not hig_init else hig_init

        return find_poi_upper_limit(
            maximum_likelihood=maximum_likelihood,
            logpdf=logpdf,
            maximum_asimov_likelihood=maximum_asimov_likelihood,
            asimov_logpdf=logpdf_asimov,
            expected=expected,
            confidence_level=confidence_level,
            allow_negative_signal=allow_negative_signal,
            low_init=low_init,
            hig_init=hig_init,
            expected_pvalue=expected_pvalue,
            maxiter=maxiter,
        )
