from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable, List, Text
from functools import partial

from spey.hypothesis_testing.utils import (
    compute_teststatistics,
    find_poi_upper_limit,
    compute_confidence_level,
)
from spey.utils import ExpectationType


class HypothesisTestingBase(ABC):
    @abstractmethod
    def likelihood(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        return_nll: Optional[bool] = True,
        **kwargs,
    ) -> float:
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = True,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[float, float]:
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
    def asimov_likelihood(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        return_nll: Optional[bool] = True,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> float:
        raise NotImplementedError("This method has not been implemented")

    @abstractmethod
    def maximize_asimov_likelihood(
        self,
        return_nll: bool = True,
        expected: ExpectationType = ExpectationType.observed,
        test_statistics: Text = "qtilde",
        **kwargs,
    ) -> Tuple[float, float]:
        raise NotImplementedError("This method has not been implemented")

    def chi2(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        **kwargs,
    ) -> float:
        """
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
        allow_negative_signal: bool = False,
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
        :param allow_negative_signal (`bool`, default `False`): _description_.
        :return `Tuple[ Callable[[], Tuple[float, float]],
        Callable[[float], float],
        Callable[[Text], Tuple[float, float]],
        Callable[[float, Text], float], ]`: _description_
        """

        muhat, nll = self.maximize_likelihood(
            expected=expected, allow_negative_signal=allow_negative_signal, **kwargs
        )
        muhatA, nllA = self.maximize_asimov_likelihood(
            expected=expected, test_statistics=test_statistics, **kwargs
        )

        logpdf = lambda mu: -self.likelihood(
            poi_test=mu if isinstance(mu, float) else mu[0],
            expected=expected,
            **kwargs,
        )

        logpdf_asimov = lambda mu: -self.asimov_likelihood(
            poi_test=mu if isinstance(mu, float) else mu[0],
            expected=expected,
            test_statistics=test_statistics,
            **kwargs,
        )

        return (muhat, nll), logpdf, (muhatA, nllA), logpdf_asimov

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
        if hasattr(self, "backend"):
            if hasattr(getattr(self, "backend"), "exclusion_confidence_level") and not kwargs.pop(
                "overwrite", False
            ):
                return getattr(getattr(self, "backend"), "exclusion_confidence_level")(
                    poi_test=poi_test,
                    expected=expected,
                    allow_negative_signal=allow_negative_signal,
                    **kwargs,
                )
        test_stat = "q" if allow_negative_signal else "qmutilde"

        (
            maximum_likelihood,
            logpdf,
            maximum_asimov_likelihood,
            logpdf_asimov,
        ) = self._prepare_for_hypotest(
            expected=expected,
            allow_negative_signal=allow_negative_signal,
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
        ) = self._prepare_for_hypotest(
            expected=expected, allow_negative_signal=False, test_statistics="q0", **kwargs
        )

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
        **kwargs,
    ) -> float:
        """
        Compute the upper limit on parameter of interest, described by the confidence level

        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true muhat is allowed to be negative
        :param confidence_level: exclusion confidence level (default 1 - CLs = 95%)
        :param kwargs: backend specific inputs.
        :return: excluded parameter of interest
        """
        assert 0.0 <= confidence_level <= 1.0, "Confidence level must be between zero and one."
        if hasattr(self, "backend"):
            if hasattr(getattr(self, "backend"), "poi_upper_limit") and not kwargs.pop(
                "overwrite", False
            ):
                return getattr(getattr(self, "backend"), "poi_upper_limit")(
                    expected=expected,
                    allow_negative_signal=allow_negative_signal,
                    confidence_level=confidence_level,
                    **kwargs,
                )

        test_stat = "q" if allow_negative_signal else "qmutilde"

        (
            maximum_likelihood,
            logpdf,
            maximum_asimov_likelihood,
            logpdf_asimov,
        ) = self._prepare_for_hypotest(
            expected=expected,
            allow_negative_signal=allow_negative_signal,
            test_statistics=test_stat,
            **kwargs,
        )

        return find_poi_upper_limit(
            maximum_likelihood=maximum_likelihood,
            logpdf=logpdf,
            maximum_asimov_likelihood=maximum_asimov_likelihood,
            asimov_logpdf=logpdf_asimov,
            expected=expected,
            confidence_level=confidence_level,
            allow_negative_signal=allow_negative_signal,
        )
