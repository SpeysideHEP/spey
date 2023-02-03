from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable, List

from spey.hypothesis_testing.utils import hypothesis_test, find_poi_upper_limit
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

    def _prepare_for_hypotest(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Callable[[bool], Tuple[float, float]], Callable[[float, bool], float]]:
        maximize_likelihood = lambda isAsimov: self.maximize_likelihood(
            return_nll=True,
            expected=expected,
            allow_negative_signal=allow_negative_signal,
            isAsimov=isAsimov,
            **kwargs,
        )

        logpdf = lambda mu, isAsimov: -self.likelihood(
            poi_test=mu if isinstance(mu, float) else mu[0],
            expected=expected,
            return_nll=True,
            isAsimov=isAsimov,
            **kwargs,
        )

        return maximize_likelihood, logpdf

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

        maximize_likelihood, logpdf = self._prepare_for_hypotest(
            expected=expected, allow_negative_signal=allow_negative_signal, **kwargs
        )
        pvalues, expected_pvalues = hypothesis_test(
            poi_test, maximize_likelihood, logpdf, allow_negative_signal
        )

        return list(
            map(
                lambda x: 1.0 - x,
                pvalues if expected == ExpectationType.observed else expected_pvalues,
            )
        )

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

        maximize_likelihood, logpdf = self._prepare_for_hypotest(
            expected=expected, allow_negative_signal=allow_negative_signal, **kwargs
        )

        return find_poi_upper_limit(
            maximize_likelihood=maximize_likelihood,
            logpdf=logpdf,
            expected=expected,
            sigma_mu=1.0,
            confidence_level=confidence_level,
            allow_negative_signal=allow_negative_signal,
        )
