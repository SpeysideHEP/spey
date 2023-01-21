import scipy

from typing import Optional

from .data import Data
from .likelihood_computer import LikelihoodComputer
from .utils_cls import compute_confidence_limit
from madstats.utils import ExpectationType


class UpperLimitComputerNew:
    """
    Toolset for computing exclusion limit for given statistical model

    :param model: contains all the information regarding the regions,
                  yields and correlation matrices
    :param ntoys: number of toy examples to run for the test statistics
    """

    __slots__ = "likelihood_computer"

    def __init__(self, model: Data, ntoys: Optional[int] = 30000):
        self.likelihood_computer: LikelihoodComputer = LikelihoodComputer(model, ntoys)

    @property
    def model(self) -> Data:
        """Return statistical model"""
        return self.likelihood_computer.model

    def _exclusion_tools(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        marginalise: Optional[bool] = False,
        allow_negative_signal: Optional[bool] = False,
        iteration_threshold: Optional[int] = 10000,
    ):
        """
        Compute tools needed for exclusion limit computation

        :param expected: observed, expected (true, apriori) or aposteriori
        :param marginalise: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :param allow_negative_signal: if true, allow negative mu
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        """
        muhat, min_nll = self.likelihood_computer.maximize_likelihood(
            return_nll=True,
            expected=expected,
            marginalise=marginalise,
            allow_negative_signal=allow_negative_signal,
            iteration_threshold=iteration_threshold,
        )

        muhat_asimov, min_nll_asimov = self.likelihood_computer.maximize_likelihood(
            return_nll=True,
            expected=expected,
            marginalise=marginalise,
            allow_negative_signal=allow_negative_signal,
            isAsimov=True,
            iteration_threshold=iteration_threshold,
        )

        negloglikelihood = lambda mu: self.likelihood_computer.likelihood(
            mu[0], expected=expected, return_nll=True, marginalize=marginalise
        )
        negloglikelihood_asimov = lambda mu: self.likelihood_computer.likelihood(
            mu[0], expected=expected, return_nll=True, marginalize=marginalise, isAsimov=True
        )

        return min_nll_asimov, negloglikelihood_asimov, min_nll, negloglikelihood

    def computeCLs(
        self,
        mu: float = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        marginalise: Optional[bool] = False,
        allow_negative_signal: Optional[bool] = False,
        iteration_threshold: Optional[int] = 10000,
    ) -> float:
        """
        Compute 1 - CLs value

        :param mu: POI (signal strength)
        :param expected: observed, expected (true, apriori) or aposteriori
        :param marginalise: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :param allow_negative_signal: if true, allow negative mu
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :return: 1 - CLs
        """
        min_nll_asimov, negloglikelihood_asimov, min_nll, negloglikelihood = self._exclusion_tools(
            expected=expected,
            marginalise=marginalise,
            allow_negative_signal=allow_negative_signal,
            iteration_threshold=iteration_threshold,
        )

        return 1.0 - compute_confidence_limit(
            mu, negloglikelihood_asimov, min_nll_asimov, negloglikelihood, min_nll
        )

    def computeUpperLimitOnMu(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        marginalise: Optional[bool] = False,
        allow_negative_signal: Optional[bool] = False,
        iteration_threshold: Optional[int] = 10000,
    ) -> float:
        """
        Compute the POI where the signal is excluded with 95% CL

        :param expected: observed, expected (true, apriori) or aposteriori
        :param marginalise: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :param allow_negative_signal: if true, allow negative mu
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :return: excluded POI value at 95% CLs
        """

        min_nll_asimov, negloglikelihood_asimov, min_nll, negloglikelihood = self._exclusion_tools(
            expected=expected,
            marginalise=marginalise,
            allow_negative_signal=allow_negative_signal,
            iteration_threshold=iteration_threshold,
        )

        computer = lambda mu: 0.05 - compute_confidence_limit(
            mu, negloglikelihood_asimov, min_nll_asimov, negloglikelihood, min_nll
        )

        low, hig = 1.0, 1.0
        while computer(low) > 0.95:
            low *= 0.1
            if low < 1e-10:
                break
        while computer(hig) < 0.95:
            hig *= 10.0
            if hig > 1e10:
                break

        return scipy.optimize.brentq(computer, low, hig, xtol=low / 100.0)
