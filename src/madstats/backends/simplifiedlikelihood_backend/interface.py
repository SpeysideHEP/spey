from madstats.base.backend_base import BackendBase
from .data import Data, expansion_output
from .utils_theta import compute_min_negloglikelihood_theta
from .utils_marginalised import marginalised_negloglikelihood
from madstats.tools.utils_cls import compute_confidence_level
from madstats.utils import ExpectationType
from madstats.backends import AvailableBackends

from typing import Optional, Tuple
import numpy as np
import scipy


class SimplifiedLikelihoodInterface(BackendBase):
    """
    Simplified Likelihood Interface

    :param model: contains all the information regarding the regions,
                  yields and correlation matrices
    :param ntoys: number of toy examples to run for the test statistics
    :raises AssertionError: if the input type is wrong.
    """

    __slots__ = "_model", "ntoys", "_third_moment_expansion"

    def __init__(self, model: Data, ntoys: Optional[int] = 30000):
        assert isinstance(model, Data) and isinstance(ntoys, int), "Invalid statistical model."
        self._model = model
        self.ntoys = ntoys
        self._third_moment_expansion: Optional[expansion_output] = None

    @property
    def model(self) -> Data:
        """Get statistical model"""
        return self._model

    @property
    def third_moment_expansion(self) -> expansion_output:
        """Get third moment expansion"""
        if self._third_moment_expansion is None:
            self._third_moment_expansion = self.model.compute_expansion()
        return self._third_moment_expansion

    @property
    def type(self) -> AvailableBackends:
        return AvailableBackends.simplified_likelihoods

    def likelihood(
        self,
        mu: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        return_nll: Optional[bool] = True,
        marginalize: Optional[bool] = False,
        isAsimov: Optional[bool] = False,
        **kwargs,
    ) -> float:
        """
        Compute the likelihood for the statistical model with a given POI

        :param mu: POI (signal strength)
        :param expected: observed, expected (true, apriori) or aposteriori
        :param return_nll: if true returns negative log-likelihood value
        :param marginalize: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :param isAsimov: if true, computes likelihood for Asimov data
        :return: (float) likelihood
        """
        current_model: Data = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )
        if isAsimov:
            # Generate the asimov data by fittin nuissance parameters to the observations
            nll0, thetahat_mu0 = compute_min_negloglikelihood_theta(
                0.0, current_model, self.third_moment_expansion
            )
            current_model = current_model.reset_observations(
                current_model.background + thetahat_mu0, f"{current_model.name}_asimov"
            )

        if marginalize:
            nll = marginalised_negloglikelihood(
                mu, current_model, self.third_moment_expansion, self.ntoys
            )
        else:
            nll, theta_hat = compute_min_negloglikelihood_theta(
                mu, current_model, self.third_moment_expansion
            )

        return nll if return_nll else np.exp(-nll)

    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = True,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        marginalise: Optional[bool] = False,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        iteration_threshold: Optional[int] = 10000,
    ) -> Tuple[float, float]:
        """
        Minimize negative log-likelihood of the statistical model with respect to POI

        :param return_nll: if true returns negative log-likelihood value
        :param expected: observed, expected (true, apriori) or aposteriori
        :param marginalise: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :param allow_negative_signal: if true, allow negative mu
        :param isAsimov: if true, computes likelihood for Asimov data
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :return: POI that minimizes the negative log-likelihood, minimum negative log-likelihood
        :raises RuntimeWarning: if optimiser cant reach required precision
        """
        negloglikelihood = lambda mu: self.likelihood(
            mu[0], expected=expected, return_nll=True, marginalize=marginalise, isAsimov=isAsimov
        )

        muhat_init = np.random.uniform(-10.0 if allow_negative_signal else 0.0, 10.0, (1,))

        # It is possible to allow user to modify the optimiser properties in the future
        opt = scipy.optimize.minimize(
            negloglikelihood,
            muhat_init,
            method="COBYLA",
            tol=1e-6,
            options={"maxiter": iteration_threshold},
        )

        if not opt.success:
            raise RuntimeWarning("Optimiser was not able to reach required precision.")

        nll, muhat = opt.fun, opt.x[0]
        if not allow_negative_signal and muhat < 0.0:
            muhat, nll = 0.0, negloglikelihood([0.0])

        return muhat, nll if return_nll else np.exp(-nll)

    def chi2(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        marginalise: Optional[bool] = False,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
    ) -> float:
        """
        Compute $$\chi^2$$

        .. math::

            \chi^2 = -2\log\left(\frac{\mathcal{L}_{\mu = 1}}{\mathcal{L}_{max}}\right)

        :param expected: observed, expected (true, apriori) or aposteriori
        :param marginalise: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :param allow_negative_signal: if true, allow negative mu
        :param isAsimov: if true, computes likelihood for Asimov data
        :return: \chi^2
        """
        return 2.0 * (
            self.likelihood(
                mu=1.0,
                expected=expected,
                return_nll=True,
                marginalize=marginalise,
                isAsimov=isAsimov,
            )
            - self.maximize_likelihood(
                return_nll=True,
                expected=expected,
                allow_negative_signal=allow_negative_signal,
                marginalise=marginalise,
                isAsimov=isAsimov,
            )[1]
        )

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
        muhat, min_nll = self.maximize_likelihood(
            return_nll=True,
            expected=expected,
            marginalise=marginalise,
            allow_negative_signal=allow_negative_signal,
            iteration_threshold=iteration_threshold,
        )

        muhat_asimov, min_nll_asimov = self.maximize_likelihood(
            return_nll=True,
            expected=expected,
            marginalise=marginalise,
            allow_negative_signal=allow_negative_signal,
            isAsimov=True,
            iteration_threshold=iteration_threshold,
        )

        negloglikelihood = lambda mu: self.likelihood(
            mu[0], expected=expected, return_nll=True, marginalize=marginalise
        )
        negloglikelihood_asimov = lambda mu: self.likelihood(
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

        return 1.0 - compute_confidence_level(
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

        computer = lambda mu: 0.05 - compute_confidence_level(
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
