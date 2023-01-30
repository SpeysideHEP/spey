from typing import Optional, Tuple, List
import numpy as np
import scipy, warnings

from spey.base.backend_base import BackendBase
from .data import Data, expansion_output
from .utils_theta import fixed_poi_fit
from .utils_marginalised import marginalised_negloglikelihood
from spey.tools.utils_cls import compute_confidence_level, find_root_limits, teststatistics
from spey.utils import ExpectationType
from spey.backends import AvailableBackends
from spey.base.recorder import Recorder

__all__ = ["SimplifiedLikelihoodInterface"]


class SimplifiedLikelihoodInterface(BackendBase):
    """
    Simplified Likelihood Interface

    :param model: contains all the information regarding the regions,
                  yields and correlation matrices
    :param ntoys: number of toy examples to run for the test statistics
    :raises AssertionError: if the input type is wrong.
    """

    __slots__ = ["_model", "ntoys", "_third_moment_expansion", "_recorder", "_asimov_nuisance"]

    def __init__(self, model: Data, ntoys: Optional[int] = 10000):
        assert isinstance(model, Data) and isinstance(ntoys, int), "Invalid statistical model."
        self._model = model
        self.ntoys = ntoys
        self._third_moment_expansion: Optional[expansion_output] = None
        self._recorder = Recorder()
        self._asimov_nuisance = {
            str(ExpectationType.observed): False,
            str(ExpectationType.apriori): False,
        }

    @property
    def model(self) -> Data:
        """Get statistical model"""
        return self._model

    @property
    def type(self) -> AvailableBackends:
        return AvailableBackends.simplified_likelihoods

    @property
    def third_moment_expansion(self) -> expansion_output:
        """Get third moment expansion"""
        if self._third_moment_expansion is None:
            self._third_moment_expansion = self.model.compute_expansion()
        return self._third_moment_expansion

    def likelihood(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        return_nll: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        marginalize: Optional[bool] = False,
        **kwargs,
    ) -> float:
        """
        Compute the likelihood for the statistical model with a given POI

        :param poi_test: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param return_nll: if true returns negative log-likelihood value
        :param isAsimov: if true, computes likelihood for Asimov data
        :param marginalize: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :return: (float) likelihood
        """
        if self._recorder.get_poi_test(expected, poi_test) is not False and not isAsimov:
            nll = self._recorder.get_poi_test(expected, poi_test)
        else:
            current_model: Data = (
                self.model if expected != ExpectationType.apriori else self.model.expected_dataset
            )
            if isAsimov:
                asimov_nuisance_key = (
                    str(ExpectationType.apriori)
                    if expected == ExpectationType.apriori
                    else str(ExpectationType.observed)
                )
                thetahat_mu0 = self._asimov_nuisance.get(asimov_nuisance_key, False)
                # NOTE for test_stat = q0 asimov mu should be 1, default qtilde!!!
                if thetahat_mu0 is False:
                    # Generate the asimov data by fittin nuissance parameters to the observations
                    nll0, thetahat_mu0 = fixed_poi_fit(
                        0.0, current_model, self.third_moment_expansion
                    )
                    self._asimov_nuisance[asimov_nuisance_key] = thetahat_mu0
                current_model = current_model.reset_observations(
                    np.clip(current_model.background + thetahat_mu0, 0.0, None),
                    f"{current_model.name}_asimov",
                )

            if marginalize:
                nll = marginalised_negloglikelihood(
                    poi_test, current_model, self.third_moment_expansion, self.ntoys
                )
            else:
                nll, theta_hat = fixed_poi_fit(poi_test, current_model, self.third_moment_expansion)

            if not isAsimov:
                self._recorder.record_poi_test(expected, poi_test, nll)

        return nll if return_nll else np.exp(-nll)

    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = True,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        marginalise: Optional[bool] = False,
        iteration_threshold: Optional[int] = 10000,
    ) -> Tuple[float, float]:
        """
        Minimize negative log-likelihood of the statistical model with respect to POI

        :param return_nll: if true returns negative log-likelihood value
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, allow negative mu
        :param isAsimov: if true, computes likelihood for Asimov data
        :param marginalise: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :return: POI that minimizes the negative log-likelihood, minimum negative log-likelihood
        :raises RuntimeWarning: if optimiser cant reach required precision
        """
        if self._recorder.get_maximum_likelihood(expected) is not False and not isAsimov:
            muhat, nll = self._recorder.get_maximum_likelihood(expected)
        else:
            negloglikelihood = lambda mu: self.likelihood(
                mu[0],
                expected=expected,
                return_nll=True,
                isAsimov=isAsimov,
                marginalize=marginalise,
            )

            muhat_init = np.random.uniform(
                self.model.minimum_poi_test if allow_negative_signal else 0.0, 10.0, (1,)
            )

            # It is possible to allow user to modify the optimiser properties in the future
            opt = scipy.optimize.minimize(
                negloglikelihood,
                muhat_init,
                method="SLSQP",
                bounds=[(self.model.minimum_poi_test if allow_negative_signal else 0.0, 40.0)],
                tol=1e-6,
                options={"maxiter": iteration_threshold},
            )

            if not opt.success:
                warnings.warn(
                    message="Optimiser was not able to reach required precision.",
                    category=RuntimeWarning,
                )

            nll, muhat = opt.fun, opt.x[0]
            if not allow_negative_signal and muhat < 0.0:
                muhat, nll = 0.0, negloglikelihood([0.0])

            if not isAsimov:
                self._recorder.record_maximum_likelihood(expected, muhat, nll)

        return muhat, nll if return_nll else np.exp(-nll)

    def chi2(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        marginalise: Optional[bool] = False,
    ) -> float:
        """
        Compute $$\chi^2$$

        .. math::

            \chi^2 = -2\log\left(\frac{\mathcal{L}_{\mu = 1}}{\mathcal{L}_{max}}\right)

        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, allow negative mu
        :param isAsimov: if true, computes likelihood for Asimov data
        :param marginalise: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :return: \chi^2
        """
        return 2.0 * (
            self.likelihood(
                poi_test=1.0,
                expected=expected,
                return_nll=True,
                isAsimov=isAsimov,
                marginalize=marginalise,
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

        :param expected: observed, apriori or aposteriori
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

        negloglikelihood = lambda poi_test: self.likelihood(
            poi_test[0], expected=expected, return_nll=True, marginalize=marginalise
        )
        negloglikelihood_asimov = lambda poi_test: self.likelihood(
            poi_test[0], expected=expected, return_nll=True, isAsimov=True, marginalize=marginalise
        )

        return min_nll_asimov, negloglikelihood_asimov, min_nll, negloglikelihood

    def exclusion_confidence_level(
        self,
        poi_test: float = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        marginalise: Optional[bool] = False,
        iteration_threshold: Optional[int] = 10000,
    ) -> List[float]:
        """
        Compute 1 - CLs value

        :param poi_test: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, allow negative mu
        :param marginalise: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :return: 1 - CLs
        """
        min_nll_asimov, negloglikelihood_asimov, min_nll, negloglikelihood = self._exclusion_tools(
            expected=expected,
            marginalise=marginalise,
            allow_negative_signal=allow_negative_signal,
            iteration_threshold=iteration_threshold,
        )

        test_stat = "qtilde" if allow_negative_signal else "q0"
        _, sqrt_qmuA, test_statistic = teststatistics(
            poi_test, negloglikelihood_asimov, min_nll_asimov, negloglikelihood, min_nll, test_stat
        )

        CLs = list(
            map(lambda x: 1.0 - x, compute_confidence_level(sqrt_qmuA, test_statistic, expected))
        )

        return CLs

    def poi_upper_limit(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        confidence_level: float = 0.95,
        allow_negative_signal: Optional[bool] = True,
        marginalise: Optional[bool] = False,
        iteration_threshold: Optional[int] = 10000,
    ) -> float:
        """
        Compute the upper limit on parameter of interest

        :param expected: observed, apriori or aposteriori
        :param confidence_level: confidence level (default 95%)
        :param allow_negative_signal: if true, allow negative mu
        :param marginalise: if true, marginalize the likelihood.
                            if false compute profiled likelihood
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :return: excluded POI value at 95% CLs
        """
        assert 0.0 <= confidence_level <= 1.0, "Confidence level must be between zero and one."

        min_nll_asimov, negloglikelihood_asimov, min_nll, negloglikelihood = self._exclusion_tools(
            expected=expected,
            marginalise=marginalise,
            allow_negative_signal=allow_negative_signal,
            iteration_threshold=iteration_threshold,
        )

        def computer(poi_test: float) -> float:
            _, sqrt_qmuA, test_statistic = teststatistics(
                poi_test,
                negloglikelihood_asimov,
                min_nll_asimov,
                negloglikelihood,
                min_nll,
                "qtilde" if allow_negative_signal else "q0",
            )
            CLs = list(
                map(
                    lambda x: 1.0 - x, compute_confidence_level(sqrt_qmuA, test_statistic, expected)
                )
            )
            return CLs[0 if expected == ExpectationType.observed else 2] - confidence_level

        low, hig = find_root_limits(computer, loc=0.0)

        return scipy.optimize.brentq(computer, low, hig, xtol=low / 100.0)
