from typing import Optional, Tuple
import numpy as np

from spey.base.backend_base import BackendBase
from .data import Data, expansion_output
from .utils import fit, twice_nll
from .utils_marginalised import marginalised_negloglikelihood
from spey.utils import ExpectationType
from spey.backends import AvailableBackends
from spey.base.recorder import Recorder
from spey.interface.statistical_model import statistical_model_wrapper

__all__ = ["SimplifiedLikelihoodInterface"]


@statistical_model_wrapper
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
            str(ExpectationType.observed): None,
            str(ExpectationType.apriori): None,
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

    def _get_asimov_data(
        self, model: Data, expected: Optional[ExpectationType] = ExpectationType.observed
    ) -> Data:
        asimov_nuisance_key = (
            str(ExpectationType.apriori)
            if expected == ExpectationType.apriori
            else str(ExpectationType.observed)
        )
        pars = self._asimov_nuisance.get(asimov_nuisance_key, None)
        # NOTE for test_stat = q0 asimov mu should be 1, default qtilde!!!
        if pars is None:
            # Generate the asimov data by fittin nuissance parameters to the observations
            init_pars = [0.0] * (len(model) + 1)
            par_bounds = [(model.minimum_poi_test, 1.0)] + [(-5.0, 5.0)] * len(model)
            nll, pars = fit(model, init_pars, par_bounds, 0.0, self.third_moment_expansion)
            self._asimov_nuisance[asimov_nuisance_key] = pars

        return model.reset_observations(model.background + pars[1:], f"{model.name}_asimov")

    def logpdf(
        self,
        nuisance_parameters: np.ndarray,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        isAsimov: Optional[bool] = False,
    ) -> float:
        """
        Compute the log value of the full density.

        :param nuisance_parameters: nuisance parameters
        :param expected: observed, apriori or aposteriori
        :param isAsimov: if true, computes likelihood for Asimov data
        :return: negative log-likelihood
        """
        current_model: Data = (
            self.model if expected != ExpectationType.apriori else self.model.expected_dataset
        )
        if isAsimov:
            current_model = self._get_asimov_data(current_model, expected)

        return -0.5 * twice_nll(
            nuisance_parameters,
            signal=current_model.signal,
            background=current_model.background,
            observed=current_model.observed,
            third_moment_expansion=self.third_moment_expansion,
        )

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
                current_model = self._get_asimov_data(current_model, expected)

            if marginalize:
                nll = marginalised_negloglikelihood(
                    poi_test, current_model, self.third_moment_expansion, self.ntoys
                )
            else:
                init_pars = [poi_test] + [0.0] * len(current_model)
                par_bounds = [(current_model.minimum_poi_test, 40.0)] + [(-5.0, 5.0)] * len(
                    current_model
                )
                nll, pars = fit(
                    current_model, init_pars, par_bounds, poi_test, self.third_moment_expansion
                )

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
            pars, nll = self._recorder.get_maximum_likelihood(expected)
        else:
            current_model: Data = (
                self.model if expected != ExpectationType.apriori else self.model.expected_dataset
            )
            if isAsimov:
                current_model = self._get_asimov_data(current_model, expected)

            # It is possible to allow user to modify the optimiser properties in the future
            init_pars = [0.0] * (len(current_model) + 1)
            par_bounds = [
                (current_model.minimum_poi_test if allow_negative_signal else 0.0, 10.0)
            ] + [(-5.0, 5.0)] * len(current_model)
            nll, pars = fit(current_model, init_pars, par_bounds, None, self.third_moment_expansion)

            if not isAsimov:
                self._recorder.record_maximum_likelihood(expected, pars, nll)

        return pars[0], nll if return_nll else np.exp(-nll)

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
