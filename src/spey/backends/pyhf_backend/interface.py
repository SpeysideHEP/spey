import copy, logging, scipy, warnings, pyhf
from typing import Dict, Union, Optional, Tuple, List
import numpy as np

from pyhf.infer.calculators import generate_asimov_data

from spey.utils import ExpectationType
from spey.base.backend_base import BackendBase
from .utils import fixed_poi_fit, compute_min_negloglikelihood
from .data import Data
from spey.backends import AvailableBackends
from spey.system.exceptions import NegativeExpectedYields
from spey.hypothesis_testing.utils_cls import find_root_limits
from spey.base.recorder import Recorder

pyhf.pdf.log.setLevel(logging.CRITICAL)
pyhf.workspace.log.setLevel(logging.CRITICAL)
pyhf.set_backend("numpy", precision="64b")


class PyhfInterface(BackendBase):
    """
    Pyhf Interface

    :param model: contains all the information regarding the regions, yields
    :raises AssertionError: if the input type is wrong.
    """

    __slots__ = ["_model", "_recorder", "_asimov_nuisance"]

    def __init__(self, model: Data):
        assert isinstance(model, Data), "Invalid statistical model."
        self._model = model
        self._recorder = Recorder()
        self._asimov_nuisance = {
            str(ExpectationType.observed): False,
            str(ExpectationType.apriori): False,
        }

    @property
    def model(self) -> Data:
        """Retrieve statistical model"""
        return self._model

    @property
    def type(self) -> AvailableBackends:
        return AvailableBackends.pyhf

    def _get_asimov_data(
        self,
        model: pyhf.pdf.Model,
        data: np.ndarray,
        expected: Optional[ExpectationType] = ExpectationType.observed,
    ) -> np.ndarray:
        """
        Generate asimov data for the statistical model (only valid for teststat = qtilde)

        :param model: statistical model
        :param data: data
        :param expected: observed, apriori or aposteriori
        :return: asimov data
        """
        asimov_nuisance_key = (
            str(ExpectationType.apriori)
            if expected == ExpectationType.apriori
            else str(ExpectationType.observed)
        )
        asimov_data = self._asimov_nuisance.get(asimov_nuisance_key, False)
        if asimov_data is False:
            asimov_data = generate_asimov_data(
                0.0,  # this is for test stat = qtilde!!!
                data,
                model,
                model.config.suggested_init(),
                model.config.suggested_bounds(),
                model.config.suggested_fixed(),
                return_fitted_pars=False,
            )
            self._asimov_nuisance[asimov_nuisance_key] = copy.deepcopy(asimov_data)
        return asimov_data

    def logpdf(
        self,
        poi_test: float,
        nuisance_parameters: np.ndarray,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        isAsimov: Optional[bool] = False,
    ) -> float:
        """
        Compute the log value of the full density.

        :param poi_test: parameter of interest
        :param nuisance_parameters: nuisance parameters
        :param expected: observed, apriori or aposteriori
        :param isAsimov:
        :return: negative log-likelihood
        """
        assert len(nuisance_parameters) == self.model.npar, (
            f"Parameters must be {self.model.npar} dimensional vector, "
            f"{len(nuisance_parameters)} dimensions has been given."
        )
        _, model, data = self.model(poi_test=1.0, expected=expected)
        poi_bounds = self.model.suggested_bounds[self.model.poi_index]

        if isAsimov:
            data = self._get_asimov_data(model, data, expected)

        if not poi_bounds[0] <= poi_test <= poi_bounds[1]:
            _, model, _ = self.model(poi_test=poi_test, expected=expected)
            poi_test = 1.0

        complete_pars = np.array(
            nuisance_parameters.tolist()[: model.config.poi_index]
            + [poi_test]
            + nuisance_parameters.tolist()[model.config.poi_index :]
        )

        return model.logpdf(complete_pars, data).astype(np.float32)[0]

    def likelihood(
        self,
        poi_test: Optional[float] = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        return_nll: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        return_theta: Optional[bool] = False,
        iteration_threshold: Optional[int] = 3,
        options: Optional[Dict] = None,
    ) -> Union[float, List[float]]:
        """
        Compute  likelihood of the given statistical model

        :param poi_test: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param return_nll: if true returns negative log-likelihood value
        :param return_theta: return fitted parameters
        :param isAsimov: if true, computes likelihood for Asimov data
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :param options: optimizer options where the default values are
                :param maxiter: maximum iterations (default 200)
                :param verbose: verbosity (default False)
                :param tolerance: Tolerance for termination. See specific optimizer
                                  for detailed meaning. (default None)
                :param solver_options: (dict) additional solver options. See
                                :func:`scipy.optimize.show_options` for additional options of
                                optimization solvers. (default {})
                :param method: optimisation method (default SLSQP)
                        Available methods are:
                        - 'Nelder-Mead' :ref:`(see here) <scipy.optimize.minimize-neldermead>`
                        - 'Powell'      :ref:`(see here) <scipy.optimize.minimize-powell>`
                        - 'CG'          :ref:`(see here) <scipy.optimize.minimize-cg>`
                        - 'BFGS'        :ref:`(see here) <scipy.optimize.minimize-bfgs>`
                        - 'Newton-CG'   :ref:`(see here) <scipy.optimize.minimize-newtoncg>`
                        - 'L-BFGS-B'    :ref:`(see here) <scipy.optimize.minimize-lbfgsb>`
                        - 'TNC'         :ref:`(see here) <scipy.optimize.minimize-tnc>`
                        - 'COBYLA'      :ref:`(see here) <scipy.optimize.minimize-cobyla>`
                        - 'SLSQP'       :ref:`(see here) <scipy.optimize.minimize-slsqp>`
                        - 'trust-constr':ref:`(see here) <scipy.optimize.minimize-trustconstr>`
                        - 'dogleg'      :ref:`(see here) <scipy.optimize.minimize-dogleg>`
                        - 'trust-ncg'   :ref:`(see here) <scipy.optimize.minimize-trustncg>`
                        - 'trust-exact' :ref:`(see here) <scipy.optimize.minimize-trustexact>`
                        - 'trust-krylov' :ref:`(see here) <scipy.optimize.minimize-trustkrylov>`
        :return: (float) likelihood
        """
        if (
            self._recorder.get_poi_test(expected, poi_test) is not False
            and not isAsimov
            and not return_theta
        ):
            returns = [self._recorder.get_poi_test(expected, poi_test)]
        else:
            _, model, data = self.model(poi_test=1.0, expected=expected)

            if isAsimov:
                data = self._get_asimov_data(model, data, expected)

            # CHECK THE MODEL BOUNDS!!
            # POI Test needs to be adjusted according to the boundaries for sake of convergence
            # see issue https://github.com/scikit-hep/pyhf/issues/620#issuecomment-579235311
            # comment https://github.com/scikit-hep/pyhf/issues/620#issuecomment-579299831
            new_poi_test = copy.deepcopy(poi_test)
            execute = True
            bounds = model.config.suggested_bounds()[model.config.poi_index]
            if not bounds[0] <= new_poi_test <= bounds[1]:
                try:
                    _, model, _ = self.model(poi_test=new_poi_test, expected=expected)
                    new_poi_test = 1.0
                except NegativeExpectedYields as err:
                    warnings.warn(
                        err.args[0] + f"\nSetting NLL({poi_test:.3f}) = inf.",
                        category=RuntimeWarning,
                    )
                    execute = False

            if execute:
                negloglikelihood, theta = fixed_poi_fit(
                    new_poi_test, data, model, iteration_threshold, options
                )
            else:
                negloglikelihood, theta = np.nan, np.nan

            if not isAsimov:
                self._recorder.record_poi_test(expected, poi_test, negloglikelihood)

            returns = [
                negloglikelihood
                if return_nll or np.isnan(negloglikelihood)
                else np.exp(-negloglikelihood)
            ]
            if return_theta:
                returns.append(theta)

        if len(returns) == 1:
            return returns[0]

        return returns

    def maximize_likelihood(
        self,
        return_nll: Optional[bool] = False,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        iteration_threshold: Optional[int] = 3,
    ) -> Tuple[float, float]:
        """
        Find the POI that maximizes the likelihood and the value of the maximum likelihood

        :param return_nll: if true, likelihood will be returned
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: allow negative POI
        :param isAsimov: if true, computes likelihood for Asimov data
        :param iteration_threshold: number of iterations to be held for convergence of the fit.
        :return: muhat, maximum of the likelihood
        """
        if self._recorder.get_maximum_likelihood(expected) is not False and not isAsimov:
            muhat, negloglikelihood = self._recorder.get_maximum_likelihood(expected)
        else:
            _, model, data = self.model(poi_test=1.0, expected=expected)
            if isAsimov:
                data = self._get_asimov_data(model, data, expected)

            muhat, negloglikelihood = compute_min_negloglikelihood(
                data, model, allow_negative_signal, iteration_threshold, self.model.minimum_poi_test
            )

            if not isAsimov:
                self._recorder.record_maximum_likelihood(expected, muhat, negloglikelihood)

        return muhat, negloglikelihood if return_nll else np.exp(-negloglikelihood)

    def chi2(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        isAsimov: Optional[bool] = False,
        **kwargs,
    ) -> float:
        """
        Compute $$\chi^2$$

        .. math::

            \chi^2 = -2\log\left(\frac{\mathcal{L}_{\mu = 1}}{\mathcal{L}_{max}}\right)

        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, allow negative mu
        :param isAsimov: if true, computes likelihood for Asimov data
        :return: chi^2
        """
        return 2.0 * (
            self.likelihood(
                poi_test=1.0,
                expected=expected,
                return_nll=True,
                isAsimov=isAsimov,
            )
            - self.maximize_likelihood(
                return_nll=True,
                expected=expected,
                allow_negative_signal=allow_negative_signal,
                isAsimov=isAsimov,
            )[1]
        )

    def exclusion_confidence_level(
        self,
        poi_test: float = 1.0,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        allow_negative_signal: Optional[bool] = True,
        iteration_threshold: int = 3,
    ) -> List[float]:
        """
        Compute exclusion confidence level.

        :param poi_test: POI (signal strength)
        :param expected: observed, apriori or aposteriori
        :param allow_negative_signal: if true, allow negative mu
        :param iteration_threshold: sets threshold on when to stop
        :return: 1 - CLs values
        """
        if not self.model.isAlive:
            return [-1] if expected == ExpectationType.observed else [-1] * 5

        def get_CLs(poi: float, model: pyhf.pdf.Model, data: np.ndarray, **keywordargs):
            try:
                CLs_obs, CLs_exp = pyhf.infer.hypotest(
                    poi,
                    data,
                    model,
                    test_stat="q" if allow_negative_signal else "qtilde",
                    par_bounds=keywordargs.get("bounds", model.config.suggested_bounds()),
                    return_expected_set=True,
                )

            except (AssertionError, pyhf.exceptions.FailedMinimization, ValueError) as err:
                logging.getLogger("MA5").debug(str(err))
                # dont use false here 1.-CLs = 0 can be interpreted as false
                return "update bounds"

            # if isnan(float(CLs_obs)) or any([isnan(float(x)) for x in CLs_exp]):
            #     return "update mu"
            CLs_obs = float(CLs_obs[0]) if isinstance(CLs_obs, (list, tuple)) else float(CLs_obs)

            return {
                "CLs_obs": [1.0 - CLs_obs],
                "CLs_exp": list(map(lambda x: float(1.0 - x), CLs_exp)),
            }

        _, model, data = self.model(poi_test=1.0, expected=expected)

        def update_bounds(bounds: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            current_bounds = []
            for idx, bound in enumerate(bounds):
                if idx != model.config.poi_index:
                    min_bound = bound[0] * 2.0 if bound[0] < 0.0 else 0.0
                    max_bound = bound[1] * 2.0
                    current_bounds.append((min_bound, max_bound))
                else:
                    min_bound = bound[0] - 5.0 if allow_negative_signal else 0.0
                    if allow_negative_signal and self.model.minimum_poi_test is not None:
                        min_bound = (
                            self.model.minimum_poi_test
                            if not np.isinf(self.model.minimum_poi_test)
                            else bound[0] - 5.0
                        )
                        current_bounds.append((min_bound, 2.0 * bound[1]))
            return current_bounds

        # pyhf can raise an error if the poi_test bounds are too stringent
        # they need to be updated dynamically.
        arguments = dict(bounds=model.config.suggested_bounds(), stats="qtilde")
        poi_test_bounds = model.config.suggested_bounds()[model.config.poi_index]
        poi_update = False
        if not poi_test_bounds[0] <= poi_test <= poi_test_bounds[1]:
            _, model, data = self.model(poi_test=poi_test, expected=expected)
            poi_update = True
        it = 0
        while True:
            CLs = get_CLs(1.0 if poi_update else poi_test, model, data, **arguments)
            if CLs == "update bounds" or np.isnan(CLs["CLs_obs"][0]):
                arguments["bounds"] = update_bounds(arguments["bounds"])
                it += 1
            else:
                break
            # hard limit on iteration required if it exceeds this value it means
            # Nsig >>>>> Nobs
            if it >= iteration_threshold:
                return [1.0] if expected == ExpectationType.observed else [1.0] * 5

        return CLs["CLs_obs" if expected == ExpectationType.observed else "CLs_exp"]

    def poi_upper_limit(
        self,
        expected: Optional[ExpectationType] = ExpectationType.observed,
        confidence_level: float = 0.95,
        allow_negative_signal: Optional[bool] = True,
        **kwargs,
    ) -> float:
        """
        Compute the POI where the signal is excluded with 95% CL

        :param expected: observed, apriori or aposteriori
        :param confidence_level: confidence level (default 95%)
        :param allow_negative_signal: if true, allow negative mu
        :return: mu
        """
        assert 0.0 <= confidence_level <= 1.0, "Confidence level must be between zero and one."

        def computer(poi_test: float) -> float:
            CLs = self.exclusion_confidence_level(
                expected=expected, poi_test=poi_test, allow_negative_signal=allow_negative_signal
            )
            return CLs[0 if expected == ExpectationType.observed else 2] - confidence_level

        low, hig = find_root_limits(computer, loc=0.0)

        return scipy.optimize.brentq(computer, low, hig, xtol=low / 100.0)
